"""Kick-triggered sidechain ducking computed as an explicit gain curve.

pedalboard's ``Compressor`` has no external sidechain input, so ducking is rendered as a
volume-shaper style gain curve (as in LFOTool or Kickstart) and multiplied into the target.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from automix._types import Audio, SampleIndices

_FRAME_SECONDS = 256 / 48_000  # about 5 ms: long enough that a sub cycle never reads as a hit
_RISE_RATIO = 2.0  # a frame must be 6 dB above the previous one to count as a new hit


def detect_onsets(
    trigger: Audio, sr: int, threshold_db: float = -24.0, min_interval_s: float = 0.05
) -> SampleIndices:
    """Return sample indices where hits start in ``trigger`` (usually an isolated kick stem).

    A hit is a frame whose peak exceeds ``threshold_db`` relative to the loudest sample and
    rises sharply over the previous frame. Hits closer than ``min_interval_s`` are merged.
    """
    mono = np.max(np.abs(trigger), axis=0)
    loudest = float(mono.max()) if mono.size else 0.0
    if loudest == 0.0:
        return np.array([], dtype=np.int64)
    threshold = loudest * 10 ** (threshold_db / 20)

    frame = max(int(sr * _FRAME_SECONDS), 1)
    n_frames = len(mono) // frame + 1
    padded = np.pad(mono, (0, n_frames * frame - len(mono)))
    frame_peaks = padded.reshape(n_frames, frame).max(axis=1)
    previous = np.concatenate([[0.0], frame_peaks[:-1]])
    candidates = np.flatnonzero((frame_peaks >= threshold) & (frame_peaks > previous * _RISE_RATIO))

    min_gap = int(min_interval_s * sr)
    onsets: list[int] = []
    for index in candidates:
        start = int(index) * frame
        sample = start + int(np.argmax(padded[start : start + frame] >= threshold))
        if not onsets or sample - onsets[-1] >= min_gap:
            onsets.append(sample)
    return np.array(onsets, dtype=np.int64)


@dataclass(frozen=True)
class DuckSettings:
    """Shape of one duck: how deep, and how fast it drops, holds and recovers."""

    depth_db: float = 6.0
    attack_s: float = 0.002
    hold_s: float = 0.03
    release_s: float = 0.1


def duck_curve(
    n_samples: int, onsets: SampleIndices, sr: int, settings: DuckSettings | None = None
) -> npt.NDArray[np.float32]:
    """Build a linear gain curve that dips by ``settings.depth_db`` at each onset, then recovers.

    Attack and release are linear in dB. Overlapping ducks take the deeper value.
    """
    s = settings or DuckSettings()
    attack, hold, release = (max(int(t * sr), 1) for t in (s.attack_s, s.hold_s, s.release_s))
    shape_db = np.concatenate(
        [
            np.linspace(0.0, -s.depth_db, attack, endpoint=False),
            np.full(hold, -s.depth_db),
            np.linspace(-s.depth_db, 0.0, release),
        ]
    )
    shape = (10 ** (shape_db / 20)).astype(np.float32)

    curve = np.ones(n_samples, dtype=np.float32)
    for onset in onsets:
        start = int(onset)
        end = min(start + len(shape), n_samples)
        np.minimum(curve[start:end], shape[: end - start], out=curve[start:end])
    return curve


def apply_ducking(audio: Audio, curve: npt.NDArray[np.float32]) -> Audio:
    """Multiply every channel of ``audio`` by ``curve``."""
    if audio.shape[-1] != curve.shape[0]:
        msg = f"curve length {curve.shape[0]} does not match audio length {audio.shape[-1]}"
        raise ValueError(msg)
    return (audio * curve[np.newaxis, :]).astype(np.float32)
