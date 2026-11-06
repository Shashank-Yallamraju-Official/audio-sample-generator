# Sample library manifest

Collected 2026-11-06. 27 packs, 26,135 audio files, about 7.7 GB. Machine-readable copy: `manifest.json`.

## Rules applied
- Only lossless audio: WAV or FLAC, 16-bit/44.1 kHz minimum (most packs are 24-bit). FLAC is lossless; convert with `ffmpeg -i in.flac out.wav` if a tool needs WAV.
- Only packs whose own LICENSE file grants free use (CC0, Unlicense, CC BY). Each pack folder keeps its original LICENSE.
- Lossy previews (31 MP3/OGG files) and script files were removed. Every file's header was probed with ffprobe; 121 corrupt WAVs in black-and-green-guitars were removed.
- To fit the project's 10 GB storage cap, integer WAVs were losslessly re-encoded to FLAC (bit depth preserved) and the .sfz references updated. WAVs with loop/cue chunks and 32-bit float WAVs were left as WAV. Sforzando .bank.xml GUIs may refuse edited programs; the plain .sfz files work in any SFZ player.
- Dropped for space (all CC0, re-downloadable): [VCSL](https://github.com/sgossner/VCSL), [Swirly Drums](https://github.com/sfzinstruments/karoryfer.swirly-drums), [Unruly Drums](https://github.com/sfzinstruments/karoryfer.unruly-drums), [Big Rusty Drums](https://github.com/sfzinstruments/karoryfer.big-rusty-drums).
- Layout: `samples/<type>/<source>/<pack>/`. Most packs include `.sfz` mappings (plain text) describing note/velocity/round-robin layout.

## Attribution required
When output uses these packs, credit the author:

- **dim-cabasa**: Dim Cabasa by kinwie, played by Dimas Syah (CC BY 4.0 (attribution required))
- **salamander-grand-piano**: Salamander Grand Piano by Alexander Holm (CC BY 3.0 (attribution required))
- **headroom-piano**: Headroom Piano by Bengt Nilsson (CC BY 4.0 (attribution required))
- **solo-sax**: MTG (Music Technology Group, Universitat Pompeu Fabra); SFZ by kinwie (CC BY 4.0 (attribution required))

## Packs

### drums

| Pack | Contents | License | Files | Formats | Size | Source |
|---|---|---|---|---|---|---|
| `drums/Karoryfer/frankensnare` | Snare drum collection: many snares, sticks/brushes, multiple mics, velocity layers and round-robins | CC0 1.0 (public domain dedication) | 3,580 | FLAC 16-bit/44.1kHz (3148); FLAC 24-bit/44.1kHz (336); WAV 16-bit/44.1kHz (96) | 444 MB | [sfzinstruments/karoryfer.frankensnare](https://github.com/sfzinstruments/karoryfer.frankensnare) @ `9151c2d` |
| `drums/Versilian-Karoryfer/virtuosity-drums` | Full acoustic drum kit (kick, snare, toms, hats, cymbals), multi-mic, KVR DC'21 entry | CC0 1.0 (public domain dedication) | 4,858 | FLAC 24-bit/48kHz (3438); FLAC 16-bit/44.1kHz (1312); WAV 24-bit/48kHz (108) | 1,201 MB | [sfzinstruments/virtuosity_drums](https://github.com/sfzinstruments/virtuosity_drums) @ `9f04cf9` |
| `drums/Karoryfer/the-hat-with-the-phat` | Giant hi-hat made from two ride cymbals, 24-bit | CC0 1.0 (public domain dedication) | 712 | WAV 24-bit/44.1kHz (712) | 943 MB | [sfzinstruments/Karoryfer.TheHatWithThePhat](https://github.com/sfzinstruments/Karoryfer.TheHatWithThePhat) @ `afa76f2` |

### percussion

| Pack | Contents | License | Files | Formats | Size | Source |
|---|---|---|---|---|---|---|
| `percussion/sfzinstruments/body-percussion` | Body percussion: claps, snaps, chest/thigh slaps | CC0 1.0 (public domain dedication) | 202 | FLAC 24-bit/44.1kHz (162); WAV float 32-bit/44.1kHz (40) | 34 MB | [sfzinstruments/body_percussion](https://github.com/sfzinstruments/body_percussion) @ `4ac9d89` |
| `percussion/kinwie/dim-cabasa` | Cabasa, 250 unprocessed 24/48 hits (SM57 > Neve 1073) | CC BY 4.0 (attribution required) | 250 | FLAC 24-bit/48kHz (250) | 13 MB | [sfzinstruments/kinwie.dim-cabasa](https://github.com/sfzinstruments/kinwie.dim-cabasa) @ `016457e` |
| `percussion/jlearman/steel-drum` | Chromatic C tenor steel drum | The Unlicense (public domain) | 368 | FLAC 16-bit/44.1kHz (368) | 43 MB | [sfzinstruments/jlearman.SteelDrum](https://github.com/sfzinstruments/jlearman.SteelDrum) @ `dc15a36` |

### bass

| Pack | Contents | License | Files | Formats | Size | Source |
|---|---|---|---|---|---|---|
| `bass/Karoryfer/black-and-blue-basses` | Two electric bass guitars, multiple articulations, 24-bit | CC0 1.0 (public domain dedication) | 2,208 | FLAC 24-bit/44.1kHz (2040); WAV 24-bit/44.1kHz (168) | 520 MB | [sfzinstruments/karoryfer.black-and-blue-basses](https://github.com/sfzinstruments/karoryfer.black-and-blue-basses) @ `6e7d674` |
| `bass/Karoryfer/meatbass` | 1958 Otto Rubner double bass (fifths tuning), arco, 24-bit | CC0 1.0 (public domain dedication) | 536 | FLAC 24-bit/44.1kHz (434); WAV 24-bit/44.1kHz (102) | 157 MB | [sfzinstruments/karoryfer.meatbass](https://github.com/sfzinstruments/karoryfer.meatbass) @ `ac9e859` |
| `bass/Karoryfer/growlybass` | Aggressive/growly electric bass, 24-bit | CC0 1.0 (public domain dedication) | 380 | FLAC 24-bit/44.1kHz (380) | 95 MB | [sfzinstruments/karoryfer.growlybass](https://github.com/sfzinstruments/karoryfer.growlybass) @ `4f48326` |
| `bass/DSmolken/double-bass` | 1958 Otto Rubner double bass, arco and pizzicato plus noises, 32-bit float | CC0 1.0 (public domain dedication) | 401 | WAV float 32-bit/44.1kHz (401) | 278 MB | [sfzinstruments/dsmolken.double-bass](https://github.com/sfzinstruments/dsmolken.double-bass) @ `c2985eb` |

### guitar

| Pack | Contents | License | Files | Formats | Size | Source |
|---|---|---|---|---|---|---|
| `guitar/Karoryfer/black-and-green-guitars` | Two electric guitars (DI-style), multiple articulations, 24-bit. 121 files corrupted upstream were removed (see REMOVED_CORRUPT_FILES.txt) | CC0 1.0 (public domain dedication) | 1,655 | FLAC 24-bit/44.1kHz (1651); WAV 24-bit/44.1kHz (4) | 472 MB | [sfzinstruments/karoryfer.black-and-green-guitars](https://github.com/sfzinstruments/karoryfer.black-and-green-guitars) @ `b3b3249` |
| `guitar/Karoryfer/emilyguitar` | Epiphone Emily the Strange electric guitar, incl. chord and multi-voice mappings, 24-bit | CC0 1.0 (public domain dedication) | 323 | FLAC 24-bit/44.1kHz (323) | 53 MB | [sfzinstruments/karoryfer.emilyguitar](https://github.com/sfzinstruments/karoryfer.emilyguitar) @ `b4920dc` |
| `guitar/Karoryfer/shinyguitar` | Archtop guitar, 24-bit | CC0 1.0 (public domain dedication) | 846 | FLAC 24-bit/44.1kHz (846) | 192 MB | [sfzinstruments/karoryfer.shinyguitar](https://github.com/sfzinstruments/karoryfer.shinyguitar) @ `57243cc` |

### keys

| Pack | Contents | License | Files | Formats | Size | Source |
|---|---|---|---|---|---|---|
| `keys/Alexander-Holm/salamander-grand-piano` | Yamaha C5 grand piano, 16 velocity layers, 24/48, with pedal/hammer/resonance noises | CC BY 3.0 (attribution required) | 641 | FLAC 24-bit/48kHz (641) | 719 MB | [sfzinstruments/SalamanderGrandPiano](https://github.com/sfzinstruments/SalamanderGrandPiano) @ `3382bf9` |
| `keys/Bengt-Nilsson/headroom-piano` | Yamaha C3 grand piano, close + Decca tree mics, 5 velocity layers | CC BY 4.0 (attribution required) | 300 | FLAC 16-bit/44.1kHz (300) | 151 MB | [sfzinstruments/BengtNilsson.HeadroomPiano](https://github.com/sfzinstruments/BengtNilsson.HeadroomPiano) @ `2a7df3f` |
| `keys/Versilian-Karoryfer/osiris-piano` | Acoustic piano by Versilian Studios and Karoryfer, sustain and noise variants | CC0 1.0 (public domain dedication) | 1,242 | FLAC 16-bit/48kHz (1242) | 477 MB | [sfzinstruments/Osiris_Piano](https://github.com/sfzinstruments/Osiris_Piano) @ `18c6afc` |

### strings

| Pack | Contents | License | Files | Formats | Size | Source |
|---|---|---|---|---|---|---|
| `strings/Karoryfer-Bigcat/cello` | Solo cello, 24-bit | CC0 1.0 (public domain dedication) | 479 | FLAC 24-bit/44.1kHz (343); WAV 24-bit/44.1kHz (136) | 119 MB | [sfzinstruments/karoryfer-bigcat.cello](https://github.com/sfzinstruments/karoryfer-bigcat.cello) @ `6fd75fb` |

### winds-brass

| Pack | Contents | License | Files | Formats | Size | Source |
|---|---|---|---|---|---|---|
| `winds-brass/Karoryfer/bear-sax` | 1926 Conn baritone saxophone | CC0 1.0 (public domain dedication) | 755 | WAV float 32-bit/44.1kHz (438); FLAC 16-bit/44.1kHz (256); WAV 24-bit/44.1kHz (61) | 138 MB | [sfzinstruments/karoryfer.bear-sax](https://github.com/sfzinstruments/karoryfer.bear-sax) @ `7abb3c6` |
| `winds-brass/Karoryfer/war-tuba` | Tuba, many articulations | CC0 1.0 (public domain dedication) | 1,443 | FLAC 16-bit/44.1kHz (1356); WAV 16-bit/44.1kHz (87) | 83 MB | [sfzinstruments/karoryfer.war-tuba](https://github.com/sfzinstruments/karoryfer.war-tuba) @ `5b62dd6` |
| `winds-brass/MTG-UPF/solo-sax` | Soprano, alto, tenor and baritone saxophones, 24/48 | CC BY 4.0 (attribution required) | 737 | FLAC 24-bit/48kHz (737) | 111 MB | [sfzinstruments/MTG.SoloSax](https://github.com/sfzinstruments/MTG.SoloSax) @ `b494d25` |

### world

| Pack | Contents | License | Files | Formats | Size | Source |
|---|---|---|---|---|---|---|
| `world/sfzinstruments/erhu` | Chinese erhu (bowed), 24-bit | CC0 1.0 (public domain dedication) | 170 | FLAC 24-bit/44.1kHz (170) | 48 MB | [sfzinstruments/aliexpress-erhu](https://github.com/sfzinstruments/aliexpress-erhu) @ `6615047` |
| `world/sfzinstruments/cithara-barbarica` | 10-string medieval lyre, 24-bit | CC0 1.0 (public domain dedication) | 200 | FLAC 24-bit/44.1kHz (200) | 86 MB | [sfzinstruments/cithara-barbarica](https://github.com/sfzinstruments/cithara-barbarica) @ `a47c10d` |
| `world/sfzinstruments/hungarian-zither` | Hungarian zither (citera), 24-bit | CC0 1.0 (public domain dedication) | 210 | FLAC 24-bit/44.1kHz (210) | 172 MB | [sfzinstruments/hungarian_zither](https://github.com/sfzinstruments/hungarian_zither) @ `973d944` |
| `world/sfzinstruments/ganjo` | Guitar-banjo, 32-bit float | CC0 1.0 (public domain dedication) | 61 | WAV float 32-bit/44.1kHz (61) | 25 MB | [sfzinstruments/ganjo](https://github.com/sfzinstruments/ganjo) @ `ccff5cd` |

### synth

| Pack | Contents | License | Files | Formats | Size | Source |
|---|---|---|---|---|---|---|
| `synth/Karoryfer/caveman-cosmonaut` | Unitra B-11 transistor organ turned into a virtual analog synth | CC0 1.0 (public domain dedication) | 370 | WAV 16-bit/44.1kHz (370) | 107 MB | [sfzinstruments/karoryfer.caveman-cosmonaut](https://github.com/sfzinstruments/karoryfer.caveman-cosmonaut) @ `9de9bcf` |
| `synth/Karoryfer/cowsynth` | Medieval bagpipe replica used as synth oscillators (organ, pad, pluck mappings) | CC0 1.0 (public domain dedication) | 40 | WAV 16-bit/44.1kHz (40) | 14 MB | [sfzinstruments/karoryfer.cowsynth](https://github.com/sfzinstruments/karoryfer.cowsynth) @ `5a5b5af` |

### orchestral

| Pack | Contents | License | Files | Formats | Size | Source |
|---|---|---|---|---|---|---|
| `orchestral/Versilian-Studios/vsco-2-ce` | Versilian Studios Chamber Orchestra 2 CE: strings, woodwinds, brass, percussion, keys | CC0 1.0 (public domain dedication) | 3,168 | FLAC 16-bit/44.1kHz (1761); FLAC 24-bit/44.1kHz (1401); WAV 16-bit/44.1kHz (6) | 1,151 MB | [sgossner/VSCO-2-CE](https://github.com/sgossner/VSCO-2-CE) @ `4403009` |
