# ✨ AP Audio Probe

[![CI](https://github.com/shouni/ap-audio-probe/actions/workflows/ci.yml/badge.svg)](https://github.com/shouni/ap-audio-probe/actions/workflows/ci.yml)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](#)
[![Language](https://img.shields.io/badge/Language-Python-blue)](https://www.python.org/)
[![Python Version](https://img.shields.io/badge/Python-3.13%2B-blue)](https://www.python.org/)
[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/shouni/ap-audio-probe)](https://github.com/shouni/ap-audio-probe/tags)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

English | [日本語](README.ja.md)

## 🎯 Overview

**AP Audio Probe** is a verification tool that measures whether a generated song **plays as its recipe instructs**. It lines the audio up with `recipe.json`, measures the actual sound section by section, and reports.

Its main job is detecting **vocals leaking into sections marked instrumental**.

Musical quality (melody, singing, how good the song is) is out of scope. It only tells whether the instructions were broken; musical judgement is left to listening.

---

## 💎 Features and design

### 🎧 Remove the backing before measuring (separate.py)

EDM saw pads and chords **sit in the same band as the voice, harmonics and all**. Telling synths from voice by band energy alone cannot avoid false positives.

So it separates the vocal stem with Demucs and actually removes the backing before measuring. The verdict becomes straightforward in principle, and in practice sung sections and interludes differ by **25dB**. About 20 s for a 3-minute song on Apple Silicon MPS; it does not rerun when already separated.

### 📏 Thresholds are relative to each song (vocals.py)

Fixed absolute values do not work across songs. Demucs never removes the backing completely; a little leaks, and the amount differs per song.

So the reference is **the median RMS of the song's sung sections**: within 12dB of it, or voiced for more than 20 % of the time, means "vocals". Building the yardstick from the song itself keeps the same criterion across genres and loudness.

### 🎣 Exclude bleed across boundaries (vocals.py)

A line can start before the boundary (a pickup) and a held ending runs past it. A section average gets dragged by that second or so: in one interlude where voice appeared only in the last 0.9 s, the RMS jumped **from −50.9dB to −30.0dB and was reported as a violation.**

So voice touching **the edge next to a sung section** is treated as bleed for up to 2.0 s, excluded from the measurement, and its length is shown alongside. Voice in the middle stays, so detection power does not change (confirmed by the positive control below).

The same song's `Intro` sat just outside the verdict at −32.2dB / 5 % voiced; excluding 0.7 s brought it down to **−50.3dB / 0 %**, the floor of separation leakage. The 2.0 s cap doubles the measured 0.9 s while staying shorter than a phrase.

### 🧾 Sections come from the recipe (recipe.py)

`start_seconds` / `end_seconds` in `recipe.json` are the only source of section boundaries. A section is instrumental when its prompt contains `Instruments only`.

Audio shorter than the recipe's declared length is normal (Lyria folds the Outro), so each section is clamped to the real length and any shortfall is reported as `short by`.

### 🚦 A verdict, never a block

Violations are not failures; it returns the numbers and the reasons. Quality problems are handled by adjusting prompts and regenerating by hand, so there is no automatic gate.

---

## 🚀 Quick start

Requirements: Python 3.14 / Google Cloud SDK (`gsutil`) / ffmpeg (used by the loudness measurement).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run downloads Demucs's pretrained model (about 300MB).

### Run (copy and paste)

`<job_id>` comes from the generation job's completion notice. Replace `<bucket>` with where the audio is stored.

```bash
JOB=<job_id>

gsutil cp gs://<bucket>/music/$JOB/master.wav      audio/$JOB.wav
gsutil cp gs://<bucket>/music/$JOB/recipe.json     recipes/$JOB.json
gsutil cp gs://<bucket>/music/$JOB/assessment.json assessments/$JOB.json  # optional

.venv/bin/python -m probe audio/$JOB.wav recipes/$JOB.json
```

A job directory holds four objects.

| Object | Contents |
|---|---|
| `audio.<ext>` | For web delivery: mp3 (192kbps) when mastering ran, otherwise the type Lyria reported |
| `master.wav` | Uncompressed distribution master. **Absent** when mastering was not applied |
| `recipe.json` | The recipe; the only source of section boundaries |
| `assessment.json` | Measurements and review at generation time (used in [Comparing with the generator's record](#-comparing-with-the-generators-record)) |

### mp3 or master.wav

Measured differences (a song from 2026-08, when the limiter ceiling was −1 dBFS):

| | peak | above 16kHz |
|---|---|---|
| `master.wav` | −1.00 dBFS | 19.6 dB |
| `audio.mp3` | **−0.88 dBFS** | 19.5 dB |

- **For vocal detection and lyric transcription, the mp3 is enough.** The verdict relies on a 25dB gap, while encoding changes things by around 0.1dB.
- **For mastering and distribution QC, use `master.wav`.** mp3 encoding overshoots the waveform, so peaks read higher than the source. The wav is what gets distributed, so it is also the product being shipped.

---

## ⚙️ Arguments

```
python -m probe <audio> <recipe> [--out OUT] [--device DEVICE]
```

| Argument | Required | Default | Description |
|---|:---:|---|---|
| `audio` | ✅ | — | Audio (mp3 / wav) |
| `recipe` | ✅ | — | `recipe.json` |
| `--out` | | `out` | Where separated stems are written |
| `--device` | | auto | Demucs device (`mps` / `cpu` / `cuda`). Without it, picks the first available of mps → cuda → cpu |

---

## 📊 Reading the output

```
Split the Role  /  126 BPM
length 174.3s  (recipe 180s)

section    type   span                vocal RMS     peak  voiced  verdict
----------------------------------------------------------------------------
Intro      instr  0-8.0s               -45.9dB   -28.5dB     1%  clean
Verse      sung   8-38.0s              -21.6dB    -7.6dB    90%  -
Chorus     sung   38-74.0s             -19.3dB    -3.5dB    95%  -
Interlude  instr  74-86.0s             -45.5dB   -39.9dB     0%  clean
Verse 2    sung   86-116.0s            -20.9dB    -7.2dB    87%  -
Chorus 2   sung   116-160.0s           -19.6dB    -3.6dB    96%  -
Outro      instr  160-174.3s           -44.5dB   -25.5dB     4%  clean (edge bleed 0.1s excluded, short by 5.7s)
----------------------------------------------------------------------------
reference level of the sung sections: -20.3 dBFS ("vocals" when within 12dB of it, or voiced for more than 20%)
* "edge bleed" is voice reaching the edge from a neighbouring sung section (pickups and held endings).

✓ every section marked instrumental is clean
```

`vocal RMS` is the level of the vocal stem. The larger the gap between sung and instrumental sections, the better the instructions were followed. Above, the gap is 25dB and the interlude is 0 % voiced.

---

## 🧪 Detection power and its limits

Measuring a clean song and getting "no violations" does not prove anything can be detected; an implementation that returns "clean" for any input would give the same result.

So `scripts/calibrate.py` synthesises **positive controls**: audio with voice deliberately mixed into an interlude, at varying levels, to see what gets detected.

```bash
.venv/bin/python scripts/calibrate.py audio/<name>.wav recipes/<name>.json
```

Measured results (injection level relative to the singing):

| Injected | Interlude vocal RMS | Voiced | Verdict |
|---:|---:|---:|---|
| ±0dB | −19.9dB | 93% | ★detected |
| −6dB | −26.1dB | 90% | ★detected |
| −12dB | −33.0dB | 83% | ★detected |
| −18dB | −40.8dB | 25% | ★detected |
| −24dB | −45.3dB | 0% | missed |
| −30dB | −45.4dB | 0% | missed |

**Injections down to 18dB below the singing are detected.** At −24dB and below they sink into Demucs's separation leakage (around −45dB for this song) and cannot be told from a clean interlude. **This is a limit of the method that no threshold can overcome**; more sensitivity would need better separation.

That is enough in practice. Reverb leaking into an interlude at −24dB is not "vocals on top", and flagging it would be crying wolf.

The low end of the sensitivity rests on **the voiced-ratio rule**. The 12dB RMS rule alone only reaches about −12dB; −18dB is caught by the "voiced for more than 20 %" condition. That is why both rules stay.

---

## 🎤 Dropped-line check

Lyria drops lines when the amount of lyrics and the length do not fit. The separated vocal stem is transcribed with faster-whisper and compared line by line with the recipe's lyrics. How lines are dropped depends on the model: with lyria-3.5 it has also been seen **singing one hook twice and dropping another line** (a song generated on 09-04: the start of Chorus 1 twice, one line of its second half missing).

**This looks at something different from the generator's lyric check.** That one compares the number of lines in the score the model returned, not the audio (the generator's `domain/lyric_fidelity.go` says "this is not a transcription of the audio"). **Only transcribing the real audio catches lines that were not sung as scored.**

```bash
.venv/bin/python scripts/check_lyrics.py audio/<name>.wav recipes/<name>.json
```

Lines are compared **after normalising to the reading (katakana)**, so kanji vs kana spelling does not cause mismatches.

### ASR pitfalls

whisper returns hallucinations such as 「ご視聴ありがとうございました」 ("thanks for watching") for windows containing silence, and compared as is, **lines that were actually sung are reported as dropped.** So transcription runs per section, with settings that suppress hallucinations (the cases hit and the fixes that did not work are in the `lyrics.transcribe` docstring).

**A section where the voice never sounds is not transcribed; it returns empty.** Passing it would fill it with hallucinations and **miss a section that lost all its lines.**

**Not every hallucination is gone.** 「ご視聴ありがとうございました」 can still end the transcript of a voiced section. Remeasuring three songs did not move any coverage figure, but a dropped line that happens to match the stock phrase could in principle cross the threshold (not seen so far).

**The lyrics are not passed as `initial_prompt`.** Recognition improves, but the model is pulled towards the given lyrics and **transcribes even the dropped lines.** It would hand over the answer to what is being checked, so it deliberately runs without one.

**English is left out of the comparison.** Japanese ASR does not reliably hear mixed-in English: `Split the role` came out as `Speed the road` every time. Sung lines would show as mismatches, so only the Japanese part is judged, and English-only lines are marked "not checked".

### Why the threshold is where it is

Coverage is measured as **the sum of matching blocks**. The longest continuous match would be broken by small misrecognitions at both ends and read lower than reality (`境界をここに引け` heard as `妖怪よ ここに行け` fell to 27 %).

Calibrated on real data:

| | Coverage |
|---|---|
| Lines actually sung | 73–100 % |
| Lines not sung (negative control) | 14–29 % |

**The groups are 44 points apart, and the 45 % threshold sits between them.** This separation is pinned in the tests.

---

## 🔊 Loudness across songs

The generator scores one song at a time, so level differences across a series are structurally invisible to it, and that is what matters on streaming services.

```bash
.venv/bin/python -m probe.loudness audio/*.wav
```

It lists BS.1770 Integrated LUFS, True Peak and LRA, and warns when **songs differ by more than 1.0 LU**. It also flags **songs above −1.0 dBTP true peak**: limiting the sample peak does not stop inter-sample peaks from going above it, which risks distortion in lossy encoding.

The first 11 songs measured spanned 1.7 LU (−13.2 to −11.5), because mastering had no loudness matching. The generator added matching to a −12.5 LUFS target on 2026-08-15. **The 21 songs generated on 2026-10-17/18 sit at −12.5 to −12.3 LUFS, within 0.2 LU.**

The same 21 songs had true peaks of −0.9 to −1.7 dBTP, and **two exceeded −1.0 dBTP.** The generator's limiter only holds the sample peak at −2 dBFS, so inter-sample peaks come out up to 1.1dB above that. The generator's `assessment.json` had recorded those two at −1.3 / −1.5 dBTP: it measured the mp3, and its meter's interpolation was wrong. Since 2026-10-25 it measures `master.wav` with the fixed meter (records with `loudness_from_master: true`; within 0.08dB of ffmpeg on the same 21 songs). For older records, remeasure the wav here.

---

## 🎛 Band balance per section

The generator's mastering compresses only 6-10kHz. This measures how its constants (`acrossover=split=6000 10000`, `acompressor=threshold`) behave on real songs.

The lower edge of the compression was 4000Hz until 2026-09-05 (compressing 4-6kHz, where consonants and presence sit, made the voice itself cross the threshold in choruses and sound muffled, so it was removed). **The freed 4-6kHz has its own column in the table**, but only 6-10kHz is used to judge compression.

```bash
.venv/bin/python -m probe.spectrum audio/<name>.wav recipes/<name>.json
```

Decisions use **50ms short-time levels**. A whole-section RMS averages 36 seconds and is not what a compressor's detector sees (every section looks "below the threshold").

Only **the processed audio** can be measured. It can show that compression "is working", never that it "is needed".

The "over" column counts on `master.wav` against the threshold the generator recorded in `assessments/<name>.json` (`mastering.treble_threshold_dbfs`) plus the loudness-matching gain. The threshold is relative to each song (the generator changed it from a fixed value on 2026-08-15). **For songs without `mastering` in the record (before the generator's change on 2026-10-29), this column and "GR" below are empty**, because counting against a guessed threshold would be mistaken for what was actually applied. Change the location with `--assessments`.

**The "GR" column** works back, from the ratio (4), how loud each frame over the threshold was before processing, and averages the dB removed over the section. **Base decisions on this one.** "over" is a share of time and cannot tell brief peaks from sustained compression. On 24 songs from 2026-10-24 to 28, verses appeared to spend a median 7 % over the threshold, above the 5 % the design assumed, but the reduction was 0.35dB (1.84dB in choruses); what crossed it were brief peaks such as cymbals and consonants. The calculation ignores knee and attack/release, so it is not precise enough for 0.5dB comparisons.

**The "punch" column** is the gap between the top 5 % and the median of the 40-120Hz short-term level. A standing kick lifts only the hits and widens the gap; bass or pads filling the low end narrow it. As of 2026-10-28, three sparkle_rock songs measured 3.8–5.3dB and the edm_pop median was 8.1dB (over whole songs). Mastering barely moves it, so **when it differs, suspect generation.** There is no pass line.

---

## 🗂 Comparing across songs

A single-song measurement cannot say whether a value is normal. This lines songs up by composing mode (`compose_mode`) and compares each with its mode's median.

```bash
.venv/bin/python -m probe.corpus audio/*.wav             # a few seconds per song
.venv/bin/python -m probe.corpus audio/*.wav --balance   # also voice − backing (runs demucs on unseparated songs)
```

Recipes come from `recipes/<name>.json` and thresholds from `assessments/<name>.json`. Songs without a recipe are skipped; songs without a record get no gain reduction. Chorus values average the sections whose name contains `Chorus`.

```
track                                      LUFS       TP    punch    6-10k       GR       GR voice-bk
                                                    dBTP (chorus) (chorus) (chorus)  (verse) (chorus)
-----------------------------------------------------------------------------------------------------
music-20261028-180903-76d0c0fa4368        -12.3     -2.1      4.7    -34.1        -        -     -3.1
music-20261028-181811-c51786e3932c        -12.3     -2.2      4.4    -34.8        -        -     -3.2
music-20261028-183414-61d15b36643e        -12.3     -2.1      4.1    -33.9        -        -     -3.5
  ↳ sparkle_rock median (3 songs)         -12.3     -2.1      4.4    -34.1        -        -     -3.2
```

There is no pass line; use it to pick songs far from their mode's median as candidates to listen to again. **The two GR columns (chorus / verse) are also the material for tuning the treble compressor's threshold** (the generator's design: "engage only in the chorus, pass the verse").

---

## 🎚 Voice vs backing

The review model often says "synths cover the voice in the chorus", but whether that is true takes listening. This compares the two separated stems (voice and backing) at the same times and reports **the "voice − backing" dB difference per sung section**.

```bash
.venv/bin/python -m probe.balance audio/<name>.wav recipes/<name>.json
```

```
section    span           voice-back   voiced
------------------------------------------------
Verse      8-38s              -0.7dB    22.8s
Chorus     38-74s             -3.4dB    31.3s
```

**Only frames where the voice sounds are measured.** Including frames between lines where only the backing plays would pull the difference negative regardless of the voice. Negative means the backing is louder.

There is no pass threshold. Use it to compare songs, and before / after a prompt change in the generator. On four songs from 2026-10-17, choruses were −4 to −7dB and verses around −1dB, matching the passages the review called buried.

**But choruses at −3 to −7dB are normal.** On 20 lyria-3.5 songs from 2026-09/10 (four each of trance, cybertrance, edm_pop, eurobeat and en/cybertrance), second choruses ranged from −0.7 to −7.0dB, and songs the review scored 5 on `vocal_balance` fell in the same range. A low value does not mean the voice is buried. Use it when a song falls outside this range, or to compare the same song before and after a prompt change.

---

## 🧷 Comparing with the generator's record

The generator leaves `assessment.json` per job: the measurements at generation time and the review of a model that listened to it (four axes: lyric intelligibility, structure match, pacing, vocal balance, plus findings).

```bash
.venv/bin/python -m probe.assessment audio/<name>.wav assessments/<name>.json
```

**They measure different things.** The generator measures the mp3 for the web (true peak and LUFS come from `master.wav` for records with `loudness_from_master: true`); this measures `master.wav`. Mismatched values are normal; the point is to **separate differences explained by encoding from those that are not.** Explained ones get a remark.

| Item | Why it differs |
|---|---|
| Length | The generator's is 52ms longer by the mp3 decoder padding (2,304 samples = two MP3 frames) |
| `peak_amplitude` | The generator takes the peak of **the channel-averaged waveform**, so it is always lower than each channel's peak by the stereo difference (same mp3: 0.903 per channel / **0.867** averaged) |
| True peak, LUFS | Records with `loudness_from_master: true` were measured on `master.wav` by the generator too and compare directly (within 0.08dB of ffmpeg). Older records measured the mp3, and their true peak is off by up to 0.6dB because of an interpolation error |

The method is reimplemented to match the generator's `inspector.go` (one channel-averaged waveform, a single second-order band-pass, 50ms silence windows). It is not `probe.spectrum`'s fourth-order Butterworth; mixing them hides where a difference comes from. **The record's `4-10kHz` is the Inspector's band, not the band the mastering compresses (6-10kHz since 2026-09-05).**

---

## ✅ Tests

```bash
.venv/bin/pytest
```

The regression tests feed synthesised stems straight to the verdict logic, so they need neither Demucs nor its model and finish in seconds. The real positive control (`calibrate.py` above) involves separation and takes minutes.

---

## 📜 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
