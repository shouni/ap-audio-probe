# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AP Audio Probe is a test bench that measures whether a generated song **plays as its recipe instructs**. Its inputs are the audio and the recipe that the generator writes to the same GCS directory; `sections[].start_seconds` / `end_seconds` in `recipe.json` is the only source of section boundaries.

Musical quality (melody, singing, how good the song is) is out of scope. It can only tell whether the instructions were broken; musical judgement stays with a human listener.

## Conventions

- **Write code comments, docstrings, CLI help, table headings and messages in English.** This is a public repository; the whole codebase moved to English on 2026-11-09, so do not mix Japanese back in. Japanese stays only as data: lyrics and transcripts in tests (the lyric check works on Japanese singing), and quoted hallucination strings such as 「ご視聴ありがとうございました」.
- **`README.md` is English and `README.ja.md` is its Japanese version.** When one changes, update the other in the same commit.

## Where things are written

An agreement so the same thing is not written three times. It started when the 2026-08-15 threshold change was scattered over three places and two of them stayed a week out of date.

- **Why something was decided (measurements, what did not work) goes in the comment on that constant or function**, where anyone changing the decision will see it.
- **What can and cannot be measured goes in the README.** Its readers run the tool and do not read the code.
- **This file holds only what is needed before reading the code, and the generator's open questions.** It points to where details live instead of repeating them.

## Commands

The venv is Python 3.14 (the in-project `.venv` created by PyCharm); requirements.txt needs 3.13+ (audioop-lts). numba and ctranslate2 both ship 3.14 wheels, so there is no need for an older Python.

```bash
.venv/bin/pip install -r requirements.txt

# fetch what to check from GCS (job_id from the job's completion notice; replace <bucket>)
gsutil cp gs://<bucket>/music/<job_id>/master.wav  audio/<name>.wav
gsutil cp gs://<bucket>/music/<job_id>/recipe.json recipes/<name>.json

.venv/bin/python -m probe audio/<name>.wav recipes/<name>.json
```

```bash
.venv/bin/pytest                                                     # regression tests (seconds)
.venv/bin/ruff check .                                               # lint (settings in pyproject.toml; pip install -r requirements-dev.txt)
.venv/bin/pytest tests/test_vocals.py::test_clean_track_reports_no_violation
.venv/bin/python scripts/calibrate.py    audio/x.wav recipes/x.json  # real positive control (minutes)
.venv/bin/python scripts/check_lyrics.py audio/x.wav recipes/x.json  # dropped-line check (minutes)
.venv/bin/python -m probe.loudness audio/*.wav                       # loudness across songs
.venv/bin/python -m probe.spectrum audio/x.wav recipes/x.json        # band levels per section
.venv/bin/python -m probe.balance  audio/x.wav recipes/x.json        # "voice − backing" per sung section
.venv/bin/python -m probe.corpus   audio/*.wav [--balance]            # line songs up by mode
.venv/bin/python -m probe.assessment audio/x.wav assessments/x.json  # compare with the generator's record
```

The four objects in a job directory are listed in a table in the README.

CI (`.github/workflows/ci.yml`) runs the tests on Python 3.13 and 3.14 with only numpy / soundfile / scipy / pykakasi / pytest plus the ffmpeg binary (the assessment tests decode with it), and ruff. Keep demucs, torch and faster-whisper imported inside the functions that need them, or CI has to install them just to import a module.

`audio/` `recipes/` `assessments/` `out/` and `*.mp3` `*.wav` are in `.gitignore`. Do not commit material; fetch it from GCS each time.

## mp3 or master.wav

The mp3 is enough for vocal detection and ASR; mastering and distribution QC use master.wav. The reasons and the measured differences are in the README (whisper downsamples to 16kHz mono internally, so for ASR the difference disappears).

## Architecture

A one-way flow: `probe/__main__.py` (display only) → `separate.py` (separation) → `vocals.py` (measurement) → `recipe.py` (section definitions). Four premises are needed before reading the code.

**Detecting instrumental sections depends on the generator's prompt convention.** `INSTRUMENTAL_MARKER = "Instruments only"` in `recipe.py` is the only basis for treating a section as instruments only. The assumption that lyric tags like `[Verse]` match `sections[].name` exactly (`parse_lyrics`) is the same kind of dependency; if the generator's wording changes, this has to follow.

**Band energy is not used to tell voice from synths.** EDM saw pads and chords sit in the same band as the voice, harmonics and all, so bands alone cannot separate them. `separate.py` removes the backing with demucs before measuring (about 20 s for a 3-minute song on MPS; skipped if the stem exists).

**Thresholds are relative to each song, not absolute.** The yardstick is the median RMS of the song's sung sections (`vocals.py: sung_reference`). `VOCAL_MARGIN_DB` and `ACTIVE_RATIO_THRESHOLD` play different roles and **neither can be removed**. The detection floor is about −18dB relative to the singing, a limit set by separation quality, not by a threshold (positive control in `scripts/calibrate.py`, measured table in the README).

**Voice touching a section boundary is excluded from the section's core.** A line can start before the boundary (a pickup) and a held ending can run past it, which makes the section average jump by tens of dB. Only **the edge next to a sung section** is excluded, capped at `BOUNDARY_BLEED_SECONDS`. The measurements behind it, and why a run over the cap is not partly excluded, are in the comments in `vocals.py`.

Audio shorter than the recipe is normal (Lyria folds the Outro). `vocals.py` clamps each section to the real length and reports the shortfall with `SectionStats.truncated`.

### Lyric check (lyrics.py)

Used naively, ASR reports sung lines as dropped. The pitfalls hit and the fixes that failed are in the docstrings of `transcribe` and `trim_to_voice`; three things must not change.

- **Never pass the expected lyrics as `initial_prompt`.** Recognition improves, but it hands over the answer to what is being checked.
- **Transcribe per section. The only thing that stopped the remaining hallucinations was `hallucination_silence_threshold`.** Trimming trailing silence, `vad_filter=True` and a lower `no_speech_threshold` all failed.
- **Never transcribe a section where the voice never sounds** (`trim_to_voice`). A section that lost all its lines looks exactly like that, and filling it with hallucinations defeats the check. Voice is judged inside the section only, while the padding comes from outside it (doing it the other way brings one of the failures back; see the docstring).

How coverage is counted and why the threshold is 45 % are in the comments on `_coverage` / `COVERAGE_THRESHOLD`; the calibration measurements are pinned in `tests/test_lyrics.py::test_threshold_sits_between_the_two_groups`.

### Comparing with the generator's record (assessment.py)

`assessment.json` is the generator measuring the mp3 (true peak and LUFS come from `master.wav` only when `loudness_from_master: true`), while this measures the wav. **The same quantities on different targets, so the values do not match.** To see whether a difference is explained by encoding and padding, it reimplements `inspector.go`'s method (one channel-averaged waveform, a single second-order band-pass, 50ms silence windows). That is not `spectrum.py`'s fourth-order Butterworth; mixing them would hide where a difference comes from. The explanation per item is in the README table and the comments on `assessment.measure` / `compare`.

### Band measurement (spectrum.py)

A measurement to check the generator's mastering constants (`acrossover=split=6000 10000`, `acompressor=threshold`). Decisions use `_short_term_dbfs` (50ms windows), not the whole-section RMS.

The threshold for the "over" column is not fixed; **it is read from `mastering` in `assessment.json` (the generator's `MasteringSettings`).** Songs without the record get no column. The reason, and how the threshold became relative on 2026-08-15, are in the comment before `BandLevels`.

**`CROSSOVER_HZ` must always equal the generator's chain (`processingChain` in `masterer.go`).** The lower edge moved from 4000Hz to 6000Hz on 2026-09-05; the reason is in that constant's comment. 4-6kHz, no longer compressed, is cut at `PRESENCE_HZ` and shown as a separate column, but not used for the verdict.

## Policy

**It gives a verdict but never blocks.** A violation is not a failure; it returns the numbers and the reasons to a human. Quality problems are handled by adjusting prompts and regenerating by hand, so there is no automatic gate.

## Known issues

- **Some songs exceed −1.0 dBTP true peak slightly.** The generator's limiter ceiling is −2 dBFS (`limiterCeiling` in `masterer.go`, by sample peak), and inter-sample peaks measured up to 1.1dB higher, so the worst case is −0.9 dBTP, 0.1dB short of the recommendation. Of 21 songs generated on 10-17/18, two exceeded it (numbers in the README). True peaks in records without `loudness_from_master: true` (before 2026-10-25) were measured on the mp3 with a meter whose interpolation was wrong; do not trust them, and remeasure with `probe.loudness`.
- **The treble compression threshold needs verification.** The +2.5dB offset of the relative threshold was calibrated on mastered audio (then 4-10kHz) and moved to the unprocessed input with corrections (0.35dB for the method, 1.4dB for the target). It stayed the same when the band moved to 6-10kHz on 09-05. **The unprocessed input is not kept, so master.wav alone cannot confirm it.** On 2026-10-29 the generator started writing `treble_dbfs` / `treble_threshold_dbfs` / `loudness_gain_db` into `mastering` in `assessment.json`, so once enough later songs accumulate, `probe.corpus`'s gain reduction (chorus / verse) can tune it. A trial on 24 songs matched against the logs on 2026-10-28 showed that raising the offset barely changes the verses (0.35 → 0.20dB even at +6.0) while chorus compression and the `harsh_treble` margin shrink first, so it stays at +2.5 (reasoning in the comment on the generator's `trebleThresholdOffsetDB`). Earlier songs only have it in the generator's logs (the line 「マスタリング定数を決めました」).
