# ✨ AP Audio Probe

[![Language](https://img.shields.io/badge/Language-Python-blue)](https://www.python.org/)
[![Python Version](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](#)


## 🎯 概要

**AP Audio Probe** は、ap-comp が生成した楽曲が**レシピの指示どおりに鳴っているか**を測る検証ツールです。音源と `recipe.json` を突き合わせ、セクションごとに実際の音を測って報告します。

主な用途は、**インスト指定のセクションにボーカルが混入していないか**の検出です。ap-comp のセクションプロンプトには `Instruments only — no lead vocal, no vocalised syllables` と明示されていますが、これを検証する仕組みは従来ありませんでした。間奏にハミングが乗っても、そのまま高スコアが付きます。

生成の良し悪し（メロディ、歌のうまさ、曲としての出来）は対象外です。ここで分かるのは「指示に反していないか」だけで、音楽的な判断は実聴に残ります。

---

## 💎 特徴と設計思想

### 🎧 伴奏を除去してから測る（separate.py）

EDM の saw 系パッドやコードは、**歌声と同じ帯域に倍音ごと乗ります**。帯域エネルギーだけでシンセと歌を見分けようとすると誤検出が避けられません。

そこで Demucs でボーカル stem を分離し、伴奏を実際に取り除いてから測ります。判定が原理的に素直になり、実測でも歌唱区間と間奏で **25dB** の差が付きます。Apple Silicon の MPS で3分の曲に20秒程度。分離済みなら再実行しません。

### 📏 閾値は曲ごとの相対（vocals.py）

固定の絶対値では曲を跨いで機能しません。Demucs は伴奏を完全には除去しきれず微量が漏れ、そのレベルは曲ごとに違うためです。

そこで**その曲の歌唱セクションの RMS 中央値**を基準に取り、そこから 12dB 以内、または有声率 20% 超を「声あり」と判定します。物差しを曲自身から作ることで、ジャンルや音圧が変わっても同じ基準で読めます。

### 🧾 区間の基準はレシピ（recipe.py）

セクションの境界は `recipe.json` の `start_seconds` / `end_seconds` が唯一の基準です。インスト指定かどうかは、セクションプロンプトに `Instruments only` が含まれるかで判定します（ap-comp のプロンプト規約に依存）。

音源がレシピの宣言尺より短いことは普通に起きるため（Lyria が Outro を畳む）、各区間は実尺で clamp し、不足があれば `尺不足` として報告します。

### 🚦 判定は出すが、止めない

違反を見つけても失敗扱いにはせず、数字と根拠を並べて返します。生成品質の問題はプロンプト調整と手動での作り直しで対処する運用のため、自動ゲートは設けていません。

---

## 🚀 クイックスタート

必要なもの: Python 3.14 / Google Cloud SDK（`gsutil`）。GPU は任意で、無ければ `--device cpu`。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

初回実行時に Demucs の学習済みモデル（約 300MB）が自動で取得されます。

### 実行（コピペ用）

`<job_id>` は Slack 通知、または ap-mcp の `get_music_detail` から取れます。

```bash
JOB=comp-20260811-014550-54471f266c38

gsutil cp gs://ap-music/music/$JOB/master.wav  audio/$JOB.wav
gsutil cp gs://ap-music/music/$JOB/recipe.json recipes/$JOB.json

.venv/bin/python -m probe audio/$JOB.wav recipes/$JOB.json
```

---

## ⚙️ 引数

```
python -m probe <audio> <recipe> [--out OUT] [--device DEVICE]
```

| 引数 | 必須 | 既定値 | 説明 |
|---|:---:|---|---|
| `audio` | ✅ | — | 音源（mp3 / wav） |
| `recipe` | ✅ | — | ap-comp の `recipe.json` |
| `--out` | | `out` | 分離した stem の出力先 |
| `--device` | | `mps` | Demucs の実行デバイス（`mps` / `cpu` / `cuda`） |

---

## 📊 出力の読み方

```
Split the Role  /  126 BPM
実尺 174.4s  (レシピ指定 180s)

section    指定     区間                vocal RMS     peak     有声率  判定
----------------------------------------------------------------------------
Intro      インスト   0-8.0s               -46.0dB   -28.6dB     2%  クリーン
Verse      歌唱     8-38.0s              -21.9dB    -7.9dB    89%  -
Chorus     歌唱     38-74.0s             -19.6dB    -3.8dB    95%  -
Interlude  インスト   74-86.0s             -45.8dB   -39.9dB     0%  クリーン
Verse 2    歌唱     86-116.0s            -21.2dB    -7.4dB    86%  -
Chorus 2   歌唱     116-160.0s           -19.9dB    -4.2dB    95%  -
Outro      インスト   160-174.4s           -45.3dB   -28.2dB     3%  クリーン (尺不足 5.6s)
----------------------------------------------------------------------------
歌唱セクションの基準値: -20.5 dBFS (この値から 12dB 以内、または有声率 20% 超で「声あり」)

✓ インスト指定セクションはすべてクリーン
```

`vocal RMS` はボーカル stem の音量です。歌唱セクションとインストセクションの差が大きいほど、指示が守られていることになります。上の例では 25dB 差、間奏の有声率は 0%。

---

## 🎚 mp3 と master.wav のどちらを測るか

ap-comp は同じ GCS ディレクトリに `audio.mp3`（192kbps）と `master.wav` の両方を置きます。実測した差は次のとおりです。

| | peak | 16kHz 以上 |
|---|---|---|
| `master.wav` | −1.00 dBFS | 19.6 dB |
| `audio.mp3` | **−0.88 dBFS** | 19.5 dB |

- **ボーカル検出なら mp3 で十分。** 判定に使う差が 25dB あるのに対し、符号化による差は 0.1dB 台で結論が動きません。
- **マスタリング・配信の QC は `master.wav`。** mp3 は符号化で波形がオーバーシュートし、ピークが元より高く出ます。配信に出すのは wav なので、出荷物を測るという意味でもこちらです。

---

## 📜 ライセンス (License)

このプロジェクトは [MIT License](https://opensource.org/licenses/MIT) の下で公開されています。
