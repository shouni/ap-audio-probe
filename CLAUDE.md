# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

AP Audio Probe は、ap-comp が生成した楽曲が**レシピの指示どおりに鳴っているか**を測る検証ベンチです。入力は ap-comp が同じ GCS ディレクトリに吐く音源とレシピの2つで、`recipe.json` の `sections[].start_seconds` / `end_seconds` が区間の唯一の基準になります。

生成の良し悪し（メロディ、歌のうまさ、曲としての出来）は対象外です。ここで測れるのは「指示に反していないか」だけで、音楽的な判断は人間の実聴に残ります。

## コマンド

venv は Python 3.14（PyCharm が作ったプロジェクト内 `.venv`）。numba も ctranslate2 も 3.14 の wheel があるので、Python を下げる必要はありません。

```bash
.venv/bin/pip install -r requirements.txt

# 検証対象を GCS から取る（job_id は ap-mcp の get_music_detail / Slack 通知から）
gsutil cp gs://ap-music/music/<job_id>/master.wav  audio/<name>.wav
gsutil cp gs://ap-music/music/<job_id>/recipe.json recipes/<name>.json

.venv/bin/python -m probe audio/<name>.wav recipes/<name>.json
```

`audio/` `recipes/` `out/` と `*.mp3` `*.wav` は `.gitignore` 済みです。素材はコミットせず、都度 GCS から取り直してください。

自動テストはまだありません。

## mp3 と master.wav のどちらを測るか

ap-comp は同じディレクトリに `audio.mp3`（192kbps）と `master.wav` の両方を置きます。実測した差は次のとおりです。

| | peak | >16kHz |
|---|---|---|
| master.wav | −1.00 dBFS | 19.6 dB |
| audio.mp3 | **−0.88 dBFS** | 19.5 dB |

- **ボーカル検出・ASR は mp3 で十分。** 判定に使う差が 25dB あるのに対し符号化による差は 0.1dB 台で、結論は動きません。whisper は内部で 16kHz モノラルに落とすため差が消えます。
- **マスタリング・配信 QC は master.wav。** mp3 は符号化で波形がオーバーシュートし、ピークが元より高く出ます。配信に出すのは wav なので、出荷物を測るという原則からもこちらです。

## アーキテクチャ

`probe/__main__.py`（表示のみ）→ `separate.py`（分離）→ `vocals.py`（測定）→ `recipe.py`（区間の定義）という一方向の流れです。非自明な判断が3つあります。

**インスト指定の判定は ap-comp のプロンプト規約に依存している。** `recipe.py` の `INSTRUMENTAL_MARKER = "Instruments only"` が、そのセクションを楽器のみと見なす唯一の根拠です。ap-comp 側のセクションプロンプトの文言が変われば、ここも追随が要ります。

**歌とシンセの区別に帯域エネルギーを使わない。** EDM の saw 系パッドやコードは歌声と同じ帯域に倍音ごと乗るため、帯域だけでは分離できません。`separate.py` は demucs で伴奏を実際に除去してから測ります（MPS で3分の曲に20秒程度）。stem が既にあれば再実行しません。

**閾値は絶対値ではなく曲ごとの相対。** `vocals.py` は歌唱セクションの RMS 中央値を基準にし、そこから `VOCAL_MARGIN_DB`（12dB）以内、または有声率 20% 超を「声あり」とします。demucs は伴奏を完全には除去しきれず微量が漏れるうえ、曲ごとにレベルが違うため、絶対値で切ると曲を跨いで機能しません。

音源がレシピの宣言尺より短いことは普通に起きます（Lyria が Outro を畳む）。`vocals.py` は各区間を実尺で clamp し、`SectionStats.truncated` で不足を報告します。

## 方針

**判定は出すが、止めない。** 違反を見つけても失敗扱いにはせず、数字と根拠を並べて人間に返します。生成品質の問題はプロンプト調整と手動での作り直しで対処する運用のため、自動ゲートは入れません。

## 既知の問題

- `README.md` は lyric-video のテンプレートが貼られたままで、内容がこのプロジェクトの説明になっていません（未追跡）。
- ap-comp 側の Audio Check は `master.wav` ではなく mp3 を、しかもデコーダの padding を含んだまま測っています（尺が 2,304 サンプル＝MP3 フレーム2つぶん長く出る）。報告される末尾無音などがその分ずれます。
