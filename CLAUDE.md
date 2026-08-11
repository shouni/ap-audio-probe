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

```bash
.venv/bin/pytest                                                     # 回帰テスト(0.2秒)
.venv/bin/pytest tests/test_vocals.py::test_clean_track_reports_no_violation
.venv/bin/python scripts/calibrate.py    audio/x.wav recipes/x.json  # 実地の陽性対照(数分)
.venv/bin/python scripts/check_lyrics.py audio/x.wav recipes/x.json  # 行落ち検査(数分)
.venv/bin/python -m probe.loudness audio/*.wav                       # 曲間のラウドネス
.venv/bin/python -m probe.spectrum audio/x.wav recipes/x.json        # セクション別の帯域
```

`audio/` `recipes/` `out/` と `*.mp3` `*.wav` は `.gitignore` 済みです。素材はコミットせず、都度 GCS から取り直してください。

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

この2つの条件は役割が違い、**どちらも外せません**。陽性対照（`scripts/calibrate.py`）で測ったところ、RMS の 12dB ルールが届くのは混入量 −12dB 付近までで、**−18dB を拾っているのは有声率のルール**です。検出できる下限は歌唱に対して約 −18dB、それ以下は demucs の分離漏れ（−45dB 前後）に埋もれて区別できません。閾値をいじっても超えられない手法上の限界なので、感度を上げたければ分離の質を上げる必要があります。

音源がレシピの宣言尺より短いことは普通に起きます（Lyria が Outro を畳む）。`vocals.py` は各区間を実尺で clamp し、`SectionStats.truncated` で不足を報告します。

### 歌詞照合（lyrics.py）

**期待した歌詞を `initial_prompt` に渡してはいけません。** 認識率は上がりますが、モデルが与えた歌詞に引きずられ、落ちた行まで書き起こされます。検証したい対象を答えとして渡すことになります。

**`to_kana` は英数字を落とします。** 日本語 ASR は混ざった英語を安定して聞き取れず、実測で `Split the role` は一貫して `Speed the road` になりました。残すと歌われている行まで不一致になります。日本語を含まない行は `LineCheck.verifiable` が偽になり、判定対象外として報告されます。

**一致率は最長の連続一致ではなく一致ブロックの合計。** 連続一致だと両端の小さな誤認識で分断されます（`境界をここに引け` → `妖怪よ ここに行け` で 27% まで低下）。合計に変えて 73% になりました。閾値 `COVERAGE_THRESHOLD`（45%）は実測で校正しており、歌われた行 73〜100% と歌われていない行 14〜29% の間に収まります。この分離は `test_threshold_sits_between_the_two_groups` で固定してあります。

## 方針

**判定は出すが、止めない。** 違反を見つけても失敗扱いにはせず、数字と根拠を並べて人間に返します。生成品質の問題はプロンプト調整と手動での作り直しで対処する運用のため、自動ゲートは入れません。

### 帯域計測（spectrum.py）

ap-comp の masterer が持つ定数（`acrossover=split=4000 10000`、`acompressor=threshold=0.025` ＝ −32.0dBFS）を検証するための計測です。

**セクション全体の RMS を圧縮の判断に使ってはいけません。** 36秒を平均した値はコンプレッサの検出器が見ている量とは別物で、全区間が「閾値以下」に見えてしまいます。`_short_term_dbfs` が 50ms 窓（masterer の attack=5ms / release=80ms に対応）で短時間レベルを出し、その95パーセンタイルと閾値超過時間の割合を判断に使います。

5曲で測った結果、**サビの短時間レベルは −29.1〜−31.4dBFS で全曲が閾値を超え、閾値超過時間は 10〜43%**。Verse は 1〜9% でほぼ素通りです。masterer の設計意図（サビだけ抑え、Verse は通す）は corpus 全体で成立しています。ただしこれは**処理後の測定なので「効いている」ことしか言えず、「必要である」ことの証明にはなりません**。プロンプト側の高域指示だけで足りるかは、処理前の take がないと判定できません。

## 既知の問題

- **`--device mps` が既定でフォールバックがない。** Apple Silicon 以外では `--device cpu` の明示が要ります。
- **ap-comp のマスターは真正ピークが −1.0 dBTP を超えている。** 手元で測った5曲すべてで超過し、うち3曲は 0.0 dBTP 以上でした。サンプルピークは −1.00 dBFS ちょうどに抑えられているので、リミッターがサンプルピーク基準で、サンプル間のピークを見ていないと思われます。配信時のロッシー変換で歪む恐れがあり、**ap-comp の masterer 側の課題**です。Integrated LUFS の揃いは 0.6 LU と良好でした。
- ap-comp 側の Audio Check は `mastered.Web`（mp3）を測っています（`internal/adapters/lyria.go:244`）。デコーダの padding を含むため尺が 2,304 サンプル＝MP3 フレーム2つぶん長く出ます。ただし実測した限りどの閾値も余裕のほうが桁で大きく、判定が反転するケースは見つかっていません。あわせて `peak_amplitude: 0.867` が wav（0.891）とも mp3（0.903）とも一致しない理由が未解明です。
