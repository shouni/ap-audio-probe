# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

AP Audio Probe は、生成された楽曲が**レシピの指示どおりに鳴っているか**を測る検証ベンチです。入力は生成側が同じ GCS ディレクトリに吐く音源とレシピの2つで、`recipe.json` の `sections[].start_seconds` / `end_seconds` が区間の唯一の基準になります。

生成の良し悪し（メロディ、歌のうまさ、曲としての出来）は対象外です。ここで測れるのは「指示に反していないか」だけで、音楽的な判断は人間の実聴に残ります。

## ドキュメントの置き場所

同じ話を三度書かないための取り決めです。2026-08-15 の閾値変更が3か所に散り、うち2か所が1週間ぶん古いまま残ったのがきっかけでした。

- **なぜそう決めたか（実測値、効かなかった手）は、その定数や関数のコメント。** 判断を変えるときに必ず目に入る場所に置きます。
- **何が測れて何が測れないかは README。** 読者は実行する人で、コードは読みません。
- **このファイルは、コードを読む前に要る前提と、生成側の未確定事項だけ。** 詳細は繰り返さず、どのファイルにあるかを指します。

## コマンド

venv は Python 3.14（PyCharm が作ったプロジェクト内 `.venv`）。numba も ctranslate2 も 3.14 の wheel があるので、Python を下げる必要はありません。

```bash
.venv/bin/pip install -r requirements.txt

# 検証対象を GCS から取る（job_id は生成ジョブの完了通知から、<bucket> は置き場所に読み替え）
gsutil cp gs://<bucket>/music/<job_id>/master.wav  audio/<name>.wav
gsutil cp gs://<bucket>/music/<job_id>/recipe.json recipes/<name>.json

.venv/bin/python -m probe audio/<name>.wav recipes/<name>.json
```

```bash
.venv/bin/pytest                                                     # 回帰テスト(数秒)
.venv/bin/pytest tests/test_vocals.py::test_clean_track_reports_no_violation
.venv/bin/python scripts/calibrate.py    audio/x.wav recipes/x.json  # 実地の陽性対照(数分)
.venv/bin/python scripts/check_lyrics.py audio/x.wav recipes/x.json  # 行落ち検査(数分)
.venv/bin/python -m probe.loudness audio/*.wav                       # 曲間のラウドネス
.venv/bin/python -m probe.spectrum audio/x.wav recipes/x.json        # セクション別の帯域
.venv/bin/python -m probe.balance  audio/x.wav recipes/x.json        # 歌唱区間の「声 − 伴奏」
.venv/bin/python -m probe.corpus   audio/*.wav [--balance]            # 曲を跨いでモード別に並べる
.venv/bin/python -m probe.assessment audio/x.wav assessments/x.json  # 生成側の記録と突き合わせ
```

ジョブのディレクトリに置かれる 4 つのオブジェクトは README の表にあります。

`audio/` `recipes/` `assessments/` `out/` と `*.mp3` `*.wav` は `.gitignore` 済みです。素材はコミットせず、都度 GCS から取り直してください。

## mp3 と master.wav のどちらを測るか

ボーカル検出・ASR は mp3 で十分、マスタリング・配信 QC は master.wav です。理由と実測した差は README にあります（whisper は内部で 16kHz モノラルに落とすため、ASR では差そのものが消えます）。

## アーキテクチャ

`probe/__main__.py`（表示のみ）→ `separate.py`（分離）→ `vocals.py`（測定）→ `recipe.py`（区間の定義）という一方向の流れです。コードを読む前に要る前提が4つあります。

**インスト指定の判定は生成側のプロンプト規約に依存している。** `recipe.py` の `INSTRUMENTAL_MARKER = "Instruments only"` が、そのセクションを楽器のみと見なす唯一の根拠です。歌詞タグ `[Verse]` が `sections[].name` と完全一致する前提（`parse_lyrics`）も同じで、生成側の文言が変われば追随が要ります。

**歌とシンセの区別に帯域エネルギーを使わない。** EDM の saw 系パッドやコードは歌声と同じ帯域に倍音ごと乗るため、帯域だけでは分離できません。`separate.py` は demucs で伴奏を実際に除去してから測ります（MPS で3分の曲に20秒程度、stem が既にあれば再実行しません）。

**閾値は絶対値ではなく曲ごとの相対。** 物差しはその曲の歌唱セクションの RMS 中央値（`vocals.py: sung_reference`）です。`VOCAL_MARGIN_DB` と `ACTIVE_RATIO_THRESHOLD` は役割が違い、**どちらも外せません**。検出できる下限は歌唱に対して約 −18dB で、これは閾値ではなく分離の質で決まる限界です（陽性対照は `scripts/calibrate.py`、実測表は README）。

**セクション境界に接した声は、区間の中身から外して測る。** 歌い出しは境界の手前から始まり（弱起）、歌尾は境界を越えて伸びるため、区間平均が数十 dB 跳ねます。外すのは**隣が歌唱セクションである側の端だけ**で、上限は `BOUNDARY_BLEED_SECONDS`。根拠の実測と、上限を超えた分を途中まで外さない理由は `vocals.py` のコメントにあります。

音源がレシピの宣言尺より短いことは普通に起きます（Lyria が Outro を畳む）。`vocals.py` は各区間を実尺で clamp し、`SectionStats.truncated` で不足を報告します。

### 歌詞照合（lyrics.py）

ASR は素直に使うと、歌われている行を「落ちた」と報告します。踏んだ地雷と効かなかった対処は `transcribe` と `trim_to_voice` の docstring にまとめてあり、変えてはいけないのは3点です。

- **期待した歌詞を `initial_prompt` に渡さない。** 認識率は上がりますが、検証したい対象を答えとして渡すことになります。
- **書き起こしはセクションごとに切る。それでも残る幻聴を止めたのは `hallucination_silence_threshold` だけ。** 末尾の無音詰め、`vad_filter=True`、`no_speech_threshold` を下げるのはいずれも効きませんでした。
- **声が一度も鳴らない区間は書き起こしに回さない**（`trim_to_voice`）。行が丸ごと落ちたセクションはまさにこの形なので、幻聴で埋まると検査そのものが意味を失います。

一致率の数え方と閾値 45% の根拠は `_coverage` / `COVERAGE_THRESHOLD` のコメント、校正に使った実測は `tests/test_lyrics.py::test_threshold_sits_between_the_two_groups` に固定してあります。

### 生成側の記録との突き合わせ（assessment.py）

`assessment.json` は生成側が mp3 を測った記録で（真正ピークと LUFS だけは、`loudness_from_master: true` なら `master.wav` を測った値）、こちらが測るのは wav です。**同じ量を別の対象で測っているので、値は一致しません。**差が符号化と padding で説明が付く範囲かどうかを見るために、`inspector.go` と同じ測り方（チャンネル平均の 1 波形、2 次バンドパス 1 段、50ms 窓の無音判定）を再実装しています。`spectrum.py` の 4 次バターワースとは別物で、混ぜると差の出どころが分からなくなります。

`peak_amplitude` は生成側がチャンネルを平均してから取るピークです。ステレオ差のぶん必ず各チャンネルのピークより低く出ます（実測: 同じ mp3 で ch 別 0.903 / ch 平均 0.867）。

### 帯域計測（spectrum.py）

生成側のマスタリング処理が持つ定数（`acrossover=split=6000 10000`、`acompressor=threshold`）を検証するための計測です。セクション全体の RMS ではなく `_short_term_dbfs`（50ms 窓）を判断に使います。

11曲の測定結果と、それが生成側の閾値相対化（2026-08-15）につながった経緯は README にあります。「圧縮域」の閾値は固定値ではなく、**`assessment.json` の `mastering`（生成側の `MasteringSettings`）から読みます。**記録が無い曲では列を出しません（理由は `BandLevels` の前のコメント）。

**圧縮の下限は 2026-09-05 に 4000Hz から 6000Hz へ動いています。**4-6kHz は子音と存在感の帯域で、そこを含めるとサビの声が自分で閾値を踏むためです（生成側 `masterer.go` の `processingChain` に根拠）。`CROSSOVER_HZ` はチェーンと必ず同じ値にしてください。圧縮対象から外れた 4-6kHz は `PRESENCE_HZ` で切って別列に出しますが、判定には使いません。

## 方針

**判定は出すが、止めない。** 違反を見つけても失敗扱いにはせず、数字と根拠を並べて人間に返します。生成品質の問題はプロンプト調整と手動での作り直しで対処する運用のため、自動ゲートは入れません。

## 既知の問題

- **真正ピークが −1.0 dBTP をわずかに超える曲がある。** 08-04〜08-11 10:45 生成の5曲は超過していました（うち3曲は 0.0 dBTP 以上）。生成側がリミッターの天井を −1 から −2 dBFS に下げて（`masterer.go` の `limiterCeiling`。サンプルピーク基準のままです）直後の6曲は −1.6〜−1.8 dBTP でしたが、10-17〜18 生成の21曲では −0.9〜−1.7 dBTP で、2曲が超えています。天井の根拠（実測の最大超過 1.1dB）からして −0.9 dBTP なので、−2 dBFS では推奨に 0.1dB 届きません。この2曲は生成側の `assessment.json` でも見逃されていました。mp3 を測っていたことと、生成側の測定器が補間でサンプル間の中点を評価していなかったことの 2 つが原因で、2026-10-25 からは `master.wav` を直した測定器で測り、`loudness_from_master: true` が付きます（同じ21曲で ffmpeg との差は最大 0.08dB）。それより前の記録の真正ピークは信用せず、`probe.loudness` で測り直してください。
- **高域圧縮の閾値は要検証。** 相対化のオフセット +2.5dB は、マスタリング後の音源（当時の 4-10kHz）で校正した値を、処理前の入力向けに補正して移したものです（測定方法の差 0.35dB、測定対象の差 1.4dB）。帯域が 09-05 に 6-10kHz へ動いたときも同じ値のままです。**処理前の入力は残らないので、master.wav だけでは確かめられません。**2026-10-29 に生成側が `assessment.json` の `mastering` へ `treble_dbfs` / `treble_threshold_dbfs` / `loudness_gain_db` を残す変更を入れたので、それ以降の曲が溜まれば `probe.corpus` の圧縮量（サビ / Verse）で詰められます。2026-10-28 に 24 曲をログと突き合わせた試算では、オフセットを上げても Verse はほとんど変わらず（+6.0 でも圧縮量 0.35 → 0.20dB）、サビの圧縮と `harsh_treble` の余裕が先に減るため、+2.5 のまま据え置いています（根拠は生成側 `trebleThresholdOffsetDB` のコメント）。それより前の曲は生成側のログ（`マスタリング定数を決めました`）にしか残っていません。
