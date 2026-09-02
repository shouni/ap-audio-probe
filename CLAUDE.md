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
```

`audio/` `recipes/` `out/` と `*.mp3` `*.wav` は `.gitignore` 済みです。素材はコミットせず、都度 GCS から取り直してください。

## mp3 と master.wav のどちらを測るか

生成側は同じディレクトリに `audio.mp3`（192kbps）と `master.wav` の両方を置きます。

- **ボーカル検出・ASR は mp3 で十分。** 判定に使う差が 25dB あるのに対し符号化による差は 0.1dB 台で、結論は動きません。whisper は内部で 16kHz モノラルに落とすため差そのものが消えます。
- **マスタリング・配信 QC は master.wav。** mp3 は符号化で波形がオーバーシュートし、ピークが元より高く出ます。配信に出すのは wav なので、出荷物を測るという原則からもこちらです。

実測した差の表は README にあります。

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

### 帯域計測（spectrum.py）

生成側のマスタリング処理が持つ定数（`acrossover=split=4000 10000`、`acompressor=threshold`）を検証するための計測です。セクション全体の RMS ではなく `_short_term_dbfs`（50ms 窓）を判断に使います。

11曲の測定結果と、それが生成側の閾値相対化（2026-08-15）につながった経緯は README にあります。**`spectrum.py` の `COMPRESSOR_THRESHOLD_DBFS` は固定値のままなので、08-15 より後に生成された曲では「圧縮域」の列は実際にかかった量ではありません。**

## 方針

**判定は出すが、止めない。** 違反を見つけても失敗扱いにはせず、数字と根拠を並べて人間に返します。生成品質の問題はプロンプト調整と手動での作り直しで対処する運用のため、自動ゲートは入れません。

## 既知の問題

- **`--device mps` が既定でフォールバックがない。** Apple Silicon 以外では `--device cpu` の明示が要ります。
- **真正ピークの超過は 2026-08-11 の日中で止まっている（解決済みと思われる）。** 手元の11曲で測ると、−1.0 dBTP を超えるのは 08-04〜**08-11 10:45** 生成の5曲（うち3曲は 0.0 dBTP 以上）で、**08-11 13:01** 以降の6曲はすべて −1.6〜−1.8 dBTP に収まります。境界は 08-11 の当日中で、この間にマスタリング処理が真正ピーク基準に変わったと思われます。生成側のコミット履歴で裏を取るのが確実です。超過していた頃はサンプルピークが −1.00 dBFS ちょうどに揃っていたので、リミッターがサンプルピーク基準で、サンプル間のピークを見ていなかったと思われます。新しい曲では再現しないため、**当時の出力を配信に回す場合にだけ問題**になります。
- **Integrated LUFS の揃い（生成側で対処済み・要検証）。** 手元の11曲で 1.7 LU 差（−13.2〜−11.5、目安 1.0 LU）でした。マスタリング処理に音量合わせが入っていなかったためで、2026-08-15 に目標 −12.5 LUFS へ寄せる処理が足されました。同じ11曲を新しいチェーンに通すと 0.2 LU に収まり、真正ピークも −1.5〜−2.2 dBTP のままでした。ただしこの検証はマスタリング済みの音源を入力に見立てた再現なので、**08-15 より後に生成された曲を `probe.loudness` で測り直すまでは確定ではありません**。
- **高域圧縮の閾値も同じく要検証。** 相対化のオフセット +2.5dB は2通りの導き方が一致した値ですが、校正に使えたのはマスタリング後の音源だけでした。生成側が測るのは処理前の入力なので、測定方法の差（scipy 4次バターワースと `acrossover` で 0.35dB）と測定対象の差（処理後は 4-10kHz が 1.4dB 低い）を補正して移しています。**08-15 より後に生成された曲を測り直して、実データで詰める必要があります。**
- 生成側の Audio Check は配信用の mp3 を測っています。デコーダの padding を含むため尺が 2,304 サンプル＝MP3 フレーム2つぶん長く出ます。ただし実測した限りどの閾値も余裕のほうが桁で大きく、判定が反転するケースは見つかっていません。あわせて `peak_amplitude: 0.867` が wav（0.891）とも mp3（0.903）とも一致しない理由が未解明です。
