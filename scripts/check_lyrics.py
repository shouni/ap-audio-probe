"""歌詞の行が実際に歌われたかを確かめる。

    .venv/bin/python scripts/check_lyrics.py audio/foo.wav recipes/foo.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe import lyrics as lyrics_mod
from probe import recipe as recipe_mod
from probe.separate import separate_vocals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--device", default="mps", help="demucs の実行デバイス")
    parser.add_argument("--model", default=lyrics_mod.MODEL, help="faster-whisper のモデル")
    args = parser.parse_args()

    rec = recipe_mod.load(args.recipe)
    if not rec.lyrics:
        raise SystemExit("このレシピには歌詞がありません(インスト曲)")

    print("[1/3] ボーカル stem を分離中 ...")
    vocals = separate_vocals(args.audio, args.out, device=args.device)

    print(f"[2/3] セクションごとに書き起こし中 ({args.model}) ... 初回はモデルの取得に時間がかかります")
    transcripts = lyrics_mod.transcribe(vocals, rec, model_name=args.model)

    print("[3/3] 歌詞と突き合わせ中 ...")
    print(lyrics_mod.render(lyrics_mod.check(transcripts, rec)))


if __name__ == "__main__":
    main()
