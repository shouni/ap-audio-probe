"""Check whether each lyric line was actually sung.

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
    parser.add_argument("--device", help="device for demucs (default: picks mps / cuda / cpu)")
    parser.add_argument("--model", default=lyrics_mod.MODEL, help="faster-whisper model")
    args = parser.parse_args()

    rec = recipe_mod.load(args.recipe)
    if not rec.lyrics:
        raise SystemExit("this recipe has no lyrics (instrumental)")

    print("[1/3] separating the vocal stem ...")
    vocals = separate_vocals(args.audio, args.out, device=args.device)

    print(f"[2/3] transcribing each section ({args.model}) ... the first run takes a while to fetch the model")
    transcripts = lyrics_mod.transcribe(vocals, rec, model_name=args.model)

    print("[3/3] comparing with the lyrics ...")
    print(lyrics_mod.render(lyrics_mod.check(transcripts, rec)))


if __name__ == "__main__":
    main()
