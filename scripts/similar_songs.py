"""Show the songs most similar to a given song (pgvector nearest-neighbour search).

Usage:
    uv run python scripts/similar_songs.py 42            # 5 nearest by CLAP
    uv run python scripts/similar_songs.py 42 -k 10 --model muq

Prints each song's prompt and audio path, so you can listen and judge.
"""

import argparse

from sqlalchemy import select

from music_gen.config import get_settings
from music_gen.db import GenerationJob, Prompt, Song, make_session_factory
from music_gen.features.store import nearest_songs


def describe(session, song_id: int) -> str:
    text, seed, path = session.execute(
        select(Prompt.text, GenerationJob.seed, Song.audio_path)
        .join(GenerationJob, Song.job_id == GenerationJob.id)
        .join(Prompt, GenerationJob.prompt_id == Prompt.id)
        .where(Song.id == song_id)
    ).one()
    return f"{text!r} (seed {seed})  data/{path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Find similar songs")
    parser.add_argument("song_id", type=int)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--model", choices=["clap", "muq"], default="clap")
    args = parser.parse_args()

    settings = get_settings()
    model_name = {
        "clap": f"clap:{settings.clap_model_id}",
        "muq": f"muq-mulan:{settings.muq_model_id}",
    }[args.model]
    with make_session_factory(settings.database_url)() as session:
        print(f"Song {args.song_id}: {describe(session, args.song_id)}\n")
        for rank, (song_id, distance) in enumerate(
            nearest_songs(session, args.song_id, model_name, args.k), 1
        ):
            print(f"{rank}. song {song_id}  similarity {1 - distance:.3f}")
            print(f"   {describe(session, song_id)}")


if __name__ == "__main__":
    main()
