"""
Step 0: inventory the read-only Melon folder, then build compact caches.

Why cache?
    The raw JSON is slow to parse and Google Drive is slow to read. We parse it
    exactly once and write parquet/npz into the private workspace. Every
    subsequent check reads the cache in seconds, so you can iterate without
    re-reading Drive.

Usage (from repo root):
    python -m src.prepare inventory     # what files exist, how big, sample rows
    python -m src.prepare build         # parse JSON -> parquet caches
    python -m src.prepare build --force # rebuild even if caches exist
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------


def _human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n} B"


def inventory(root: Path | None = None) -> pd.DataFrame:
    """List the top levels of the dataset folder and locate the JSON files."""
    root = Path(root or C.MELON_ROOT)
    print(f"Melon root : {root}")
    print(f"Exists     : {root.exists()}")
    if not root.exists():
        print("\nDrive not mounted, or the path is wrong. Nothing else to do.")
        return pd.DataFrame()

    print("\n--- Top-level contents ---")
    rows = []
    for p in sorted(root.iterdir()):
        try:
            size = p.stat().st_size if p.is_file() else -1
        except OSError:
            size = -1
        kind = "dir" if p.is_dir() else "file"
        rows.append({"name": p.name, "kind": kind, "bytes": size})
        print(f"  [{kind:4}] {p.name:<45} {_human(size) if size >= 0 else ''}")

    print("\n--- Located JSON files ---")
    found = C.find_dataset_files(root)
    for key, name in C.EXPECTED_FILES.items():
        if key in found:
            size = found[key].stat().st_size
            flag = "REQUIRED" if key in C.REQUIRED_KEYS else "optional"
            print(f"  [ok]      {name:<25} {_human(size):>12}   ({flag})")
        else:
            flag = "REQUIRED - MISSING" if key in C.REQUIRED_KEYS else "optional, absent"
            print(f"  [--]      {name:<25} {'':>12}   ({flag})")

    # Peek at one record of each required file so we can verify the schema
    # rather than trusting the documentation.
    for key in C.REQUIRED_KEYS:
        if key not in found:
            continue
        print(f"\n--- First record of {C.EXPECTED_FILES[key]} ---")
        try:
            with open(found[key], "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                rec = data[0]
                if isinstance(rec, dict):
                    for k, v in rec.items():
                        s = repr(v)
                        print(f"  {k:<28} {type(v).__name__:<8} {s[:70]}")
                else:
                    print(f"  (list of {type(rec).__name__})")
                print(f"  -> {len(data):,} records total")
            elif isinstance(data, dict):
                items = list(data.items())[:5]
                for k, v in items:
                    print(f"  {k!r:<28} -> {repr(v)[:70]}")
                print(f"  -> dict with {len(data):,} keys")
            del data
        except Exception as e:  # noqa: BLE001
            print(f"  could not parse: {e}")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cache building
# --------------------------------------------------------------------------

SONGS_PARQUET = "songs.parquet"
INTERACTIONS_PARQUET = "interactions.parquet"
PLAYLISTS_PARQUET = "playlists.parquet"
GENRE_MAP_PARQUET = "genre_map.parquet"
REFERENCED_PARQUET = "referenced_songs.parquet"


def _first_or_none(x):
    if isinstance(x, list) and x:
        return x[0]
    return None


def _song_id(rec: dict):
    """Melon's real schema uses `id`; some documentation says `_id`. Accept both."""
    v = rec.get("id")
    return rec.get("_id") if v is None else v


def build_song_table(path: Path) -> pd.DataFrame:
    """song_meta.json -> one row per track, with list columns preserved."""
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  loaded {len(raw):,} song records in {time.time() - t0:.1f}s")

    ids = [_song_id(r) for r in raw]
    if all(v is None for v in ids):
        keys = sorted(raw[0].keys()) if raw else []
        raise KeyError(
            "Could not find a song id field in song_meta.json.\n"
            f"Available keys: {keys}\n"
            "Expected 'id' (real Melon schema) or '_id' (as documented)."
        )
    n_missing = sum(v is None for v in ids)
    if n_missing:
        print(f"  WARNING: {n_missing:,} records have no id and will be dropped")

    df = pd.DataFrame(
        {
            "song_id": ids,
            "album_id": [r.get("album_id") for r in raw],
            "album_name": [r.get("album_name") for r in raw],
            "song_name": [r.get("song_name") for r in raw],
            "issue_date_raw": [r.get("issue_date") for r in raw],
            "artist_ids": [r.get("artist_id_basket") or [] for r in raw],
            "artist_names": [r.get("artist_name_basket") or [] for r in raw],
            "genres": [r.get("song_gn_gnr_basket") or [] for r in raw],
            "subgenres": [r.get("song_gn_dtl_gnr_basket") or [] for r in raw],
        }
    )
    del raw

    # Hard dtype contract: song_id must be int64 everywhere in this project,
    # otherwise merges against the interaction table silently fail.
    df = df[df["song_id"].notna()].copy()
    df["song_id"] = df["song_id"].astype("int64")

    df["n_artists"] = df["artist_ids"].apply(len)
    df["n_genres"] = df["genres"].apply(len)
    df["n_subgenres"] = df["subgenres"].apply(len)
    # "Primary artist" = first listed. This mirrors ACARec's single-artist
    # assumption, which we keep as a comparison point rather than as the
    # only option (see check 4).
    df["artist_primary"] = df["artist_ids"].apply(_first_or_none)
    df["genre_primary"] = df["genres"].apply(_first_or_none)
    df["subgenre_primary"] = df["subgenres"].apply(_first_or_none)
    return df


def build_interaction_table(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train.json -> (playlist table, long-form playlist x track interactions)."""
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  loaded {len(raw):,} playlists in {time.time() - t0:.1f}s")

    playlists = pd.DataFrame(
        {
            "playlist_id": [r.get("id") for r in raw],
            "title": [r.get("plylst_title") for r in raw],
            "n_tags": [len(r.get("tags") or []) for r in raw],
            "tags": [r.get("tags") or [] for r in raw],
            "n_songs": [len(r.get("songs") or []) for r in raw],
            "like_cnt": [r.get("like_cnt") for r in raw],
            "updt_date": [r.get("updt_date") for r in raw],
        }
    )

    # Long form: one row per (playlist, track, position).
    pl_ids, song_ids, positions = [], [], []
    for r in raw:
        pid = r.get("id")
        songs = r.get("songs") or []
        pl_ids.extend([pid] * len(songs))
        song_ids.extend(songs)
        positions.extend(range(len(songs)))
    del raw

    inter = pd.DataFrame(
        {
            "playlist_id": np.asarray(pl_ids, dtype=np.int64),
            "song_id": np.asarray(song_ids, dtype=np.int64),
            "position": np.asarray(positions, dtype=np.int32),
        }
    )
    print(f"  built {len(inter):,} playlist-track interactions")
    return playlists, inter


def build_referenced_songs(files: dict[str, Path]) -> pd.DataFrame:
    """Which song IDs are referenced by any playlist, in any official split?

    This exists because `song_meta.json` turns out to carry one record per ID
    across the whole 0..707,988 space, not 649,091 real tracks. To report any
    share (e.g. "share of artists with a thin catalog") we must first say what
    the denominator is, and that requires knowing which metadata records
    correspond to tracks the platform actually uses.
    """
    frames = []
    for split in ("train", "val", "test"):
        if split not in files:
            continue
        with open(files[split], "r", encoding="utf-8") as f:
            raw = json.load(f)
        ids = sorted({s for r in raw for s in (r.get("songs") or [])})
        print(f"  {split}.json: {len(raw):,} playlists reference {len(ids):,} distinct songs")
        frames.append(pd.DataFrame({"song_id": np.asarray(ids, dtype=np.int64),
                                    "split": split}))
        del raw

    if not frames:
        return pd.DataFrame({"song_id": pd.Series(dtype="int64")})

    long = pd.concat(frames, ignore_index=True)
    wide = (
        long.assign(v=True)
        .pivot_table(index="song_id", columns="split", values="v",
                     aggfunc="any", fill_value=False)
        .reset_index()
    )
    for split in ("train", "val", "test"):
        if split not in wide.columns:
            wide[split] = False
    wide = wide.rename(columns={s: f"in_{s}" for s in ("train", "val", "test")})
    wide["referenced_anywhere"] = (
        wide["in_train"] | wide["in_val"] | wide["in_test"]
    )
    print(f"  union across splits: {len(wide):,} distinct song IDs referenced")
    return wide


def build_genre_map(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return pd.DataFrame({"code": list(raw.keys()), "name": list(raw.values())})


def build(force: bool = False) -> dict[str, Path]:
    """Parse the raw JSON once and write parquet caches to the workspace."""
    C.ensure_dirs()
    files = C.require_dataset_files()

    out = {
        "songs": C.CACHE_DIR / SONGS_PARQUET,
        "interactions": C.CACHE_DIR / INTERACTIONS_PARQUET,
        "playlists": C.CACHE_DIR / PLAYLISTS_PARQUET,
        "genre_map": C.CACHE_DIR / GENRE_MAP_PARQUET,
        "referenced": C.CACHE_DIR / REFERENCED_PARQUET,
    }

    if not force and all(p.exists() for p in out.values()):
        print("All caches already present. Use --force to rebuild.")
        return out

    print("\n[1/4] song_meta.json")
    songs = build_song_table(files["song_meta"])
    songs.to_parquet(out["songs"], index=False)
    print(f"  wrote {out['songs']}  ({_human(out['songs'].stat().st_size)})")
    del songs

    print("\n[2/4] train.json")
    playlists, inter = build_interaction_table(files["train"])
    playlists.to_parquet(out["playlists"], index=False)
    inter.to_parquet(out["interactions"], index=False)
    print(f"  wrote {out['playlists']}  ({_human(out['playlists'].stat().st_size)})")
    print(f"  wrote {out['interactions']}  ({_human(out['interactions'].stat().st_size)})")
    del playlists, inter

    print("\n[3/4] referenced song IDs across train/val/test")
    ref = build_referenced_songs(files)
    ref.to_parquet(out["referenced"], index=False)
    print(f"  wrote {out['referenced']}  ({_human(out['referenced'].stat().st_size)})")
    del ref

    print("\n[4/4] genre_gn_all.json")
    gmap = build_genre_map(files["genre_map"])
    gmap.to_parquet(out["genre_map"], index=False)
    print(f"  wrote {out['genre_map']}  ({_human(out['genre_map'].stat().st_size)})")

    print("\nCaches built. Subsequent checks read these, not Drive JSON.")
    return out


# --------------------------------------------------------------------------
# Loaders used by the checks
# --------------------------------------------------------------------------


def _load(name: str, filename: str) -> pd.DataFrame:
    path = C.CACHE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Cache missing: {path}\nRun: python -m src.prepare build"
        )
    return pd.read_parquet(path)


def load_songs() -> pd.DataFrame:
    return _load("songs", SONGS_PARQUET)


def load_interactions() -> pd.DataFrame:
    return _load("interactions", INTERACTIONS_PARQUET)


def load_playlists() -> pd.DataFrame:
    return _load("playlists", PLAYLISTS_PARQUET)


def load_referenced() -> pd.DataFrame:
    return _load("referenced", REFERENCED_PARQUET)


def load_genre_map() -> dict[str, str]:
    df = _load("genre_map", GENRE_MAP_PARQUET)
    return dict(zip(df["code"], df["name"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Melon data preparation")
    ap.add_argument("command", choices=["inventory", "build"])
    ap.add_argument("--force", action="store_true", help="rebuild caches")
    args = ap.parse_args(argv)

    if args.command == "inventory":
        inventory()
    else:
        build(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
