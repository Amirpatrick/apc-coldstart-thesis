"""
Central configuration: paths, constants, and file discovery.

Design rules for this repo
--------------------------
1. The Melon dataset folder is READ-ONLY. Nothing is ever written there.
2. Large derived artefacts (parquet caches, CF factors) go to the Drive
   WORKSPACE, which is NOT committed to git.
3. Small human-readable outputs (markdown reports, CSV tables, PNG figures)
   go to REPORT_DIR inside the repo, and ARE committed. These are what the
   committee reads.

Override any path with an environment variable before importing, e.g.:
    import os
    os.environ["MELON_ROOT"] = "/some/other/path"
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

#: Read-only Melon dataset root (Drive shortcut to the owner's shared folder).
MELON_ROOT = Path(
    os.environ.get(
        "MELON_ROOT",
        "/content/drive/MyDrive/Last Attempt/Thesis_Data_Access/melon-dataset",
    )
)

#: Your own Drive workspace. Big derived files live here and survive between
#: Colab sessions. Created automatically; never committed to git.
WORKSPACE = Path(
    os.environ.get(
        "THESIS_WORKSPACE",
        "/content/drive/MyDrive/Last Attempt/thesis_workspace",
    )
)

CACHE_DIR = WORKSPACE / "cache"        # parquet + npz, regenerable
LOG_DIR = WORKSPACE / "logs"

#: Repo root = parent of src/. Reports go inside the repo so they are versioned.
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

# --------------------------------------------------------------------------
# Dataset file names we expect inside MELON_ROOT (searched recursively)
# --------------------------------------------------------------------------

EXPECTED_FILES = {
    "song_meta": "song_meta.json",
    "train": "train.json",
    "val": "val.json",
    "test": "test.json",
    "genre_map": "genre_gn_all.json",
}

#: Files the five data checks actually require. The 240 GB of arena_mel_*.tar
#: is NOT needed for any of them.
REQUIRED_KEYS = ["song_meta", "train", "genre_map"]

# --------------------------------------------------------------------------
# Analysis constants
# --------------------------------------------------------------------------

#: Ferraro et al. (ICASSP 2021) treat tracks appearing in >= 10 playlists as
#: warm and everything else as cold-start. We reuse the threshold so our
#: warm/cold vocabulary matches the dataset's own published baseline.
WARM_MIN_APPEARANCES = 10

#: Ferraro et al. discard playlists shorter than this before building splits.
MIN_PLAYLIST_LEN = 5

#: Catalog-size buckets used throughout. The thesis lives in the first three.
CATALOG_SIZE_BINS = [1, 2, 3, 5, 10, 20, 50, 10**9]
CATALOG_SIZE_LABELS = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50+"]

#: Collaborative-filtering teacher (matches the WRMF/BPR-MF setups used by the
#: cold-start baselines we will later compare against).
CF_FACTORS = 128
CF_REGULARIZATION = 0.01
CF_ITERATIONS = 20
CF_ALPHA = 40.0

#: Reproducibility.
SEED = 42

#: Cap on pairs sampled per group when measuring cohesion, so that a genre node
#: covering ~21k tracks does not blow up the pairwise computation.
MAX_PAIRS_PER_GROUP = 2000

#: Grouping levels examined as candidate "rungs" of the backoff ladder.
GROUP_LEVELS = ["artist_primary", "artist_any", "album", "subgenre", "genre"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def ensure_dirs() -> None:
    """Create every writable directory. Never touches MELON_ROOT."""
    for d in (CACHE_DIR, LOG_DIR, REPORT_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def find_dataset_files(root: Path | None = None, max_depth: int = 3) -> dict[str, Path]:
    """Locate the Melon JSON files under `root`, whatever the folder layout.

    Returns a dict mapping logical key -> resolved Path, omitting anything
    not found. Search is depth-limited so we never walk into the extracted
    spectrogram directories.
    """
    root = Path(root or MELON_ROOT)
    if not root.exists():
        raise FileNotFoundError(
            f"MELON_ROOT does not exist: {root}\n"
            "Check that Drive is mounted and the shortcut path is correct."
        )

    wanted = {name: key for key, name in EXPECTED_FILES.items()}
    found: dict[str, Path] = {}
    root_depth = len(root.parts)

    for path in root.rglob("*.json"):
        if len(path.parts) - root_depth > max_depth:
            continue
        key = wanted.get(path.name)
        if key and key not in found:
            found[key] = path

    return found


def require_dataset_files(root: Path | None = None) -> dict[str, Path]:
    """Like find_dataset_files, but raises if a required file is missing."""
    found = find_dataset_files(root)
    missing = [k for k in REQUIRED_KEYS if k not in found]
    if missing:
        raise FileNotFoundError(
            "Missing required Melon file(s): "
            + ", ".join(EXPECTED_FILES[k] for k in missing)
            + f"\nSearched under: {root or MELON_ROOT}\n"
            "Run `python -m src.prepare inventory` to see what is actually there."
        )
    return found
