# Cold-track insertion in Automatic Playlist Continuation

Master's thesis code. Stage 0: **pre-commitment data checks** on the Melon
Playlist Dataset.

These five checks are run *before* any modelling, and each one ends in an
explicit go / reshape / stop verdict. The point is that the direction of the
thesis is decided by measured properties of the data, not by assumption.

---

## Why these five checks

The thesis proposes conditioning a cold track's representation on **which
grouping level to trust** (artist → album → sub-genre → genre) and **which
members within it to attend to**, using the seed playlist as the conditioning
signal. That design rests on four empirical claims. Each check tests one.

| Check | Question | What it decides | Kills / reshapes the thesis if |
|---|---|---|---|
| **1** | How many tracks does the average artist actually have? | Whether the thin-catalog regime is populated enough to study | Thin catalogs are rare → pivot from "thin" to "heterogeneous" |
| **2** | Does the CF space cluster by artist/album/genre, and does that survive small groups? | Whether ACARec's prototype anchor transfers to Melon at all | No group cohesion anywhere → the whole prototype-anchor family fails |
| **3** | How big is each grouping level? | Which rungs of the backoff ladder are real, and which need sampling rather than attention | No rung has usable size → the ladder collapses to one level |
| **4** | How many tracks have >1 artist? | Whether collaborator context is a contribution or a footnote | Rate too low → demote to a single ablation row |
| **5** | Is `issue_date` clean enough for a release-time split? | Whether we can run both a frequency split and a temporal split | Dates unusable → frequency split only |

Check 2 is the load-bearing one. ACARec's own ablation shows that removing the
artist-mean anchor drops their model to baseline level, so if Melon's CF space
lacks group structure, the architecture does not transfer.

---

## Setup (Colab + Drive + GitHub only — nothing local)

Open `notebooks/00_data_checks.ipynb` in Colab and run it top to bottom. It:

1. mounts Drive,
2. sets `MELON_ROOT` (the shared read-only folder) and `THESIS_WORKSPACE`
   (your own folder for caches),
3. clones this repo into `/content`,
4. installs `requirements.txt`,
5. runs the checks and displays the verdicts.

### Paths

```python
MELON_ROOT       = /content/drive/MyDrive/Last Attempt/Thesis_Data_Access/melon-dataset
THESIS_WORKSPACE = /content/drive/MyDrive/Last Attempt/thesis_workspace
```

`MELON_ROOT` is **never written to**. All derived files go to the workspace.

### Command line

```bash
python run_checks.py inventory     # what files exist; verify the schema
python run_checks.py build         # JSON -> parquet caches (run once)
python run_checks.py all           # checks 1-5
python run_checks.py 1 3 4 5       # metadata only, fast, no CF training
python run_checks.py 2 --force-cf  # retrain the CF teacher
```

---

## What gets stored where, and why

| Location | Contents | In git? |
|---|---|---|
| `MELON_ROOT` (Drive, shared) | Raw Melon dataset | No — read-only, never modified |
| `THESIS_WORKSPACE/cache` (Drive) | parquet caches, CF factors (`.npz`) | No — large, regenerable |
| `reports/` (this repo) | Markdown findings, CSV tables, PNG figures | **Yes** |

`reports/` is the evidence trail. Everything the committee needs to audit a
preprocessing decision is a versioned file with a timestamp, a seed, and the
threshold that produced it.

The 240 GB of `arena_mel_*.tar` spectrograms are **not touched** by any of these
checks. They are only needed later, for the audio content channel.

---

## Data derivation policy

Melon's official `val.json` / `test.json` ground truth is held privately on the
Kakao Arena platform, which has been unavailable since February 2024. The
dataset's own baseline (Ferraro et al., ICASSP 2021) constructs an alternative
split for exactly this reason, and we follow the same approach: **all splits are
derived from `train.json`.**

Three rules govern every derived split in this project.

**1. Filters are fixed in advance, from task requirements.**
Each filter is justified by what the task needs (e.g. a playlist must have ≥5
tracks for a seed/target split to be definable), never by which threshold
produces the best result. Filter rules and their cumulative effect on
playlist / track / artist counts are reported in `reports/`.

**2. Every method sees the identical dataset.**
Baselines and the proposed model train and evaluate on the same frozen split,
candidate pool, and metric implementation. Consequently **published numbers are
not copied**: Ferraro's Melon figures and the RecSys 2018 leaderboard are used
as motivation, and all baselines are re-implemented and re-run here.

**3. The phenomenon under study is never filtered away.**

| Entity | Filtering allowed | Must be preserved |
|---|---|---|
| Playlists | Yes — minimum length | — |
| Warm / training tracks | Yes — minimum appearances, so the CF teacher is trainable | — |
| **Cold / test tracks** | **No popularity or catalog-size filter** | full thin-catalog spectrum |
| **Artists** | **No minimum catalog size** | artists with 1, 2, 3 tracks |

Rule 3 matters specifically here: a routine `k`-core filter on artists would
delete the exact population the thesis is about.

---

## Methodological notes

**Warm threshold.** ≥10 playlist appearances, matching Ferraro et al., so our
warm/cold vocabulary lines up with the dataset's own published baseline.

**Two artist definitions.** `artist_primary` (first listed, matching ACARec's
stated single-artist assumption) and `artist_any` (every listed artist, which
Melon supports natively). Both are reported throughout; check 4 decides whether
the difference is material.

**Size-matched null in check 2.** A raw within-group cosine is meaningless on
its own, and leave-one-out means over 2 items are noisier than over 50 — so a
thin group could look weak for purely statistical reasons. Every cohesion
number is therefore reported against a null built from an equally sized random
set, and only the *lift* is interpreted.

**No sequence models.** Melon's `songs` order has no documented semantics and no
per-track add timestamps exist (`updt_date` is playlist-level). All playlist
encoding is set-based. LARP makes the same choice.

**Leakage control in check 5.** When evaluating a temporal cutoff `T`, an
artist's "warm" status is computed from pre-`T` tracks only. Computing it
globally would let post-cutoff information define the training context.

---

## Layout

```
src/config.py        paths, thresholds, seeds — single source of truth
src/prepare.py       inventory + JSON -> parquet caching
src/report.py        markdown / CSV / figure emitters
src/checks_meta.py   checks 1, 3, 4, 5 (metadata only)
src/checks_cf.py     check 2 (CF teacher + cohesion)
run_checks.py        CLI entry point
notebooks/           Colab drivers
reports/             committed findings, tables, figures
```

## Reproducing

Every check is deterministic given `SEED = 42` in `src/config.py`. Reports carry
a UTC timestamp, the seed, and the warm threshold in their header.
