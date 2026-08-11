"""
Metadata-only checks: 1, 3, 4, 5.

These need song_meta.json and train.json only. No audio, no CF training.
Each writes reports/checkN_*.md with an explicit go/no-go verdict.

    Check 1  Tracks per artist               -> is the thin-catalog premise real?
    Check 3  Group sizes per rung            -> which backoff levels are usable?
    Check 4  Multi-artist rate               -> is collaborator context real?
    Check 5  issue_date validity             -> is a release-time split possible?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import prepare
from .report import Report, bar_figure, bucket_counts, describe_series


# --------------------------------------------------------------------------
# shared derivations
# --------------------------------------------------------------------------


def track_appearance_counts(inter: pd.DataFrame) -> pd.Series:
    """Playlist-appearance count per track (Melon's only popularity proxy)."""
    return inter.groupby("song_id").size().rename("appearances")


def attach_appearances(songs: pd.DataFrame, inter: pd.DataFrame) -> pd.DataFrame:
    counts = track_appearance_counts(inter)
    out = songs.merge(counts, left_on="song_id", right_index=True, how="left")
    out["appearances"] = out["appearances"].fillna(0).astype(int)
    out["is_warm"] = out["appearances"] >= C.WARM_MIN_APPEARANCES
    return out


def explode_groups(songs: pd.DataFrame, level: str) -> pd.DataFrame:
    """Long form (song_id, group_id) for one grouping level.

    artist_primary : first listed artist only (ACARec's assumption)
    artist_any     : every listed artist (Melon-native, multi-membership)
    """
    if level == "artist_primary":
        df = songs[["song_id", "artist_primary"]].rename(
            columns={"artist_primary": "group_id"}
        )
    elif level == "artist_any":
        df = songs[["song_id", "artist_ids"]].explode("artist_ids").rename(
            columns={"artist_ids": "group_id"}
        )
    elif level == "album":
        df = songs[["song_id", "album_id"]].rename(columns={"album_id": "group_id"})
    elif level == "subgenre":
        df = songs[["song_id", "subgenres"]].explode("subgenres").rename(
            columns={"subgenres": "group_id"}
        )
    elif level == "genre":
        df = songs[["song_id", "genres"]].explode("genres").rename(
            columns={"genres": "group_id"}
        )
    else:
        raise ValueError(f"unknown level: {level}")
    return df.dropna(subset=["group_id"])


# --------------------------------------------------------------------------
# Check 1 — tracks per artist
# --------------------------------------------------------------------------


def check1() -> None:
    print("\n=== Check 1: tracks per artist ===")
    songs = prepare.load_songs()
    inter = prepare.load_interactions()
    songs = attach_appearances(songs, inter)

    r = Report(
        1,
        "artist_catalog_sizes",
        "Artist catalog size distribution",
        "Does Melon actually contain enough thin-catalog artists for the "
        "thesis premise to be testable? The mean is ~6 tracks/artist, but the "
        "mean is the wrong statistic — what matters is the median and the "
        "share of artists with 1 or 2 tracks.",
    )

    # -- 1a. Warm/cold vocabulary -----------------------------------------
    n_tracks = len(songs)
    n_warm = int(songs["is_warm"].sum())
    n_zero = int((songs["appearances"] == 0).sum())
    n_le1 = int((songs["appearances"] <= 1).sum())
    r.heading("1a. Warm / cold split of the catalogue")
    r.kv(
        {
            "Tracks in song_meta.json": f"{n_tracks:,}",
            f"Warm (>= {C.WARM_MIN_APPEARANCES} appearances)": f"{n_warm:,} ({n_warm/n_tracks:.1%})",
            "Cold (below threshold)": f"{n_tracks-n_warm:,} ({1-n_warm/n_tracks:.1%})",
            "Never in any train playlist": f"{n_zero:,} ({n_zero/n_tracks:.1%})",
            "In 0 or 1 playlists": f"{n_le1:,} ({n_le1/n_tracks:.1%})",
        }
    )
    r.text(
        "Ferraro et al. report 81,219 warm tracks, i.e. 12.5% of the catalogue. "
        "Any deviation here is because we compute appearances from `train.json` "
        "only (val/test ground truth is private), which is the split we must "
        "use anyway."
    )

    # -- 1b. Catalog size distribution, two artist definitions -------------
    r.heading("1b. Tracks per artist")
    summaries, bucket_tables = [], []
    for level in ["artist_primary", "artist_any"]:
        g = explode_groups(songs, level)
        sizes_all = g.groupby("group_id").size()

        warm_ids = set(songs.loc[songs["is_warm"], "song_id"])
        g_warm = g[g["song_id"].isin(warm_ids)]
        sizes_warm = g_warm.groupby("group_id").size()

        s_all = describe_series(sizes_all, f"{level}: all tracks")
        s_warm = describe_series(sizes_warm, f"{level}: warm tracks only")
        summaries += [s_all.set_index("statistic"), s_warm.set_index("statistic")]

        b = bucket_counts(sizes_warm, C.CATALOG_SIZE_BINS, C.CATALOG_SIZE_LABELS,
                          "warm_catalog_size")
        b.insert(0, "level", level)
        bucket_tables.append(b)

    summary = pd.concat(summaries, axis=1).reset_index()
    r.text(
        "Two definitions are reported. `artist_primary` is the first-listed "
        "artist, which is the assumption ACARec makes. `artist_any` counts a "
        "track under every listed artist, which Melon supports natively."
    )
    r.table(summary.round(2), "catalog_size_summary")

    r.heading("1c. Warm-catalog size buckets (this is the decisive table)", 3)
    r.text(
        "For a cold track, the usable artist context is the artist's **warm** "
        "tracks — the ones that have a CF embedding. A cold track whose artist "
        "has 0 warm tracks cannot be served by any artist-conditioned method "
        "at all; ACARec excludes exactly these."
    )
    buckets = pd.concat(bucket_tables, ignore_index=True)
    r.table(buckets.round(4), "warm_catalog_buckets")

    prim = buckets[buckets["level"] == "artist_primary"]
    thin_share = float(prim[prim["warm_catalog_size"].isin(["1", "2"])]["share"].sum())

    fig = bar_figure(
        prim["warm_catalog_size"],
        prim["share"],
        "Warm tracks by this artist",
        "Share of artists",
        "Warm catalog size per artist (primary-artist definition)",
    )
    r.figure(fig, "warm_catalog_buckets", "Distribution of usable artist context size.")

    # -- 1d. The view that actually matters: per cold track ---------------
    r.heading("1d. Per cold *track*: how much artist context is available?")
    r.text(
        "The previous table counts artists. This one counts cold tracks, which "
        "is what the evaluation will actually iterate over. An artist with a "
        "large catalog contributes many cold tracks, so the two distributions "
        "differ, and this is the one that determines sample sizes per bucket."
    )
    gp = explode_groups(songs, "artist_primary")
    warm_ids = set(songs.loc[songs["is_warm"], "song_id"])
    warm_per_artist = (
        gp[gp["song_id"].isin(warm_ids)].groupby("group_id").size()
    )
    cold = songs[~songs["is_warm"]][["song_id", "artist_primary"]].copy()
    cold["warm_catalog"] = cold["artist_primary"].map(warm_per_artist).fillna(0).astype(int)

    ctab = bucket_counts(
        cold["warm_catalog"].clip(lower=0),
        [0] + C.CATALOG_SIZE_BINS,
        ["0"] + C.CATALOG_SIZE_LABELS,
        "warm_catalog_size",
    )
    r.table(ctab.round(4), "cold_track_context_availability")

    zero_ctx = float(ctab.loc[ctab["warm_catalog_size"] == "0", "share"].iloc[0])
    thin_ctx = float(
        ctab[ctab["warm_catalog_size"].isin(["1", "2"])]["share"].sum()
    )

    fig = bar_figure(
        ctab["warm_catalog_size"],
        ctab["share"],
        "Warm tracks available from this cold track's artist",
        "Share of cold tracks",
        "Artist context available per cold track",
    )
    r.figure(fig, "cold_context_availability",
             "How much artist catalog a cold track can actually attend over.")

    # -- verdict ----------------------------------------------------------
    decision = "GO" if (thin_ctx + zero_ctx) >= 0.15 else "RESHAPE"
    r.verdict(
        decision,
        f"""
{thin_ctx:.1%} of cold tracks come from an artist with only 1-2 warm tracks, and
{zero_ctx:.1%} have no warm artist context at all ({thin_ctx + zero_ctx:.1%} combined).
{thin_share:.1%} of artists hold a warm catalog of 1-2 tracks.

Read this against ACARec, which reports that only 6.5% (M4A-Onion) and 15%
(Yambda) of cold interactions come from cold artists, and which never tests a
context set below 3 items.

- If the combined figure is comfortably above ~15%, the thin-catalog regime is
  well populated and the granularity-router contribution has a real target.
- If it is small, the framing should shift from "thin catalogs" toward
  "heterogeneous catalogs" (check 2 will tell you whether that alternative is
  live), keeping the architecture unchanged.
""",
    )
    r.write()


# --------------------------------------------------------------------------
# Check 3 — group sizes for every candidate rung
# --------------------------------------------------------------------------


def check3() -> None:
    print("\n=== Check 3: group sizes per grouping level ===")
    songs = prepare.load_songs()
    inter = prepare.load_interactions()
    songs = attach_appearances(songs, inter)
    gmap = prepare.load_genre_map()
    warm_ids = set(songs.loc[songs["is_warm"], "song_id"])

    r = Report(
        3,
        "group_sizes",
        "Group sizes across the candidate backoff ladder",
        "Which grouping levels (artist / album / sub-genre / genre) are usable "
        "as rungs? A rung is only useful if its groups are big enough to carry "
        "collaborative signal but small enough to be specific.",
    )

    rows = []
    for level in C.GROUP_LEVELS:
        g = explode_groups(songs, level)
        g_warm = g[g["song_id"].isin(warm_ids)]
        sizes = g_warm.groupby("group_id").size()
        if len(sizes) == 0:
            continue
        rows.append(
            {
                "level": level,
                "n_groups": len(sizes),
                "median_warm_size": float(sizes.median()),
                "mean_warm_size": float(sizes.mean()),
                "p90_warm_size": float(sizes.quantile(0.90)),
                "max_warm_size": int(sizes.max()),
                "share_groups_size_1": float((sizes == 1).mean()),
                "coverage_of_cold_tracks": float(
                    songs.loc[~songs["is_warm"], "song_id"]
                    .isin(set(g["song_id"]))
                    .mean()
                ),
            }
        )

    summary = pd.DataFrame(rows)
    r.heading("3a. Rung comparison")
    r.text(
        "`coverage_of_cold_tracks` is the share of cold tracks that have *any* "
        "group at this level — i.e. how often the rung is even available as a "
        "fallback. `median_warm_size` is the amount of collaborative evidence "
        "the rung typically supplies."
    )
    r.table(summary.round(4), "rung_summary")

    r.heading("3b. Attendability warning")
    r.text(
        "A rung whose median warm size runs into the thousands cannot be "
        "attended over directly — the cross-attention cost and the dilution "
        "both become prohibitive. For such rungs the model must either sample "
        "(e.g. top-n by popularity, as ACARec does at inference) or use the "
        "prototype mean only. Levels flagged below need that treatment."
    )
    summary["attendable_directly"] = summary["median_warm_size"] <= 200
    r.table(
        summary[["level", "median_warm_size", "max_warm_size", "attendable_directly"]],
        "rung_attendability",
    )

    fig = bar_figure(
        summary["level"],
        summary["median_warm_size"],
        "Grouping level",
        "Median warm group size (log)",
        "Evidence available per rung",
    )
    fig.axes[0].set_yscale("log")
    r.figure(fig, "rung_sizes", "Median number of warm tracks per group, by rung.")

    # Top-level genre spot check, since it is the known blow-up risk.
    g = explode_groups(songs, "genre")
    g_warm = g[g["song_id"].isin(warm_ids)]
    top = (
        g_warm.groupby("group_id").size().sort_values(ascending=False).head(15).reset_index()
    )
    top.columns = ["genre_code", "warm_tracks"]
    top["genre_name"] = top["genre_code"].map(gmap)
    r.heading("3c. Largest top-level genre nodes")
    r.table(top, "largest_genre_nodes")

    usable = summary[
        (summary["median_warm_size"] >= 2) & (summary["coverage_of_cold_tracks"] >= 0.5)
    ]["level"].tolist()

    r.verdict(
        "LADDER = " + " > ".join(usable) if usable else "NO USABLE LADDER",
        f"""
Rungs that are both well-populated (median warm size >= 2) and broadly available
(cover >= 50% of cold tracks): {usable}.

Design consequences:
- Rungs marked `attendable_directly = False` enter the model as a prototype mean
  or a popularity-sampled subset, not as a full attention context. This mirrors
  ACARec's finding that ~20 context items saturate performance, so sampling
  costs little.
- If `album` has a median warm size near 1, it is a weak rung on its own but may
  still be useful as a *feature* of the router rather than as an attention target.
- Ito & Shiokawa's log-degree weighting would give the largest genre nodes the
  *most* teleport mass. Our router is designed to do the opposite, and the table
  above is the evidence for why that inversion matters.
""",
    )
    r.write()


# --------------------------------------------------------------------------
# Check 4 — multi-artist rate
# --------------------------------------------------------------------------


def check4() -> None:
    print("\n=== Check 4: multi-artist rate ===")
    songs = prepare.load_songs()
    inter = prepare.load_interactions()
    songs = attach_appearances(songs, inter)

    r = Report(
        4,
        "multi_artist",
        "Multi-artist tracks",
        "ACARec names the single-artist assumption as an explicit limitation. "
        "Melon's `artist_id_basket` is a list, so we can relax it — but only if "
        "enough tracks actually have more than one artist, and only if the "
        "extra artists add context that the primary artist does not.",
    )

    n = len(songs)
    multi = songs["n_artists"] > 1
    r.heading("4a. How common are multi-artist tracks?")
    r.kv(
        {
            "Tracks total": f"{n:,}",
            "With 0 listed artists": f"{int((songs['n_artists'] == 0).sum()):,}",
            "With exactly 1 artist": f"{int((songs['n_artists'] == 1).sum()):,} ({(songs['n_artists'] == 1).mean():.1%})",
            "With >1 artist": f"{int(multi.sum()):,} ({multi.mean():.1%})",
            "Max artists on one track": int(songs["n_artists"].max()),
            "Multi-artist rate among COLD tracks": f"{multi[~songs['is_warm']].mean():.1%}",
            "Multi-artist rate among WARM tracks": f"{multi[songs['is_warm']].mean():.1%}",
        }
    )

    dist = songs["n_artists"].value_counts().sort_index().head(10).reset_index()
    dist.columns = ["n_artists", "n_tracks"]
    dist["share"] = dist["n_tracks"] / n
    r.table(dist.round(5), "n_artists_distribution")

    # -- 4b. Does the collaborator add context the primary artist lacks? ---
    r.heading("4b. Does a collaborator actually widen the context?")
    r.text(
        "The interesting case is a cold track whose *primary* artist is thin "
        "but whose *secondary* artist is warm. For those tracks, relaxing the "
        "single-artist assumption converts an unservable case into a servable "
        "one. This is the concrete payoff of the multi-artist extension."
    )
    warm_ids = set(songs.loc[songs["is_warm"], "song_id"])
    gp = explode_groups(songs, "artist_primary")
    warm_per_artist = gp[gp["song_id"].isin(warm_ids)].groupby("group_id").size()

    cold_multi = songs[(~songs["is_warm"]) & multi].copy()
    if len(cold_multi):
        cold_multi["ctx_primary"] = (
            cold_multi["artist_primary"].map(warm_per_artist).fillna(0).astype(int)
        )
        cold_multi["ctx_any"] = cold_multi["artist_ids"].apply(
            lambda ids: int(sum(warm_per_artist.get(a, 0) for a in ids))
        )
        cold_multi["gain"] = cold_multi["ctx_any"] - cold_multi["ctx_primary"]
        rescued = (cold_multi["ctx_primary"] < 3) & (cold_multi["ctx_any"] >= 3)
        r.kv(
            {
                "Cold multi-artist tracks": f"{len(cold_multi):,}",
                "Median context, primary artist only": int(cold_multi["ctx_primary"].median()),
                "Median context, all listed artists": int(cold_multi["ctx_any"].median()),
                "Mean context gained": f"{cold_multi['gain'].mean():.2f}",
                "Rescued (ctx <3 -> >=3 warm tracks)": f"{int(rescued.sum()):,} ({rescued.mean():.1%} of cold multi-artist)",
                "Rescued as share of ALL cold tracks": f"{int(rescued.sum())/max(int((~songs['is_warm']).sum()),1):.2%}",
            }
        )
        rescue_share = float(int(rescued.sum()) / max(int((~songs["is_warm"]).sum()), 1))
    else:
        rescue_share = 0.0

    multi_rate = float(multi.mean())
    if multi_rate >= 0.10 and rescue_share >= 0.01:
        decision = "GO — promote to a first-class model input"
    elif multi_rate >= 0.05:
        decision = "PARTIAL — keep as an ablation, not a headline"
    else:
        decision = "DROP — mention as an unexploited Melon affordance"

    r.verdict(
        decision,
        f"""
{multi_rate:.1%} of tracks list more than one artist; the collaborator extension
rescues {rescue_share:.2%} of all cold tracks from a thin (<3) to a usable (>=3)
artist context.

- High rate + meaningful rescue: `artist_any` becomes a rung in its own right,
  sitting between `artist_primary` and `album` on the ladder, and the thesis
  directly relaxes a limitation ACARec states but cannot address on its data.
- Low rate: keep the single-artist assumption for the main model and report the
  multi-artist variant as one ablation row, noting that Melon supports it even
  though the payoff here is small.
""",
    )
    r.write()


# --------------------------------------------------------------------------
# Check 5 — issue_date validity
# --------------------------------------------------------------------------


def _parse_issue_date(v) -> tuple[int | None, int | None, int | None, str]:
    """Return (year, month, day, status). Melon stores YYYYMMDD as str or int."""
    if v is None:
        return None, None, None, "missing"
    s = str(v).strip()
    if not s or s in {"0", "00000000"}:
        return None, None, None, "zero"
    if len(s) != 8 or not s.isdigit():
        return None, None, None, "malformed"
    y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
    if y < 1900 or y > 2025:
        return y, m, d, "implausible_year"
    if m == 0 and d == 0:
        return y, None, None, "year_only"
    if d == 0:
        return y, m, None, "year_month_only"
    if not (1 <= m <= 12) or not (1 <= d <= 31):
        return y, m, d, "bad_month_day"
    return y, m, d, "full"


def check5() -> None:
    print("\n=== Check 5: issue_date validity ===")
    songs = prepare.load_songs()
    inter = prepare.load_interactions()
    songs = attach_appearances(songs, inter)

    r = Report(
        5,
        "issue_date",
        "Release-date quality and temporal split feasibility",
        "Can we build a release-time cold split (train on tracks released "
        "before T, treat post-T tracks as cold)? That requires `issue_date` to "
        "be present, well-formed, and to produce a usable number of cold tracks "
        "with enough playlist appearances to evaluate against.",
    )

    parsed = songs["issue_date_raw"].apply(_parse_issue_date)
    songs["year"] = [p[0] for p in parsed]
    songs["month"] = [p[1] for p in parsed]
    songs["date_status"] = [p[3] for p in parsed]

    status = songs["date_status"].value_counts().reset_index()
    status.columns = ["status", "n_tracks"]
    status["share"] = status["n_tracks"] / len(songs)
    r.heading("5a. Parse status")
    r.text(
        "`full` = usable to the day. `year_only` / `year_month_only` are common "
        "placeholder patterns in large catalogues and are still usable for a "
        "year-level cutoff. Anything else must be dropped and reported."
    )
    r.table(status.round(5), "issue_date_status")

    usable = songs["year"].notna()
    r.kv(
        {
            "Tracks with a usable year": f"{int(usable.sum()):,} ({usable.mean():.1%})",
            "Tracks with full YYYYMMDD": f"{int((songs['date_status']=='full').sum()):,}",
            "Unusable": f"{int((~usable).sum()):,} ({(~usable).mean():.1%})",
        }
    )

    # -- 5b. Year distribution -------------------------------------------
    r.heading("5b. Release-year distribution (tracks appearing in train.json)")
    in_train = songs[songs["appearances"] > 0]
    ydist = (
        in_train[in_train["year"].notna()]
        .groupby("year")
        .agg(n_tracks=("song_id", "size"), total_appearances=("appearances", "sum"))
        .reset_index()
        .sort_values("year")
    )
    recent = ydist[ydist["year"] >= 2000]
    r.table(recent.tail(30), "release_year_distribution")

    fig = bar_figure(
        recent["year"].astype(int),
        recent["n_tracks"],
        "Release year",
        "Tracks in train playlists",
        "Release-year distribution (2000+)",
    )
    fig.axes[0].tick_params(axis="x", rotation=70)
    r.figure(fig, "release_years", "Melon collection ends mid-2020, so 2020 is partial.")

    # -- 5c. Candidate cutoffs -------------------------------------------
    r.heading("5c. Candidate temporal cutoffs")
    r.text(
        "For each cutoff T: tracks released on/after T become the cold set. We "
        "report how many such tracks exist, how many playlist appearances they "
        "attract (this is the evaluation ground truth, so it must not be tiny), "
        "and what fraction of them have a warm artist — i.e. how many our "
        "method can serve at all."
    )
    warm_ids = set(songs.loc[songs["is_warm"], "song_id"])
    gp = explode_groups(songs, "artist_primary")

    cut_rows = []
    for cutoff in [2016, 2017, 2018, 2019, 2020]:
        pre = songs[(songs["year"].notna()) & (songs["year"] < cutoff)]
        post = songs[(songs["year"].notna()) & (songs["year"] >= cutoff)]
        if len(post) == 0:
            continue
        # Warm artist status must be computed from PRE-cutoff tracks only,
        # otherwise the split leaks.
        pre_ids = set(pre["song_id"])
        gp_pre_warm = gp[gp["song_id"].isin(pre_ids & warm_ids)]
        warm_per_artist_pre = gp_pre_warm.groupby("group_id").size()
        ctx = post["artist_primary"].map(warm_per_artist_pre).fillna(0)
        cut_rows.append(
            {
                "cutoff_year": cutoff,
                "cold_tracks": len(post),
                "cold_appearances": int(post["appearances"].sum()),
                "cold_tracks_with_>=1_appearance": int((post["appearances"] > 0).sum()),
                "share_with_warm_artist": float((ctx >= 1).mean()),
                "share_with_>=3_warm_artist_tracks": float((ctx >= 3).mean()),
                "median_artist_context": float(ctx.median()),
            }
        )
    cuts = pd.DataFrame(cut_rows)
    r.table(cuts.round(4), "temporal_cutoff_candidates")

    best = None
    if len(cuts):
        viable = cuts[cuts["cold_tracks_with_>=1_appearance"] >= 5000]
        if len(viable):
            best = int(viable.iloc[-1]["cutoff_year"])

    usable_share = float(usable.mean())
    if usable_share >= 0.90 and best:
        decision = f"GO — temporal split viable, suggested cutoff {best}"
    elif usable_share >= 0.90:
        decision = "PARTIAL — dates are clean but no cutoff yields enough cold interactions"
    else:
        decision = "FREQUENCY SPLIT ONLY"

    r.verdict(
        decision,
        f"""
{usable_share:.1%} of tracks yield a usable release year.

The thesis should report **both** split protocols and treat the agreement
between them as a robustness result:

1. **Frequency split** (appearances < {C.WARM_MIN_APPEARANCES} = cold). Comparable to
   Ferraro et al.'s published Melon numbers, which is what makes our motivating
   figures (nDCG 0.0395 warm -> 0.0014 cold) meaningful in our own setting.
2. **Release-time split** (issue_date >= T = cold). Closer to the real deployment
   scenario and immune to the circularity of defining coldness by an outcome.

Note the leakage trap this check already controls for: an artist's "warm" status
at cutoff T must be computed from pre-T tracks only. Computing it globally would
let post-cutoff information define the training context.
""",
    )
    r.write()
