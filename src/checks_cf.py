"""
Check 2 — does Melon's CF space cluster by group, and does that survive thin groups?

This is the load-bearing check. ACARec's whole architecture rests on an
unstated assumption: that an artist's CF item embeddings are "often centered
around a shared artist-specific vector", which is why the artist mean works as
a GRU anchor. Their ablation shows that removing the anchor drops the model to
baseline level. So if Melon's CF space does not have that structure, the
prototype-anchor family does not transfer and the thesis must be re-rung.

What we measure
---------------
For a group g (an artist, album, sub-genre...) with warm members M:

    leave-one-out prototype similarity
        s_i = cos( e_i , mean(E_{M \\ {i}}) )

This is precisely the quantity ACARec's GRU consumes as its hidden state, so it
is the right diagnostic — more so than plain within-group pairwise cosine.

Two null models, because a raw cosine has no meaning on its own:

    global null   cos(e_i, mean of all warm embeddings)
    size-matched  cos(e_i, mean of |M|-1 random warm embeddings)

The size-matched null matters: leave-one-out means over 2 items are noisy, so a
small group could look weak for purely statistical reasons. Comparing against a
random group of the *same size* removes that confound.

Everything is then stratified by group size, which answers the actual thesis
question: does the prototype anchor still carry signal when the catalog is thin?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as C
from . import prepare
from .checks_meta import attach_appearances, explode_groups
from .report import Report

FACTORS_NPZ = "cf_item_factors.npz"


# --------------------------------------------------------------------------
# CF teacher
# --------------------------------------------------------------------------


def build_warm_matrix(
    songs: pd.DataFrame, inter: pd.DataFrame
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Playlist x warm-track binary matrix, plus the song_id for each column."""
    warm = songs.loc[songs["is_warm"], "song_id"].to_numpy()
    warm_sorted = np.sort(warm)
    col_of = {s: i for i, s in enumerate(warm_sorted)}

    sub = inter[inter["song_id"].isin(set(warm_sorted))]
    rows_u, row_of = pd.factorize(sub["playlist_id"].to_numpy())
    cols = sub["song_id"].map(col_of).to_numpy()

    mat = sp.csr_matrix(
        (np.ones(len(sub), dtype=np.float32), (rows_u, cols)),
        shape=(len(row_of), len(warm_sorted)),
    )
    print(f"  matrix: {mat.shape[0]:,} playlists x {mat.shape[1]:,} warm tracks, "
          f"{mat.nnz:,} nnz, density {mat.nnz / (mat.shape[0]*mat.shape[1]):.2e}")
    return mat, warm_sorted


def _fit_als(mat: sp.csr_matrix, use_gpu: bool) -> np.ndarray:
    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=C.CF_FACTORS,
        regularization=C.CF_REGULARIZATION,
        iterations=C.CF_ITERATIONS,
        random_state=C.SEED,
        use_gpu=use_gpu,
    )
    # Confidence weighting c = 1 + alpha*r, applied by scaling the data.
    model.fit((mat * C.CF_ALPHA).tocsr())
    factors = model.item_factors
    if hasattr(factors, "to_numpy"):  # implicit returns a GPU matrix wrapper
        factors = factors.to_numpy()
    return np.asarray(factors, dtype=np.float32)


def train_cf(mat: sp.csr_matrix) -> np.ndarray:
    """Train an implicit-ALS teacher.

    Tries GPU, falls back to CPU, then falls back to TruncatedSVD. implicit's
    HAS_CUDA is a *build* flag, so it can be True on a CPU-only Colab runtime;
    we therefore never trust it without a try/except around the actual fit.
    """
    try:
        import implicit  # noqa: F401
    except ImportError:
        print("  implicit not installed -> TruncatedSVD fallback "
              "(weaker teacher; `pip install implicit` for the real thing)")
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=C.CF_FACTORS, random_state=C.SEED)
        svd.fit(mat)
        return np.asarray(svd.components_.T, dtype=np.float32)

    try:
        from implicit.gpu import HAS_CUDA
    except Exception:  # noqa: BLE001
        HAS_CUDA = False

    if HAS_CUDA:
        try:
            print(f"  implicit ALS on GPU (factors={C.CF_FACTORS})")
            return _fit_als(mat, use_gpu=True)
        except Exception as e:  # noqa: BLE001
            print(f"  GPU fit failed ({type(e).__name__}: {e}); retrying on CPU")

    print(f"  implicit ALS on CPU (factors={C.CF_FACTORS})")
    return _fit_als(mat, use_gpu=False)


def get_item_factors(force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (factors [n_warm, d], song_ids [n_warm]), cached to the workspace."""
    C.ensure_dirs()
    path = C.CACHE_DIR / FACTORS_NPZ
    if path.exists() and not force:
        z = np.load(path)
        print(f"  loaded cached factors {z['factors'].shape} from {path.name}")
        return z["factors"], z["song_ids"]

    songs = prepare.load_songs()
    inter = prepare.load_interactions()
    songs = attach_appearances(songs, inter)
    mat, song_ids = build_warm_matrix(songs, inter)
    factors = train_cf(mat)
    np.savez_compressed(path, factors=factors, song_ids=song_ids)
    print(f"  cached factors -> {path}")
    return factors, song_ids


# --------------------------------------------------------------------------
# Cohesion measurement
# --------------------------------------------------------------------------


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def group_cohesion(
    factors_n: np.ndarray,
    members_by_group: dict,
    rng: np.random.Generator,
    max_groups: int = 20000,
    max_members: int = 200,
) -> pd.DataFrame:
    """Leave-one-out prototype similarity vs a size-matched random null."""
    n_items = factors_n.shape[0]
    global_mean = _l2(factors_n.mean(axis=0, keepdims=True))[0]

    keys = [g for g, m in members_by_group.items() if len(m) >= 2]
    if len(keys) > max_groups:
        keys = list(rng.choice(np.array(keys, dtype=object), max_groups, replace=False))

    rows = []
    for g in keys:
        idx = np.asarray(members_by_group[g], dtype=np.int64)
        if len(idx) > max_members:
            idx = rng.choice(idx, max_members, replace=False)
        n = len(idx)
        E = factors_n[idx]                       # already L2-normalised

        # leave-one-out prototypes
        total = E.sum(axis=0, keepdims=True)
        proto = _l2((total - E) / (n - 1))
        loo = np.einsum("ij,ij->i", E, proto)

        # size-matched random null, chunked so a 200-member group cannot
        # allocate a (200, 199, d) tensor in one go on the real catalogue
        null_parts = []
        chunk = max(1, 4096 // max(n - 1, 1))
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            rnd_idx = rng.integers(0, n_items, size=(e - s, n - 1))
            rnd_proto = _l2(factors_n[rnd_idx].mean(axis=1))
            null_parts.append(np.einsum("ij,ij->i", E[s:e], rnd_proto))
        null = np.concatenate(null_parts)

        glob = E @ global_mean

        rows.append(
            {
                "group_id": g,
                "size": n,
                "loo_proto_cos": float(loo.mean()),
                "size_matched_null_cos": float(null.mean()),
                "global_null_cos": float(glob.mean()),
                "lift_vs_size_matched": float(loo.mean() - null.mean()),
            }
        )
    return pd.DataFrame(rows)


def _stratify(df: pd.DataFrame) -> pd.DataFrame:
    bins = [1, 2, 3, 5, 10, 20, 50, 10**9]
    labels = ["2", "3-4", "5-9", "10-19", "20-49", "50+"]
    # sizes >= 2 only; first bin edge 1 makes size 2 land in the first label
    cat = pd.cut(df["size"], bins=[1, 2, 4, 9, 19, 49, 10**9], labels=labels)
    out = (
        df.assign(size_bucket=cat)
        .groupby("size_bucket", observed=False)
        .agg(
            n_groups=("group_id", "size"),
            loo_proto_cos=("loo_proto_cos", "mean"),
            size_matched_null_cos=("size_matched_null_cos", "mean"),
            lift=("lift_vs_size_matched", "mean"),
        )
        .reset_index()
    )
    return out


def check2(force_cf: bool = False) -> None:
    print("\n=== Check 2: CF-space group cohesion ===")
    rng = np.random.default_rng(C.SEED)

    factors, warm_song_ids = get_item_factors(force=force_cf)
    factors_n = _l2(factors.astype(np.float32))
    row_of = {int(s): i for i, s in enumerate(warm_song_ids)}

    songs = prepare.load_songs()
    inter = prepare.load_interactions()
    songs = attach_appearances(songs, inter)

    r = Report(
        2,
        "cf_group_cohesion",
        "Does the CF space cluster by group?",
        "ACARec anchors its prediction on the mean CF embedding of the "
        "artist's catalog, and its ablation shows that without this anchor the "
        "model is no better than the baseline. That only works if the CF space "
        "is group-clustered. Does Melon's CF space have that structure — and "
        "crucially, does it survive when the group has only 2 or 3 members?",
    )

    r.heading("2a. CF teacher")
    r.kv(
        {
            "Warm tracks (columns)": f"{len(warm_song_ids):,}",
            "Embedding dimension": factors.shape[1],
            "Model": f"implicit ALS, reg={C.CF_REGULARIZATION}, "
                     f"iters={C.CF_ITERATIONS}, alpha={C.CF_ALPHA}",
            "Seed": C.SEED,
        }
    )
    r.text(
        "Matrix factorisation on the playlist x track matrix, warm tracks only. "
        "This is the same family of teacher used by DeepMusic, Heater, GAR and "
        "ACARec, and by Ferraro's own Melon baseline (WARP-MF), so the "
        "diagnostic transfers to whichever teacher we settle on."
    )

    # -- per level --------------------------------------------------------
    all_strat, headline = [], []
    warm_id_set = set(row_of.keys())
    for level in C.GROUP_LEVELS:
        print(f"  [{level}] exploding...", flush=True)
        g = explode_groups(songs, level)
        g = g[g["song_id"].isin(warm_id_set)].copy()
        if len(g) == 0:
            print(f"  [{level}] no warm members, skipped")
            continue
        g["row"] = g["song_id"].map(row_of)
        members = g.groupby("group_id")["row"].apply(list).to_dict()
        n_multi = sum(1 for m in members.values() if len(m) >= 2)
        print(f"  [{level}] {len(members):,} groups "
              f"({n_multi:,} with >=2 warm members); measuring cohesion...",
              flush=True)

        coh = group_cohesion(factors_n, members, rng)
        if len(coh) == 0:
            print(f"  [{level}] no group had >=2 warm members, skipped")
            continue
        coh.insert(0, "level", level)

        strat = _stratify(coh)
        strat.insert(0, "level", level)
        all_strat.append(strat)

        headline.append(
            {
                "level": level,
                "n_groups_size>=2": len(coh),
                "loo_proto_cos": coh["loo_proto_cos"].mean(),
                "size_matched_null": coh["size_matched_null_cos"].mean(),
                "global_null": coh["global_null_cos"].mean(),
                "lift": coh["lift_vs_size_matched"].mean(),
                "share_groups_with_positive_lift": float(
                    (coh["lift_vs_size_matched"] > 0).mean()
                ),
            }
        )

    head = pd.DataFrame(headline)
    r.heading("2b. Cohesion by grouping level")
    r.text(
        "`loo_proto_cos` is the cosine between a warm track's CF vector and the "
        "mean of its group-mates. `size_matched_null` is the same quantity "
        "against an equally sized random set. `lift` is the difference — the "
        "only number that means anything on its own. A lift near zero says the "
        "group carries no collaborative structure beyond chance."
    )
    r.table(head.round(4), "cohesion_by_level")

    strat_all = pd.concat(all_strat, ignore_index=True)
    r.heading("2c. Cohesion by group size — the decisive table")
    r.text(
        "This is what check 1 cannot tell you. If lift stays roughly flat as "
        "group size falls to 2-3, a thin catalog is still informative and the "
        "prototype anchor is safe everywhere. If lift decays sharply, then the "
        "artist mean is *unreliable exactly where the thesis says it is* — "
        "which is direct empirical support for the router contribution, since "
        "the router's job is to detect that and back off."
    )
    r.table(strat_all.round(4), "cohesion_by_size")

    # figure: lift vs size, one line per level
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for level, sub in strat_all.groupby("level"):
        ax.plot(sub["size_bucket"].astype(str), sub["lift"], marker="o", label=level)
    ax.axhline(0, ls="--", c="grey", lw=1)
    ax.set_xlabel("Group size (warm members)")
    ax.set_ylabel("Prototype cohesion lift over size-matched null")
    ax.set_title("Does group structure survive thin groups?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    r.figure(fig, "cohesion_by_size",
             "If these curves fall toward zero on the left, thin groups are "
             "unreliable and the granularity router has an empirical mandate.")

    # -- verdict ----------------------------------------------------------
    best = head.sort_values("lift", ascending=False).iloc[0] if len(head) else None
    art = head[head["level"] == "artist_primary"]
    art_lift = float(art["lift"].iloc[0]) if len(art) else float("nan")

    # Compare thin vs deep groups at the ARTIST level specifically -- that is
    # the rung the thesis argument is about. Averaging across all levels would
    # mix in genre, whose group sizes are never thin.
    art_strat = strat_all[strat_all["level"] == "artist_primary"]
    thin = art_strat[art_strat["size_bucket"].astype(str).isin(["2", "3-4"])]
    deep = art_strat[art_strat["size_bucket"].astype(str).isin(["20-49", "50+"])]
    thin_lift = float(thin["lift"].mean()) if thin["lift"].notna().any() else float("nan")
    deep_lift = float(deep["lift"].mean()) if deep["lift"].notna().any() else float("nan")

    if np.isnan(thin_lift) or np.isnan(deep_lift):
        direction = "not computable (a size stratum is empty)"
    elif thin_lift < deep_lift * 0.6:
        direction = (
            "thin groups carry MUCH LESS structure than deep ones "
            "-> supports the router framing"
        )
    elif thin_lift < deep_lift * 0.9:
        direction = "thin groups carry somewhat less structure than deep ones"
    else:
        direction = (
            "thin groups carry AS MUCH structure as deep ones "
            "-> the thin-catalog framing is NOT supported; pivot to heterogeneity"
        )

    if art_lift > 0.10:
        artist_call = "artist clustering is strong; ACARec's anchor transfers"
    elif art_lift > 0.03:
        artist_call = "artist clustering is present but modest; anchor transfers weakly"
    else:
        artist_call = "artist clustering is essentially absent; the anchor does NOT transfer"

    r.verdict(
        f"Primary rung = {best['level'] if best is not None else 'n/a'} "
        f"(lift {best['lift']:.3f})" if best is not None else "INCONCLUSIVE",
        f"""
Artist-level lift = {art_lift:.3f} -> {artist_call}.
Strongest rung overall: {best['level'] if best is not None else 'n/a'}.

At the artist level: thin groups (2-4 warm tracks) lift = {thin_lift:.3f};
deep groups (20+ warm tracks) lift = {deep_lift:.3f}.
Reading: {direction}.

Note that the null is size-matched, so a difference across size buckets is a
statement about structure, not about the extra variance of averaging fewer
vectors — that confound is already removed.

How to read this for the thesis:

- **Artist lift high AND decay small** -> the ACARec anchor transfers cleanly and
  thin catalogs are still informative. The router's value then comes from
  *heterogeneous* rather than *thin* catalogs; reframe around dispersion.
- **Artist lift high AND decay large** -> the ideal case for the stated framing.
  The anchor works where catalogs are deep and degrades where they are thin,
  which is exactly the failure the granularity router is designed to catch.
  Quote this number in the problem statement.
- **Artist lift low, album or sub-genre higher** -> Melon's ~2.4 tracks/album and
  singles-driven Korean release culture mean artist is the wrong primary unit.
  Re-rung the ladder around the winning level. The architecture is unchanged and
  the finding itself ("artist is the wrong grouping unit in a singles-driven
  catalogue") is a legitimate contribution.
- **All lifts near zero** -> the prototype-anchor family does not transfer to
  Melon at all. Stop, and either switch the anchor to a learned group embedding
  or reconsider the dataset. This is the one genuinely blocking outcome.
""",
    )
    r.write()
