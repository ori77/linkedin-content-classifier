"""
STEP 4 - Group the posts into clusters (draft sub-topics).

Reads : embeddings.npy + embeddings_meta.json   (step 3)
        keyphrases.jsonl                         (step 2)
Writes: clusters.csv        - one row per post: which cluster it belongs to
        clusters_report.md  - one section per cluster, for YOU to read

Usage (PowerShell, venv active):
    pip install umap-learn scikit-learn
    python step4_cluster.py keyphrases.jsonl --sweep                   (compare settings first)
    python step4_cluster.py keyphrases.jsonl --method leaf --min-cluster-size 12

With embeddings made by the subject-only experiment, plus a second pass over the noise:
    python step4_cluster.py keyphrases.jsonl --emb-prefix embeddings_subject ^
        --method leaf --min-cluster-size 12 --rescue-noise --out-prefix clusters_final
"""

import argparse
import csv
import json
import math
import sys
import warnings
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import umap
    from sklearn.cluster import HDBSCAN   # built into scikit-learn 1.3 or newer
except ImportError:
    sys.exit("Missing packages. Run:  pip install umap-learn scikit-learn\n"
             "(HDBSCAN needs scikit-learn 1.3 or newer: pip install -U scikit-learn)")

from step3_embeddings import load_keyphrases, clean_phrases, make_drop_pattern

# UMAP prints a harmless note when we fix the random seed (it then runs on one CPU core).
warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
# scikit-learn announces a future default change in HDBSCAN; it doesn't affect our results.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Words too common to describe a cluster.
STOPWORDS = {"and", "for", "in", "of", "to", "the", "a", "an", "on", "with", "vs",
             "from", "as", "at", "by", "is", "how", "your", "you", "it", "its"}

MIN_RESCUE_POOL = 30     # fewer noise posts than this: a second pass is pointless


# ----------------------------------------------------------------------------
# 1. LOADING
# ----------------------------------------------------------------------------
def load_inputs(prefix, keyphrases_path):
    """Returns vectors, ids, original phrases per post, and the phrases AS THE MODEL SAW THEM."""
    vectors = np.load(f"{prefix}.npy")
    with open(f"{prefix}_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    ids = meta["activity_ids"]
    if len(ids) != len(vectors):
        sys.exit(f"{prefix}.npy has {len(vectors)} rows but the meta file lists {len(ids)} posts")
    phrases_by_id = load_keyphrases(keyphrases_path)
    missing = [i for i in ids if i not in phrases_by_id]
    if missing:
        sys.exit(f"{len(missing)} embedded posts have no keyphrases. Re-run step 3 after step 2.")

    # Step 3 may have ignored some words (e.g. 'interview'). Rebuild exactly what it saw,
    # so the top words and the report describe what the clustering really used.
    drop_pattern = make_drop_pattern(meta.get("dropped_words", []))
    seen_by_id = {i: clean_phrases(phrases_by_id[i], drop_pattern) for i in ids}
    return vectors, ids, phrases_by_id, seen_by_id


# ----------------------------------------------------------------------------
# 2. THE TWO ALGORITHMS
# ----------------------------------------------------------------------------
def squash(vectors, n_neighbors, n_components, seed):
    """UMAP: 768 numbers per post -> 10 numbers per post, keeping 'who is near whom'."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,   # how many nearby posts each post looks at to learn the local shape
        n_components=n_components, # how many numbers per post we keep
        min_dist=0.0,              # allow points to sit tightly packed (good for clustering)
        metric="cosine",           # 'near' means similar meaning, same as step 3
        random_state=seed,         # same seed -> same result every run
    )
    return reducer.fit_transform(vectors)


def find_clusters(reduced, min_cluster_size, min_samples, method):
    """HDBSCAN: find dense groups. Label -1 means 'not clearly in any group' (noise)."""
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=method,
    )
    return model.fit_predict(reduced)


def centre_similarities(vectors, rows):
    """How close each of these posts is to the centre of the group (1.0 = identical)."""
    centre = vectors[rows].mean(axis=0)
    centre = centre / (np.linalg.norm(centre) or 1.0)
    return vectors[rows] @ centre


def peel_outliers(vectors, rows, bar, min_keep):
    """Remove the most out-of-place post, one at a time, until every remaining post is at
    least `bar` close to the group's centre. One at a time, because junk posts drag the
    centre away from the real members; removing them gradually lets the centre recover.
    Returns the surviving rows, or an empty array if fewer than `min_keep` can survive."""
    rows = list(rows)
    while len(rows) >= min_keep:
        sims = centre_similarities(vectors, np.array(rows))
        worst = int(np.argmin(sims))
        if sims[worst] >= bar:
            return np.array(rows)
        rows.pop(worst)
    return np.array([], dtype=int)


def rescue_noise(vectors, labels, n_neighbors, n_components, seed,
                 min_cluster_size, min_samples, method, trim_percentile=10):
    """A second pass over ONLY the noise posts, with a smaller minimum cluster size.

    The first pass asks for crowds of at least 12 posts, so a real but small subject
    (say 8 posts about local LLMs) is left as noise. Here the leftovers are squashed
    and clustered on their own, so such small subjects can appear.
    The clusters from the first pass are never touched.

    SAFETY GUARD: a bag of unrelated leftovers can still produce junk "clusters".
    So a rescued post must be at least as close to its new cluster's centre as the
    least-typical `trim_percentile` percent of posts in the first-pass clusters.
    Posts that fail go back to noise, and a cluster needs at least `min_cluster_size`
    survivors to be kept.

    Returns (new_labels, set_of_new_cluster_ids).
    """
    noise_rows = np.where(labels == -1)[0]
    if len(noise_rows) < MIN_RESCUE_POOL:
        return labels, set()

    # The bar a rescued post has to clear, learned from the first-pass clusters.
    first_pass = [centre_similarities(vectors, np.where(labels == c)[0])
                  for c in sorted(set(labels.tolist()) - {-1})]
    bar = float(np.percentile(np.concatenate(first_pass), trim_percentile)) if first_pass else 0.0

    # We are hunting for SMALL groups, so each post should only look at about that many
    # neighbours; with 15 its view would spill over into unrelated groups.
    # (UMAP also needs n_neighbors and n_components to be smaller than the number of points.)
    neighbours = min(n_neighbors, max(min_cluster_size, 5), len(noise_rows) - 1)
    components = min(n_components, len(noise_rows) - 2)
    reduced = squash(vectors[noise_rows], neighbours, components, seed)
    sub_labels = find_clusters(reduced, min_cluster_size, min_samples, method)

    new_labels = labels.copy()
    next_id = max(int(labels.max()) + 1, 0)
    rescued = set()
    for c in sorted(set(sub_labels.tolist()) - {-1}):
        rows = noise_rows[sub_labels == c]
        keep = peel_outliers(vectors, rows, bar, min_cluster_size)
        if len(keep) >= min_cluster_size:
            new_labels[keep] = next_id
            rescued.add(next_id)
            next_id += 1
    return new_labels, rescued


# ----------------------------------------------------------------------------
# 3. DESCRIBING EACH CLUSTER
# ----------------------------------------------------------------------------
def words_of(phrases):
    """All meaningful words in a post's keyphrases (years and filler removed)."""
    words = []
    for phrase in phrases:
        for w in phrase.replace("-", " ").split():
            if w not in STOPWORDS and len(w) > 1:
                words.append(w)
    return words


def top_words_per_cluster(labels, word_lists, how_many=8):
    """c-TF-IDF: words that are frequent INSIDE a cluster but rare in the other clusters."""
    clusters = sorted(set(int(l) for l in labels) - {-1})
    counts = {c: Counter() for c in clusters}
    for label, words in zip(labels, word_lists):
        if label != -1:
            counts[int(label)].update(words)

    overall = Counter()
    for c in clusters:
        overall.update(counts[c])
    avg_words_per_cluster = sum(overall.values()) / max(len(clusters), 1)

    result = {}
    for c in clusters:
        total = sum(counts[c].values()) or 1
        scored = {
            w: (n / total) * math.log(1 + avg_words_per_cluster / overall[w])
            for w, n in counts[c].items()
        }
        result[c] = [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:how_many]]
    return result


def representatives(vectors, labels, cluster, how_many=6):
    """Row numbers of the posts closest to the cluster's centre (its 'typical' posts)."""
    rows = np.where(labels == cluster)[0]
    order = np.argsort(-centre_similarities(vectors, rows))
    return [int(rows[i]) for i in order[:how_many]]


# ----------------------------------------------------------------------------
# 4. WRITING THE RESULTS
# ----------------------------------------------------------------------------
def stage_of(label, rescued):
    if label == -1:
        return "noise"
    return "rescued" if label in rescued else "cluster"


def write_csv(path, ids, labels, rescued, phrases_by_id, seen_by_id):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["activity_id", "cluster", "stage", "keyphrases", "keyphrases_used"])
        for post_id, label in zip(ids, labels):
            label = int(label)
            writer.writerow([post_id, label, stage_of(label, rescued),
                             "; ".join(phrases_by_id[post_id]),
                             "; ".join(seen_by_id[post_id])])


def write_report(path, args, ids, labels, rescued, vectors, seen_by_id, top_words):
    sizes = Counter(int(l) for l in labels)
    noise = sizes.pop(-1, 0)
    ordered = sorted(sizes, key=lambda c: -sizes[c])
    main_count = len([c for c in ordered if c not in rescued])

    lines = [
        "# Cluster report (draft sub-topics)",
        "",
        f"- Posts: **{len(ids)}** | Clusters: **{len(ordered)}** "
        f"({main_count} main + {len(rescued)} rescued from noise) | "
        f"Unclustered (noise): **{noise}** ({noise / len(ids):.0%})",
        f"- Settings: min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}, "
        f"method={args.method}, n_neighbors={args.n_neighbors}, seed={args.seed}, "
        f"embeddings={args.emb_prefix}"
        + (f", rescue_min_size={args.rescue_min_size}" if args.rescue_noise else ""),
        "",
        "Phrases shown are exactly what the embedding model saw (years and any dropped "
        "words removed). The CSV has both versions.",
        "Clusters are ordered biggest first. Numbers are labels only, with no meaning in the order.",
        "",
    ]
    for c in ordered:
        tag = " (rescued from noise)" if c in rescued else ""
        lines.append(f"## Cluster {c} ({sizes[c]} posts){tag}")
        lines.append(f"**Top words:** {', '.join(top_words[c])}")
        lines.append("")
        lines.append("Typical posts (keyphrases):")
        for row in representatives(vectors, labels, c):
            lines.append(f"- {'; '.join(seen_by_id[ids[row]])}")
        lines.append("")

    if noise:
        lines.append(f"## Unclustered ({noise} posts)")
        lines.append("A few examples:")
        for row in np.where(labels == -1)[0][:10]:
            lines.append(f"- {'; '.join(seen_by_id[ids[row]])}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------------
# 5. TRYING SEVERAL SETTINGS AT ONCE
# ----------------------------------------------------------------------------
def sweep(reduced, min_samples):
    """Cluster the same squashed data with several settings and print a comparison.
    UMAP (the slow part) ran once; HDBSCAN is fast, so trying many settings is cheap."""
    print(f"\n{'method':<7}{'min size':>9}{'clusters':>10}{'noise':>8}{'largest':>9}")
    for method in ("leaf", "eom"):
        for size in (6, 8, 10, 12, 15, 20):
            labels = find_clusters(reduced, size, min_samples, method)
            sizes = Counter(int(l) for l in labels)
            noise = sizes.pop(-1, 0)
            largest = max(sizes.values()) if sizes else 0
            print(f"{method:<7}{size:>9}{len(sizes):>10}{noise / len(labels):>8.0%}{largest:>9}")


# ----------------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keyphrases_path")
    parser.add_argument("--emb-prefix", default="embeddings")
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--method", choices=["leaf", "eom"], default="leaf")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="clusters")
    parser.add_argument("--rescue-noise", action="store_true",
                        help="second pass over the noise posts with a smaller minimum size")
    parser.add_argument("--rescue-min-size", type=int, default=6)
    parser.add_argument("--rescue-trim-percentile", type=float, default=10,
                        help="strictness of the junk guard (lower = more lenient)")
    parser.add_argument("--sweep", action="store_true",
                        help="only compare several settings in a table; write no files")
    args = parser.parse_args()

    vectors, ids, phrases_by_id, seen_by_id = load_inputs(args.emb_prefix, args.keyphrases_path)
    print(f"Loaded {len(ids)} posts, {vectors.shape[1]} numbers each")

    print("Squashing with UMAP...")
    reduced = squash(vectors, args.n_neighbors, args.n_components, args.seed)

    if args.sweep:
        sweep(reduced, args.min_samples)
        return

    print("Finding clusters with HDBSCAN...")
    labels = find_clusters(reduced, args.min_cluster_size, args.min_samples, args.method)
    first_pass_noise = int((labels == -1).sum())

    rescued = set()
    if args.rescue_noise:
        print(f"Second pass over {first_pass_noise} noise posts "
              f"(min cluster size {args.rescue_min_size})...")
        labels, rescued = rescue_noise(vectors, labels, args.n_neighbors, args.n_components,
                                       args.seed, args.rescue_min_size, args.min_samples,
                                       args.method, args.rescue_trim_percentile)
        recovered = first_pass_noise - int((labels == -1).sum())
        print(f"  rescued {len(rescued)} new clusters, {recovered} posts moved out of noise")

    word_lists = [words_of(seen_by_id[i]) for i in ids]
    top_words = top_words_per_cluster(labels, word_lists)

    sizes = Counter(int(l) for l in labels)
    noise = sizes.pop(-1, 0)
    ranked = sorted(sizes.values(), reverse=True)
    print(f"\nClusters: {len(sizes)} | noise: {noise} ({noise / len(ids):.0%})")
    if ranked:
        print(f"Biggest 5 sizes: {ranked[:5]} | smallest: {ranked[-1]}")
    for c in sorted(sizes, key=lambda c: -sizes[c])[:10]:
        print(f"  cluster {c:>3} ({sizes[c]:>3} posts): {', '.join(top_words[c][:5])}")

    write_csv(f"{args.out_prefix}.csv", ids, labels, rescued, phrases_by_id, seen_by_id)
    write_report(f"{args.out_prefix}_report.md", args, ids, labels, rescued, vectors,
                 seen_by_id, top_words)
    print(f"\nSaved {args.out_prefix}.csv and {args.out_prefix}_report.md")


if __name__ == "__main__":
    main()