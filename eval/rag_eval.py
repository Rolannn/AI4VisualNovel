#!/usr/bin/env python3
"""
RAG Retrieval Quality Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Measures top-k recall against manually curated ground-truth query-answer pairs.
Distinguishes **retrieval failures** (keyword in corpus but not recalled) from
**corpus gaps** (keyword absent from entire knowledge base).

Usage:
    python eval/rag_eval.py                # default: test Chainsaw Man KB
    python eval/rag_eval.py --kb <path>    # custom knowledge base JSON
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.knowledge_builder import KnowledgeBuilder  # noqa: E402


# ============================================================
# Ground Truth: query -> list of expected keywords / phrases
# ============================================================
# Design principles (after v2 review):
#   - Every keyword MUST exist somewhere in the corpus (verified).
#   - Keywords must represent *incremental knowledge* — information
#     you can only learn by retrieving the correct passage, not by
#     simply reading the query.
#   - Stem prefixes (e.g. "manipulat") are used intentionally to
#     tolerate morphological variation.

GROUND_TRUTH: Dict[str, List[str]] = {
    # ---- Makima ----
    "Makima personality control manipulative": [
        "Control Devil",        # core identity — only in Appearances section
        "manipulat",            # stem: manipulating / manipulative
        "fear",                 # "born from humanity's fear of control"
    ],
    "Makima appearance gentle woman": [
        "ringed",               # objective trait: ringed eyes
        "braid",                # objective trait: braided hair
        "uniform",              # Public Safety uniform
    ],
    "Makima relationship with Denji obedience dog": [
        "obedien",              # stem: obedience — core dynamic
        "threaten",             # threatens to hunt him down
        "companion",            # destroys his companion Pochita
    ],
    # ---- Denji ----
    "Denji chainsaw devil transformation protagonist": [
        "Pochita",              # key transformation medium
        "heart",                # Pochita becomes his heart
        "cord",                 # pull cord on chest triggers transform
    ],
    "Denji creation design Fujimoto inspiration": [
        "Fire Punch",           # Fujimoto's prior work
        "Adventure Time",       # explicit interview reference
        "Finn",                 # Adventure Time character cited
    ],
    # ---- Power ----
    "Power blood fiend abilities": [
        "Blood Fiend",          # core race designation
        "horn",                 # distinctive physical feature + power indicator
        "weapons",              # blood-based weapon creation (in list page)
    ],
    "Power personality Cartman selfish": [
        "Cartman",              # meta-knowledge: South Park reference
        "South Park",           # source of the inspiration
        "childish",             # personality description in Appearances
    ],
    # ---- Aki Hayakawa ----
    "Aki Hayakawa devil hunter Public Safety": [
        "Fox Devil",            # latent: contract devil (in list page Devils section)
        "Curse",                # latent: Curse Devil contract
        "Gun Devil",            # latent: his revenge target
    ],
    # ---- World / Franchise ----
    "Chainsaw Man manga story world overview": [
        "Fujimoto",             # creator name
        "Shonen Jump",          # serialization venue (partial match via "Jump")
        "combat",               # world-building: combat against Devils
    ],
    "Public Safety Devil Hunters government organization": [
        "government",           # organizational nature
        "organization",         # explicit description
        "Tokyo Special",        # specific division (Tokyo Special Division 4)
    ],
}


# ============================================================
# Corpus audit helper
# ============================================================

def audit_corpus(kb: KnowledgeBuilder, ground_truth: Dict[str, List[str]]):
    """
    Check which ground-truth keywords actually exist in the corpus.
    Returns (in_corpus, not_in_corpus) keyword sets.
    """
    corpus_text = " ".join(d["text"] for d in kb.store.documents).lower()

    in_corpus: List[str] = []
    not_in_corpus: List[str] = []
    for keywords in ground_truth.values():
        for kw in keywords:
            if kw.lower() in corpus_text:
                in_corpus.append(kw)
            else:
                not_in_corpus.append(kw)
    return in_corpus, not_in_corpus


# ============================================================
# Recall@K computation
# ============================================================

def recall_at_k(
    kb: KnowledgeBuilder,
    query: str,
    expected_keywords: List[str],
    k: int = 3,
    franchise_filter: str = "",
    corpus_text: str = "",
) -> Tuple[float, int, int, List[str], List[str], List[str]]:
    """
    Compute recall@k for a single query.

    Returns:
        (recall, hits, total, hit_keywords, retrieval_misses, corpus_gaps)
    """
    results = kb.search(query, n_results=k, franchise_filter=franchise_filter)
    retrieved_text = " ".join(r["text"] for r in results).lower()

    hit_kw: List[str] = []
    retrieval_miss: List[str] = []
    corpus_gap: List[str] = []

    for kw in expected_keywords:
        kw_lower = kw.lower()
        if kw_lower in retrieved_text:
            hit_kw.append(kw)
        elif corpus_text and kw_lower in corpus_text:
            retrieval_miss.append(kw)   # exists in corpus but not retrieved
        else:
            corpus_gap.append(kw)       # not in corpus at all

    hits = len(hit_kw)
    total = len(expected_keywords)
    return hits / total if total else 1.0, hits, total, hit_kw, retrieval_miss, corpus_gap


# ============================================================
# Main evaluation routine
# ============================================================

def run_eval(kb_path: str, k_values: List[int] = None, franchise: str = "Chainsaw Man"):
    if k_values is None:
        k_values = [1, 3, 5]

    # Load knowledge base
    kb = KnowledgeBuilder(store_path=kb_path)
    doc_count = kb.get_document_count()
    corpus_text = " ".join(d["text"] for d in kb.store.documents).lower()

    print("=" * 72)
    print("  RAG Retrieval Quality Evaluation  (v2 — strict incremental GT)")
    print("=" * 72)
    print(f"  Knowledge base : {kb_path}")
    print(f"  Total documents: {doc_count}")
    print(f"  Queries        : {len(GROUND_TRUTH)}")
    print(f"  K values       : {k_values}")
    print("=" * 72)

    # ── Corpus audit ──
    in_corpus, not_in_corpus = audit_corpus(kb, GROUND_TRUTH)
    all_kw = sum(len(v) for v in GROUND_TRUTH.values())
    print(f"\n  Corpus Audit: {len(in_corpus)}/{all_kw} keywords found in corpus")
    if not_in_corpus:
        print(f"  WARNING — {len(not_in_corpus)} keywords NOT in corpus (will show as 'CORPUS GAP'):")
        for kw in not_in_corpus:
            print(f"    - \"{kw}\"")
    else:
        print("  All ground-truth keywords verified present in corpus.")

    # ── Per-k results ──
    for k in k_values:
        print(f"\n{'─' * 72}")
        print(f"  Recall@{k}")
        print(f"{'─' * 72}")

        recalls = []
        total_retrieval_misses = 0
        total_corpus_gaps = 0

        for query, expected in GROUND_TRUTH.items():
            recall, hits, total, hit_kw, ret_miss, corp_gap = recall_at_k(
                kb, query, expected, k=k,
                franchise_filter=franchise, corpus_text=corpus_text,
            )
            recalls.append(recall)
            total_retrieval_misses += len(ret_miss)
            total_corpus_gaps += len(corp_gap)

            status = "PASS" if recall == 1.0 else "MISS"
            print(f"  [{status}] {recall:.2f}  ({hits}/{total})  Q: \"{query}\"")
            if ret_miss:
                print(f"         retrieval miss : {ret_miss}")
            if corp_gap:
                print(f"         corpus gap     : {corp_gap}")

        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        perfect = sum(1 for r in recalls if r == 1.0)
        print(f"\n  >> Overall Recall@{k} = {avg_recall:.4f}  "
              f"({perfect}/{len(recalls)} queries fully recalled)")
        print(f"     Retrieval misses: {total_retrieval_misses}  |  Corpus gaps: {total_corpus_gaps}")

    # ── Summary table ──
    print(f"\n{'=' * 72}")
    print("  Summary")
    print(f"{'=' * 72}")
    header = "  {:>12s}" + " | {:>10s}" * len(k_values)
    print(header.format("Metric", *[f"Recall@{k}" for k in k_values]))
    print("  " + "-" * (14 + 13 * len(k_values)))

    avg_row = []
    perfect_row = []
    for k in k_values:
        recalls = []
        for query, expected in GROUND_TRUTH.items():
            r, _, _, _, _, _ = recall_at_k(
                kb, query, expected, k=k,
                franchise_filter=franchise, corpus_text=corpus_text,
            )
            recalls.append(r)
        avg_row.append(sum(recalls) / len(recalls))
        perfect_row.append(sum(1 for r in recalls if r == 1.0))

    data_fmt = "  {:>12s}" + " | {:>10s}" * len(k_values)
    print(data_fmt.format("Avg Recall", *[f"{v:.4f}" for v in avg_row]))
    print(data_fmt.format("Full Recall", *[f"{v}/{len(GROUND_TRUTH)}" for v in perfect_row]))
    print(f"{'=' * 72}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Quality Evaluation")
    parser.add_argument(
        "--kb",
        default=os.path.join(PROJECT_ROOT, "data", "rag", "knowledge_Chainsaw_Man.json"),
        help="Path to the knowledge base JSON file",
    )
    parser.add_argument(
        "--franchise",
        default="Chainsaw Man",
        help="Franchise name for metadata filtering",
    )
    parser.add_argument(
        "-k",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="K values to evaluate (default: 1 3 5)",
    )
    args = parser.parse_args()
    run_eval(kb_path=args.kb, k_values=args.k, franchise=args.franchise)


if __name__ == "__main__":
    main()
