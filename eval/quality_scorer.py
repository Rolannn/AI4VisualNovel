"""
Story Quality Scorer
~~~~~~~~~~~~~~~~~~~~
Computes interpretable quality scores (0–1) for AI-generated visual novels.

Structural features (from game_design.json):
    branching_factor   – avg children per non-leaf node
    path_entropy       – Shannon entropy over ending reachability (uniform random walk)
    merge_ratio        – fraction of nodes that are merge nodes (>1 parent)

Text features (from story.txt):
    lines_per_character    – dict[char_name → line count]
    expression_diversity   – dict[char_name → unique expression count]

Quality scores (0 = worst, 1 = best):
    character_balance_score – 1 – Gini(line counts) across characters
    branch_balance_score    – normalised path entropy across endings
    scene_diversity_score   – 1 – Herfindahl index of scene usage

OOC detection:
    Samples up to SAMPLE_K dialogues per character, retrieves canon personality
    from RAG, then calls LLM to score consistency (0–10).
    Returns per-character OOC score (0 = fully OOC, 1 = perfectly in-character).

Integration helper:
    format_producer_feedback()  – formats low scores as structured text for Producer
    format_ooc_warning()        – formats OOC findings as long-term-memory warning
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ── Thresholds for "low score" warnings ───────────────────────────────────────
THRESHOLD_CHAR_BALANCE  = 0.60   # below → character monopoly
THRESHOLD_BRANCH_BALANCE = 0.55  # below → one dominant path
THRESHOLD_SCENE_DIV     = 0.55   # below → scene overuse
THRESHOLD_OOC           = 0.60   # below → OOC warning injected into context

SAMPLE_K = 5   # dialogues sampled per character for OOC detection


# ══════════════════════════════════════════════════════════════════════════════
# Graph helpers (same logic as game_validator, kept local to avoid circular dep)
# ══════════════════════════════════════════════════════════════════════════════

def _build_children(nodes: List[Dict], edges: List[Dict]) -> Dict[str, List[str]]:
    node_ids = {n["id"] for n in nodes}
    children: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        src = e.get("from", e.get("source", ""))
        dst = e.get("to",   e.get("target", ""))
        if src in node_ids and dst in node_ids:
            children[src].append(dst)
    return children


def _build_parents(nodes: List[Dict], edges: List[Dict]) -> Dict[str, List[str]]:
    node_ids = {n["id"] for n in nodes}
    parents: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        src = e.get("from", e.get("source", ""))
        dst = e.get("to",   e.get("target", ""))
        if src in node_ids and dst in node_ids:
            parents[dst].append(src)
    return parents


def _path_entropy(children: Dict[str, List[str]]) -> float:
    """
    Simulate a uniform random walk from all roots.
    Compute the probability of reaching each ending node.
    Return normalised Shannon entropy (0 = one dominant ending, 1 = perfectly uniform).
    """
    roots = [n for n, ch in children.items() if not any(n in ch2 for ch2 in children.values())]
    # Re-derive roots correctly: nodes with no parents
    all_nodes = set(children.keys())
    all_dsts: set = set()
    for ch_list in children.values():
        all_dsts.update(ch_list)
    roots = list(all_nodes - all_dsts)

    if not roots:
        return 0.0

    prob: Dict[str, float] = defaultdict(float)
    for r in roots:
        prob[r] += 1.0 / len(roots)

    # Topological sort (Kahn's algorithm)
    in_deg: Dict[str, int] = defaultdict(int)
    for node, ch_list in children.items():
        for child in ch_list:
            in_deg[child] += 1

    queue = deque([n for n in all_nodes if in_deg[n] == 0])
    topo: List[str] = []
    temp_in = dict(in_deg)
    while queue:
        node = queue.popleft()
        topo.append(node)
        for child in children.get(node, []):
            temp_in[child] -= 1
            if temp_in[child] == 0:
                queue.append(child)

    for node in topo:
        ch_list = children.get(node, [])
        if ch_list:
            share = prob[node] / len(ch_list)
            for child in ch_list:
                prob[child] += share

    endings = [n for n in all_nodes if not children.get(n)]
    if not endings:
        return 0.0

    total = sum(prob[e] for e in endings)
    if total == 0:
        return 0.0
    ps = [prob[e] / total for e in endings]
    H = -sum(p * math.log2(p) for p in ps if p > 0)
    max_H = math.log2(len(endings)) if len(endings) > 1 else 1.0
    return H / max_H if max_H > 0 else 1.0


def _gini(values: List[float]) -> float:
    """Gini coefficient (0 = perfectly equal, 1 = one entity has everything)."""
    if not values or sum(values) == 0:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    total = sum(arr)
    # Standard formula (1-indexed rank): G = (2 * Σ(rank * val)) / (n * total) - (n+1)/n
    weighted_sum = sum((i + 1) * v for i, v in enumerate(arr))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def _herfindahl(counts: Dict) -> float:
    """Herfindahl–Hirschman index (market concentration)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum((v / total) ** 2 for v in counts.values())


# ══════════════════════════════════════════════════════════════════════════════
# story.txt parsers
# ══════════════════════════════════════════════════════════════════════════════

_RE_CONTENT = re.compile(r'<content\s+id="([^"]+)">([^<]*)</content>', re.IGNORECASE)
_RE_IMAGE   = re.compile(r'<image\s+id="([^"]+)">([^<]*)</image>', re.IGNORECASE)
_RE_SCENE   = re.compile(r'<scene>([^<]+)</scene>', re.IGNORECASE)


def _parse_story_features(story_text: str, characters: List[Dict]) -> Dict:
    """
    Extract text-level features from story.txt.
    Returns:
        lines_per_character  : dict[char_name → int]
        tokens_per_character : dict[char_name → int]
        expression_per_char  : dict[char_name → set of expressions]
        scene_usage          : dict[scene_name → int]
        dialogues_per_char   : dict[char_name → list of dialogue strings]
    """
    # Build id→name mapping (case-insensitive)
    id_to_name: Dict[str, str] = {}
    for c in characters:
        cid  = c.get("id", "").upper()
        name = c.get("name", cid)
        id_to_name[cid]  = name
        id_to_name[name.upper()] = name

    lines_per_char:   Dict[str, int]       = defaultdict(int)
    tokens_per_char:  Dict[str, int]       = defaultdict(int)
    expr_per_char:    Dict[str, set]       = defaultdict(set)
    scene_usage:      Dict[str, int]       = defaultdict(int)
    dialogues:        Dict[str, List[str]] = defaultdict(list)

    for m in _RE_CONTENT.finditer(story_text):
        cid_raw = m.group(1).strip().upper()
        if cid_raw in ("NARRATOR", "NARRATION", "\u65c1\u767d"):
            continue
        name = id_to_name.get(cid_raw, m.group(1).strip())
        text = m.group(2).strip()
        lines_per_char[name]  += 1
        tokens_per_char[name] += len(text.split())
        dialogues[name].append(text)

    for m in _RE_IMAGE.finditer(story_text):
        cid_raw = m.group(1).strip().upper()
        expr    = m.group(2).strip().lower()
        name    = id_to_name.get(cid_raw, m.group(1).strip())
        if expr:
            expr_per_char[name].add(expr)

    for m in _RE_SCENE.finditer(story_text):
        scene_usage[m.group(1).strip()] += 1

    return {
        "lines_per_character":   dict(lines_per_char),
        "tokens_per_character":  dict(tokens_per_char),
        "expression_per_char":   {k: list(v) for k, v in expr_per_char.items()},
        "scene_usage":           dict(scene_usage),
        "dialogues_per_char":    dict(dialogues),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main scorer
# ══════════════════════════════════════════════════════════════════════════════

class QualityScorer:
    """
    Computes structural and textual quality scores for a generated visual novel.

    Usage (standalone):
        scorer = QualityScorer.from_files()
        result = scorer.score_all()
        print(scorer.format_producer_feedback(result))

    Usage (in workflow, with OOC):
        scorer = QualityScorer(game_design, story_text, rag_agent, llm_client)
        result = scorer.score_all(include_ooc=True)
    """

    def __init__(
        self,
        game_design: Dict,
        story_text: Optional[str] = None,
        rag_agent=None,
        llm_client=None,
    ):
        self.design     = game_design
        self.story_text = story_text
        self.rag_agent  = rag_agent
        self.llm_client = llm_client

        sg = game_design.get("story_graph", {})

        # nodes may be a list [{id, ...}] or a dict {id: {id, ...}}
        raw_nodes = sg.get("nodes", [])
        if isinstance(raw_nodes, dict):
            self._nodes = list(raw_nodes.values())
        else:
            self._nodes = list(raw_nodes)

        # edges may be a list or a dict keyed by edge id
        raw_edges = sg.get("edges", [])
        if isinstance(raw_edges, dict):
            self._edges = list(raw_edges.values())
        else:
            self._edges = list(raw_edges)

        self._chars     = game_design.get("characters", [])
        self._scenes    = game_design.get("scenes",    [])
        self._franchise = game_design.get("franchise", "")

    @classmethod
    def from_files(
        cls,
        design_path: Path = ROOT / "data" / "game_design.json",
        story_path:  Path = ROOT / "data" / "story.txt",
        rag_agent=None,
        llm_client=None,
    ) -> "QualityScorer":
        with open(design_path, encoding="utf-8") as f:
            design = json.load(f)
        story = story_path.read_text(encoding="utf-8") if story_path.exists() else None
        return cls(design, story, rag_agent, llm_client)

    # ── Structural features ───────────────────────────────────────────────────

    def _structural_features(self) -> Dict:
        children = _build_children(self._nodes, self._edges)
        parents  = _build_parents(self._nodes, self._edges)
        n        = len(self._nodes)

        non_leaf   = [nd for nd in children if children[nd]]
        avg_branch = (sum(len(children[nd]) for nd in non_leaf) / len(non_leaf)
                      if non_leaf else 0.0)

        merge_nodes = [nd for nd in parents if len(parents[nd]) > 1]
        merge_ratio = len(merge_nodes) / n if n else 0.0

        pe = _path_entropy(children)

        return {
            "node_count":       n,
            "edge_count":       len(self._edges),
            "branching_factor": round(avg_branch, 3),
            "path_entropy":     round(pe, 4),
            "merge_ratio":      round(merge_ratio, 4),
            "ending_count":     sum(1 for nd in children if not children[nd]),
        }

    # ── Text features ─────────────────────────────────────────────────────────

    def _text_features(self) -> Optional[Dict]:
        if not self.story_text:
            return None
        feats = _parse_story_features(self.story_text, self._chars)
        # expression diversity: avg unique expressions per character
        expr_counts = {c: len(v) for c, v in feats["expression_per_char"].items()}
        avg_expr_div = (sum(expr_counts.values()) / len(expr_counts)
                        if expr_counts else 0.0)
        return {
            "lines_per_character":     feats["lines_per_character"],
            "tokens_per_character":    feats["tokens_per_character"],
            "expression_diversity":    expr_counts,
            "avg_expression_diversity": round(avg_expr_div, 2),
            "scene_usage":             feats["scene_usage"],
            "_dialogues_per_char":     feats["dialogues_per_char"],  # internal, for OOC
        }

    # ── Quality scores ────────────────────────────────────────────────────────

    def _score_character_balance(self, text_feats: Dict) -> Tuple[float, str]:
        """1 – Gini(line counts). 1 = perfectly balanced, 0 = one character dominates."""
        counts = list(text_feats["lines_per_character"].values())
        if len(counts) < 2:
            return 1.0, "Only one character; balance N/A."
        g = _gini([float(c) for c in counts])
        score = round(1 - g, 4)
        total = sum(counts)
        top_char = max(text_feats["lines_per_character"], key=lambda k: text_feats["lines_per_character"][k])
        top_pct  = round(100 * text_feats["lines_per_character"][top_char] / total, 1)
        detail   = (f"Gini={g:.3f}; '{top_char}' has {top_pct}% of all lines. "
                    + ("⚠️ Character monopoly detected." if score < THRESHOLD_CHAR_BALANCE else "✓"))
        return score, detail

    def _score_branch_balance(self, struct_feats: Dict) -> Tuple[float, str]:
        """Normalised path entropy. 1 = all endings equally reachable, 0 = one dominant path."""
        score  = round(struct_feats["path_entropy"], 4)
        detail = (f"Path entropy={score:.4f} over {struct_feats['ending_count']} endings. "
                  + ("⚠️ One path dominates." if score < THRESHOLD_BRANCH_BALANCE else "✓"))
        return score, detail

    def _score_scene_diversity(self, text_feats: Dict) -> Tuple[float, str]:
        """1 – HHI(scene usage). 1 = scenes used uniformly, 0 = one scene dominates."""
        usage = text_feats["scene_usage"]
        if not usage:
            return 1.0, "No scene data in story.txt."
        hhi   = _herfindahl(usage)
        score = round(1 - hhi, 4)
        top_scene = max(usage, key=usage.get)
        total     = sum(usage.values())
        top_pct   = round(100 * usage[top_scene] / total, 1)
        detail    = (f"HHI={hhi:.3f}; '{top_scene}' used {top_pct}% of scene switches. "
                     + ("⚠️ Scene overuse." if score < THRESHOLD_SCENE_DIV else "✓"))
        return score, detail

    # ── OOC detection ─────────────────────────────────────────────────────────

    def _detect_ooc(self, dialogues_per_char: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        For each character, sample up to SAMPLE_K dialogues,
        retrieve canon personality via RAG, call LLM for OOC score.
        Returns dict[char_name → {score, reason, samples_used}].
        """
        if self.rag_agent is None or self.llm_client is None:
            logger.warning("OOC detection skipped: rag_agent or llm_client not provided.")
            return {}

        results: Dict[str, Dict] = {}

        for char in self._chars:
            name    = char.get("name", "")
            char_id = char.get("id", name)
            if not name:
                continue

            # Collect sample dialogues (try both name and ID keys)
            raw_dialogues = (dialogues_per_char.get(name)
                             or dialogues_per_char.get(char_id)
                             or [])
            if not raw_dialogues:
                logger.debug(f"OOC: no dialogues found for '{name}', skipping.")
                continue

            sample = random.sample(raw_dialogues, min(SAMPLE_K, len(raw_dialogues)))

            # RAG: retrieve canon personality
            try:
                canon_ctx = self.rag_agent.retrieve_character_context(name, n_results=3)
            except Exception as e:
                logger.warning(f"OOC: RAG retrieval failed for '{name}': {e}")
                canon_ctx = char.get("personality", "No canon info available.")

            if not canon_ctx:
                canon_ctx = char.get("personality", "No canon info available.")

            # Build prompt
            dialogue_block = "\n".join(f'  [{i+1}] "{d}"' for i, d in enumerate(sample))
            prompt = _OOC_USER_PROMPT.format(
                name=name,
                franchise=self._franchise or self.design.get("title", "this story"),
                canon_info=canon_ctx[:1500],
                dialogue_sample=dialogue_block,
            )

            try:
                response = self.llm_client.chat_completion(
                    messages=[
                        {"role": "system", "content": _OOC_SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.2,
                )
                score_raw, reason = _parse_ooc_response(response)
                results[name] = {
                    "score":        round(score_raw / 10, 3),   # normalise to 0–1
                    "raw_score":    score_raw,
                    "reason":       reason,
                    "samples_used": len(sample),
                }
                logger.info(f"OOC [{name}]: {score_raw}/10 — {reason[:80]}")
            except Exception as e:
                logger.warning(f"OOC: LLM call failed for '{name}': {e}")

        return results

    # ── Public API ────────────────────────────────────────────────────────────

    def score_all(self, include_ooc: bool = False) -> Dict:
        """
        Compute all features and scores.

        Returns a result dict with keys:
            structural_features, text_features,
            character_balance_score, branch_balance_score, scene_diversity_score,
            ooc_results (if include_ooc=True),
            overall_structural_score
        """
        result: Dict = {}

        # ── Structural ────────────────────────────────────────────────────────
        sf = self._structural_features()
        result["structural_features"] = sf

        bs, bd = self._score_branch_balance(sf)
        result["branch_balance_score"]  = {"score": bs, "detail": bd}

        # ── Text ──────────────────────────────────────────────────────────────
        tf = self._text_features()
        result["text_features"] = tf

        if tf:
            cb, cd = self._score_character_balance(tf)
            sd, sdd = self._score_scene_diversity(tf)
            result["character_balance_score"] = {"score": cb, "detail": cd}
            result["scene_diversity_score"]   = {"score": sd, "detail": sdd}
        else:
            result["character_balance_score"] = None
            result["scene_diversity_score"]   = None

        # ── OOC ───────────────────────────────────────────────────────────────
        if include_ooc and tf:
            result["ooc_results"] = self._detect_ooc(tf["_dialogues_per_char"])
        else:
            result["ooc_results"] = {}

        # ── Overall structural score (simple average of available scores) ─────
        scores = [bs]
        if tf:
            scores.append(result["character_balance_score"]["score"])
            scores.append(result["scene_diversity_score"]["score"])
        result["overall_structural_score"] = round(sum(scores) / len(scores), 4)

        return result

    # ── Integration helpers ───────────────────────────────────────────────────

    def format_producer_feedback(self, result: Dict) -> str:
        """
        Returns a structured text block for injection into the Producer critique loop.
        Only covers GRAPH-STRUCTURAL issues that the Designer can actually fix:
          - branch_balance (path entropy) — fix by rebalancing edges
        Story-level issues (scene diversity, character balance) are handled separately
        via format_writer_guidance() and injected into the Writer's context instead.
        Returns empty string if all structural scores are acceptable.
        """
        lines: List[str] = []

        bs = result["branch_balance_score"]
        if bs["score"] < THRESHOLD_BRANCH_BALANCE:
            lines.append(
                f"- Branch balance score: {bs['score']:.2f} "
                f"(threshold {THRESHOLD_BRANCH_BALANCE}) — {bs['detail']} "
                f"Consider adding more edges to underrepresented endings."
            )

        if not lines:
            return ""

        return (
            "\n[Automated Quality Analysis — Action Required]\n"
            "The following graph-structure issues were detected:\n"
            + "\n".join(lines)
            + "\nPlease revise the story graph structure to address these issues.\n"
        )

    def format_writer_guidance(self, result: Dict) -> str:
        """
        Returns a guidance block for injection into the Writer's context
        (long_term_memory / system prompt addendum) to fix story-level issues
        that the Producer cannot control: scene diversity and character balance.
        Returns empty string if all scores are acceptable.
        """
        lines: List[str] = []

        tf = result.get("text_features")
        if tf:
            sd = result.get("scene_diversity_score")
            if sd and sd["score"] < THRESHOLD_SCENE_DIV:
                usage = tf.get("scene_usage", {})
                defined_scenes = [s.get("name", "") for s in self._scenes]
                overused = max(usage, key=usage.get) if usage else "unknown"
                underused = [s for s in defined_scenes if s not in usage or usage[s] == 0]
                lines.append(
                    f"- Scene diversity is LOW (score {sd['score']:.2f}): "
                    f"'{overused}' is overused. "
                    + (f"Try using underused scenes: {', '.join(underused[:3])}."
                       if underused else "Vary scene usage across nodes.")
                )

            cb = result.get("character_balance_score")
            if cb and cb["score"] < THRESHOLD_CHAR_BALANCE:
                lpc = tf.get("lines_per_character", {})
                dominant = max(lpc, key=lpc.get) if lpc else "unknown"
                lines.append(
                    f"- Character balance is LOW (score {cb['score']:.2f}): "
                    f"'{dominant}' dominates dialogue. Give other characters more lines."
                )

        if not lines:
            return ""

        return (
            "\n[Story Balance Advisory — from quality scorer]\n"
            + "\n".join(lines)
            + "\nPlease vary scenes and distribute dialogue more evenly across characters.\n"
        )

    @staticmethod
    def format_ooc_warning(ooc_results: Dict) -> str:
        """
        Returns a warning block for injection into long_term_memory
        when OOC scores are below threshold.
        Returns empty string if no OOC issues detected.
        """
        warnings: List[str] = []
        for name, data in ooc_results.items():
            if data["score"] < THRESHOLD_OOC:
                warnings.append(
                    f"  - {name} (OOC score {data['score']:.2f}/1.0): {data['reason']}"
                )
        if not warnings:
            return ""
        return (
            "\n[OOC Warning — from previous node]\n"
            "The following characters showed dialogue inconsistent with their canon personality:\n"
            + "\n".join(warnings)
            + "\nEnsure subsequent dialogue strictly follows each character's established traits.\n"
        )

    def print_report(self, result: Dict):
        """Pretty-print the quality report to console."""
        SEP  = "─" * 60
        BOLD = "\033[1m"
        RST  = "\033[0m"
        GRN  = "\033[32m"
        YEL  = "\033[33m"
        RED  = "\033[31m"

        def _color_score(s: float) -> str:
            color = GRN if s >= 0.7 else (YEL if s >= 0.5 else RED)
            return f"{color}{s:.4f}{RST}"

        print(f"\n{'═' * 60}")
        print(f"  Story Quality Report")
        print(f"{'═' * 60}")

        # Structural
        print(f"\n{BOLD}Structural Features{RST}")
        print(SEP)
        sf = result["structural_features"]
        for k, v in sf.items():
            print(f"  {k:<24}: {v}")

        # Scores
        print(f"\n{BOLD}Quality Scores{RST}")
        print(SEP)
        for key in ["branch_balance_score", "character_balance_score", "scene_diversity_score"]:
            val = result.get(key)
            if val is None:
                print(f"  {key:<30}: N/A (story.txt not available)")
            else:
                print(f"  {key:<30}: {_color_score(val['score'])}")
                print(f"  {'':30}  {val['detail']}")

        print(f"\n  {'overall_structural_score':<30}: {_color_score(result['overall_structural_score'])}")

        # OOC
        ooc = result.get("ooc_results", {})
        if ooc:
            print(f"\n{BOLD}OOC Detection{RST}")
            print(SEP)
            for name, data in ooc.items():
                print(f"  {name:<20}: {_color_score(data['score'])} ({data['samples_used']} samples)")
                print(f"  {'':20}  {data['reason']}")

        print(f"\n{'═' * 60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# OOC prompt templates
# ══════════════════════════════════════════════════════════════════════════════

_OOC_SYSTEM_PROMPT = (
    "You are an expert on anime/manga characters and their canonical personalities. "
    "Your task is to evaluate whether sampled dialogue lines are consistent with a "
    "character's known personality from source material."
)

_OOC_USER_PROMPT = """\
Character: {name}
Franchise: {franchise}

[Canon Personality (from source material)]
{canon_info}

[Dialogue Sample from Generated Story]
{dialogue_sample}

Rate how consistent these dialogue lines are with the character's canonical personality.
Be strict but fair — small stylistic differences are acceptable; OOC means behaviour or \
attitude that clearly contradicts canon.

Output EXACTLY in this format (no extra text):
Score: <integer 0-10>
Reason: <one sentence explaining the main consistency issue or confirmation>"""


def _parse_ooc_response(response: str) -> Tuple[float, str]:
    """Parse LLM OOC response. Returns (score 0-10, reason)."""
    score_m  = re.search(r"Score:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
    reason_m = re.search(r"Reason:\s*(.+)",           response, re.IGNORECASE | re.DOTALL)
    score  = float(score_m.group(1))  if score_m  else 5.0
    reason = reason_m.group(1).strip()[:200] if reason_m else response.strip()[:200]
    score  = max(0.0, min(10.0, score))
    return score, reason


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Story Quality Scorer")
    parser.add_argument("--design", default=str(ROOT / "data" / "game_design.json"))
    parser.add_argument("--story",  default=str(ROOT / "data" / "story.txt"))
    parser.add_argument("--ooc",    action="store_true", help="Run OOC detection (requires API keys)")
    parser.add_argument("--out",    default="", help="Save result JSON to this path")
    args = parser.parse_args()

    rag_agent  = None
    llm_client = None

    if args.ooc:
        try:
            from agents.rag_agent  import RAGAgent
            from agents.llm_client import LLMClient
            from agents.config     import APIConfig
            design_tmp = json.loads(Path(args.design).read_text(encoding="utf-8"))
            franchise  = design_tmp.get("franchise", "")
            rag_agent  = RAGAgent(franchise=franchise) if franchise else None
            llm_client = LLMClient()
            logger.info("OOC mode: RAG + LLM initialised.")
        except Exception as e:
            print(f"⚠️  Could not init RAG/LLM for OOC: {e}")

    scorer = QualityScorer.from_files(
        design_path=Path(args.design),
        story_path =Path(args.story),
        rag_agent  =rag_agent,
        llm_client =llm_client,
    )
    result = scorer.score_all(include_ooc=args.ooc)
    scorer.print_report(result)

    if args.out:
        out = {k: v for k, v in result.items() if k != "text_features" or True}
        # Remove internal key before saving
        if result.get("text_features"):
            result["text_features"].pop("_dialogues_per_char", None)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Result saved → {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
