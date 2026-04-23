"""
Game Validator
~~~~~~~~~~~~~~
Rule-based validation for AI-generated game_design.json and story.txt.

Checks:
    1-A  Node count in reasonable range
    1-B  Dangling edges (reference undefined nodes)
    1-C  Duplicate edges
    2-A  Character IDs in story.txt match game_design.json
    2-B  Scene names in story.txt match game_design.json
    2-C  Expression name quality (non-empty, no spaces)

Output:
    ValidationReport – structured dict of ERROR / WARNING / INFO items,
    with console-printable summary.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Thresholds ────────────────────────────────────────────────────────────────
NODE_MIN = 5
NODE_MAX = 30

_RE_CONTENT = re.compile(r'<content\s+id="([^"]+)">', re.IGNORECASE)
_RE_IMAGE   = re.compile(r'<image\s+id="([^"]+)">([^<]*)</image>', re.IGNORECASE)
_RE_SCENE   = re.compile(r'<scene>([^<]+)</scene>', re.IGNORECASE)


# ── Report data structures ────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    level:   str   # "ERROR" | "WARNING" | "INFO"
    code:    str   # short machine-readable code
    message: str
    repaired: bool = False


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    # convenience properties
    @property
    def errors(self)   -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "ERROR"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "WARNING"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def add(self, level: str, code: str, message: str, repaired: bool = False):
        self.issues.append(ValidationIssue(level, code, message, repaired))

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "error_count":   len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {"level": i.level, "code": i.code,
                 "message": i.message, "repaired": i.repaired}
                for i in self.issues
            ],
        }

    def print_console(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"\n[GameValidator] {status}  "
              f"({len(self.errors)} errors, {len(self.warnings)} warnings)")
        for issue in self.issues:
            tag = {"ERROR": "❌", "WARNING": "⚠️ ", "INFO": "ℹ️ "}.get(issue.level, "  ")
            rep = " [repaired]" if issue.repaired else ""
            print(f"  {tag} [{issue.code}] {issue.message}{rep}")


# ── Validator ─────────────────────────────────────────────────────────────────

class GameValidator:
    """
    Validates game_design.json and story.txt.

    Usage:
        validator = GameValidator()
        report    = validator.validate()
        report.print_console()
    """

    def __init__(
        self,
        design_path: Path = ROOT / "data" / "game_design.json",
        story_path:  Path = ROOT / "data" / "story.txt",
        repaired_design_path: Optional[Path] = ROOT / "data" / "game_design_repaired.json",
    ):
        self.design_path         = Path(design_path)
        self.story_path          = Path(story_path)
        self.repaired_design_path = (Path(repaired_design_path)
                                     if repaired_design_path else None)
        self._design: Optional[Dict] = None
        self._story:  Optional[str]  = None

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load(self) -> ValidationReport:
        report = ValidationReport()
        if not self.design_path.exists():
            report.add("ERROR", "FILE_MISSING",
                       f"game_design.json not found: {self.design_path}")
            return report
        try:
            with open(self.design_path, encoding="utf-8") as f:
                self._design = json.load(f)
        except json.JSONDecodeError as e:
            report.add("ERROR", "JSON_INVALID", f"game_design.json parse error: {e}")
            return report

        if self.story_path.exists():
            self._story = self.story_path.read_text(encoding="utf-8")
        else:
            report.add("INFO", "STORY_MISSING",
                       "story.txt not found – skipping story consistency checks.")
        return report

    # ── Graph checks ─────────────────────────────────────────────────────────

    def _check_graph(self, report: ValidationReport):
        sg    = self._design.get("story_graph", {})
        raw_nodes = sg.get("nodes", [])
        edges = sg.get("edges", [])

        # Normalise nodes: dict {id: {...}} → list [{id, ...}]
        if isinstance(raw_nodes, dict):
            nodes = list(raw_nodes.values())
        else:
            nodes = list(raw_nodes)

        # Normalise edges
        if isinstance(edges, dict):
            edges = list(edges.values())

        node_ids = {n.get("id", "") for n in nodes}

        # 1-A: node count
        n = len(nodes)
        if n < NODE_MIN:
            report.add("ERROR", "NODE_COUNT_LOW",
                       f"Only {n} nodes (minimum {NODE_MIN}). Story may be too short.")
        elif n > NODE_MAX:
            report.add("WARNING", "NODE_COUNT_HIGH",
                       f"{n} nodes (maximum recommended {NODE_MAX}). "
                       "Story may be too complex for one session.")
        else:
            report.add("INFO", "NODE_COUNT_OK", f"Node count = {n} ✓")

        # 1-B: dangling edges
        dangling = []
        for e in edges:
            src = e.get("from", e.get("source", ""))
            dst = e.get("to",   e.get("target", ""))
            if src not in node_ids:
                dangling.append(f"edge source '{src}' undefined")
            if dst not in node_ids:
                dangling.append(f"edge target '{dst}' undefined")
        if dangling:
            for msg in dangling[:5]:   # cap at 5 to avoid flooding
                report.add("ERROR", "DANGLING_EDGE", msg)
        else:
            report.add("INFO", "EDGES_OK", f"All {len(edges)} edges reference valid nodes ✓")

        # 1-C: duplicate edges
        seen_pairs: set = set()
        dups = []
        for e in edges:
            src = e.get("from", e.get("source", ""))
            dst = e.get("to",   e.get("target", ""))
            pair = (src, dst)
            if pair in seen_pairs:
                dups.append(f"{src} → {dst}")
            seen_pairs.add(pair)
        if dups:
            for dup in dups[:5]:
                report.add("WARNING", "DUPLICATE_EDGE", f"Duplicate edge: {dup}")

    # ── Story consistency checks ──────────────────────────────────────────────

    def _check_story_consistency(self, report: ValidationReport):
        if not self._story:
            return

        design_char_ids = set()
        for c in self._design.get("characters", []):
            cid  = c.get("id",   "").strip().upper()
            name = c.get("name", "").strip().upper()
            if cid:  design_char_ids.add(cid)
            if name: design_char_ids.add(name)
        design_scene_names = {
            s.get("name", "").strip()
            for s in self._design.get("scenes", [])
        }

        # 2-A: character ID consistency
        story_char_ids = {m.group(1).strip().upper()
                          for m in _RE_CONTENT.finditer(self._story)
                          if m.group(1).strip().upper() not in ("NARRATOR", "NARRATION", "\u65c1\u767d")}
        unknown_chars = story_char_ids - design_char_ids
        if unknown_chars:
            for cid in sorted(unknown_chars)[:5]:
                report.add("WARNING", "UNKNOWN_CHAR",
                           f"story.txt references character ID '{cid}' "
                           "not defined in game_design.json")
        else:
            report.add("INFO", "CHAR_IDS_OK",
                       f"All {len(story_char_ids)} character IDs match design ✓")

        # 2-B: scene name consistency
        story_scenes = {m.group(1).strip() for m in _RE_SCENE.finditer(self._story)}
        unknown_scenes = story_scenes - design_scene_names
        if unknown_scenes:
            for sc in sorted(unknown_scenes)[:5]:
                report.add("WARNING", "UNKNOWN_SCENE",
                           f"story.txt uses scene '{sc}' not defined in game_design.json")
        else:
            report.add("INFO", "SCENE_NAMES_OK",
                       f"All {len(story_scenes)} scene names match design ✓")

        # 2-C: expression name quality
        bad_exprs = []
        for m in _RE_IMAGE.finditer(self._story):
            expr = m.group(2).strip()
            if not expr:
                bad_exprs.append(f"character '{m.group(1)}' has empty expression")
            elif " " in expr:
                bad_exprs.append(f"expression '{expr}' contains spaces (may cause asset errors)")
        if bad_exprs:
            for msg in bad_exprs[:5]:
                report.add("WARNING", "BAD_EXPRESSION", msg)
        else:
            report.add("INFO", "EXPRESSIONS_OK", "All expression names look valid ✓")

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self) -> ValidationReport:
        """Run all checks and return a ValidationReport."""
        report = self._load()
        if not self._design:
            return report

        self._check_graph(report)
        self._check_story_consistency(report)
        return report
