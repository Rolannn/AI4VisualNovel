"""
Knowledge Builder
~~~~~~~~~~~~~~~~~~~~~~
Fetches IP / character knowledge from Wikipedia and local documents,
builds a local knowledge base with clean chunking, metadata, and retrieval.

Dependencies:
    pip install wikipedia-api numpy
"""

import os
import json
import math
import logging
import re
from collections import Counter
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Section filters
# ─────────────────────────────────────────────

NOISE_SECTIONS = {
    "References", "External links", "See also",
    "Notes", "Bibliography", "Gallery", "Further reading",
    "Footnotes", "Citations",
    "Reception", "Critical reception", "Awards", "Accolades",
    "Voice actor", "Voice actors", "Voice cast", "Cast",
    "Poll ranking", "Popularity polls", "Merchandise",
    "Music", "Soundtrack", "Discography",
    "Production", "Development", "Staff",
    "Legacy", "Impact", "Cultural impact",
    # Chinese equivalents
    "\u53c2\u8003\u8d44\u6599", "\u5916\u90e8\u94fe\u63a5", "\u53c2\u89c1",
    "\u6ce8\u91ca", "\u83b7\u5956", "\u58f0\u4f18",
}
# precompute lowercase set for case-insensitive matching
_NOISE_LOWER = {s.lower() for s in NOISE_SECTIONS}

PRIORITY_SECTIONS = {
    "personality", "character", "abilities", "powers", "skills",
    "background", "history", "relationships", "appearance",
    "role", "plot", "story", "overview", "description",
}

TRAIT_KEYWORDS = [
    "calm", "mysterious", "manipulative", "cheerful", "aggressive",
    "loyal", "cold", "warm", "brave", "cowardly", "intelligent",
    "naive", "serious", "playful", "dominant", "submissive",
    "caring", "selfish", "honest", "deceptive", "determined",
    "reckless", "strategic", "impulsive", "stoic", "emotional",
    "confident", "insecure", "protective", "ruthless", "gentle",
    "charismatic", "quiet", "energetic", "laid-back", "optimistic",
    "pessimistic", "sadistic", "empathetic", "arrogant", "humble",
    # expanded forms that Wikipedia actually uses
    "manipulator", "cunning", "intimidating", "caring", "selfless",
    "fearless", "cold-blooded", "hot-headed", "level-headed",
]


# ─────────────────────────────────────────────
# 1. Wikipedia Fetcher
# ─────────────────────────────────────────────

class WikipediaFetcher:
    """Fetch IP / character knowledge pages from Wikipedia."""

    def __init__(self, language: str = "en"):
        self.language = language
        self.available = False
        try:
            import wikipediaapi
            self.wiki = wikipediaapi.Wikipedia(
                language=language,
                user_agent="AI4VisualNovel/1.0 (educational-project)"
            )
            self.available = True
            logger.info(f"wikipedia-api ready (language: {language})")
        except ImportError:
            logger.warning(
                "wikipedia-api not installed. Run: pip install wikipedia-api"
            )

    def fetch_page(self, title: str) -> Optional[Dict]:
        """Fetch a Wikipedia page, filtering noise sections (case-insensitive)."""
        if not self.available:
            return None

        logger.info(f"   Fetching Wikipedia: [{title}]")
        try:
            page = self.wiki.page(title)
        except Exception as e:
            logger.warning(f"   Wikipedia request failed: {e}")
            return None

        if not page.exists():
            logger.warning(f"   Page not found: {title}")
            return None

        # Fix 6: case-insensitive noise filtering
        sections = {}
        for section in page.sections:
            if section.title.lower() in _NOISE_LOWER:
                continue
            if section.text.strip():
                sections[section.title] = section.text.strip()

        return {
            "title": page.title,
            "summary": page.summary[:3000] if page.summary else "",
            "sections": sections,
            "url": page.fullurl,
        }

    def fetch_character(self, character_name: str, franchise: str = "") -> Optional[Dict]:
        """
        Try multiple strategies to find the correct character page.
        Priority:
          1. "<character> (<franchise>)"  – most precise
          2. "<character> (character)"
          3. "<character>"               – fallback, validated via summary
        """
        candidates = []
        if franchise:
            candidates.append(f"{character_name} ({franchise})")
        candidates.append(f"{character_name} (character)")
        candidates.append(character_name)

        franchise_lower = franchise.lower() if franchise else ""

        for title in candidates:
            result = self.fetch_page(title)
            if not result:
                continue

            if franchise and franchise.lower() in title.lower():
                result["type"] = "character"
                result["character_name"] = character_name
                logger.info(f"   Found character page (exact): {result['title']}")
                return result

            if franchise_lower and franchise_lower not in result.get("summary", "").lower():
                logger.warning(
                    f"   Skipping disambiguation [{title}] "
                    f"(summary missing '{franchise}')"
                )
                continue

            result["type"] = "character"
            result["character_name"] = character_name
            logger.info(f"   Found character page (summary-validated): {result['title']}")
            return result

        logger.warning(f"   Character page not found: {character_name}")
        return None

    def fetch_franchise(self, franchise_name: str) -> Optional[Dict]:
        result = self.fetch_page(franchise_name)
        if result:
            result["type"] = "franchise"
        return result


# ─────────────────────────────────────────────
# 2. SimpleVectorStore  (BM25 + global-IDF cosine rerank)
# ─────────────────────────────────────────────

class SimpleVectorStore:
    """
    Lightweight document store + two-stage retrieval.

    Stage 1 – BM25 recall      : fast keyword matching, top-N candidates.
                                  Uses precomputed global DF (O(1) lookup).
    Stage 2 – TF-IDF cosine    : re-scores candidates using GLOBAL IDF
                                  (from full corpus, not just candidates).

    Documents are persisted as a JSON file.
    """

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.documents: List[Dict] = []
        # Fix 3: global DF cache + real avg doc length
        self._df: Counter = Counter()       # token → document frequency
        self._avg_doc_len: float = 150.0    # updated on each add
        self._load()
        self._rebuild_index()  # recompute cache after load

    # ── Persistence ─────────────────────────────────────

    def _load(self):
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
                logger.info(
                    f"   Loaded knowledge base: {self.store_path} "
                    f"({len(self.documents)} documents)"
                )
            except Exception as e:
                logger.warning(f"   Failed to load knowledge base: {e}; starting empty.")
                self.documents = []

    def _save(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump({"documents": self.documents}, f, ensure_ascii=False, indent=2)

    def _rebuild_index(self):
        """Recompute global DF cache and avg_doc_len from all documents."""
        self._df = Counter()
        total_len = 0
        for doc in self.documents:
            toks = self._tokenize(doc["text"])
            self._df.update(set(toks))     # DF = docs containing token
            total_len += len(toks)
        self._avg_doc_len = total_len / max(len(self.documents), 1)

    def clear(self):
        self.documents = []
        self._df = Counter()
        self._avg_doc_len = 150.0
        self._save()
        logger.info("Knowledge base cleared.")

    # ── Write ────────────────────────────────────────────

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        if not texts:
            return
        metadatas = metadatas or [{}] * len(texts)
        added = 0
        for text, meta in zip(texts, metadatas):
            if len(text.strip().split()) < 5:
                continue
            doc_id = f"doc_{len(self.documents)}"
            self.documents.append({"id": doc_id, "text": text, "metadata": meta})
            # Fix 3: incrementally update DF cache
            toks = self._tokenize(text)
            self._df.update(set(toks))
            added += 1

        if added:
            # Update avg_doc_len
            total_len = sum(len(self._tokenize(d["text"])) for d in self.documents)
            self._avg_doc_len = total_len / max(len(self.documents), 1)
            self._save()

    # ── Tokenizer ─────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # include digits so "Division 4", "Unit 1" are preserved
        return re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())

    # ── Stage 1: BM25 scoring ────────────────────────────

    def _bm25_score(self, query_tokens: List[str], doc_text: str) -> float:
        """BM25 using precomputed global DF (O(1) per token, not O(N))."""
        k1, b = 1.5, 0.75
        avg_len = self._avg_doc_len          # real avg, not hardcoded

        doc_tokens = self._tokenize(doc_text)
        doc_len = max(len(doc_tokens), 1)
        doc_freq = Counter(doc_tokens)
        N = max(len(self.documents), 1)

        score = 0.0
        for token in set(query_tokens):
            tf = doc_freq.get(token, 0)
            if tf == 0:
                continue
            df = self._df.get(token, 0)      # O(1) lookup
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
            score += idf * tf_norm
        return score

    # ── Stage 2: global-IDF cosine rerank ────────────────

    def _tfidf_cosine_rerank(
        self, query: str, candidates: List[Dict], n_results: int
    ) -> List[Dict]:
        """
        Re-rank BM25 candidates using TF-IDF cosine similarity.
        Fix 4: IDF is computed from the FULL corpus (self._df),
               not from the local candidate set.
        """
        try:
            N_global = max(len(self.documents), 1)
            all_texts = [query] + [d["text"] for d in candidates]
            tokenized = [self._tokenize(t) for t in all_texts]

            # Build vocabulary from query + candidates
            vocab = list({tok for toks in tokenized for tok in toks})
            vocab_idx = {tok: i for i, tok in enumerate(vocab)}
            V = len(vocab)
            n = len(all_texts)   # query + candidates

            # Fix 4: IDF from global corpus DF, not local candidate DF
            idf = np.zeros(V, dtype=np.float64)
            for tok, idx in vocab_idx.items():
                df = self._df.get(tok, 0)
                idf[idx] = math.log((N_global + 1) / (df + 1)) + 1.0

            # TF-IDF matrix (float64)
            mat = np.zeros((n, V), dtype=np.float64)
            for i, toks in enumerate(tokenized):
                total = max(len(toks), 1)
                for tok, cnt in Counter(toks).items():
                    if tok in vocab_idx:
                        mat[i, vocab_idx[tok]] = (cnt / total) * idf[vocab_idx[tok]]

            # Cosine similarity: query vs. each candidate
            q_vec = mat[0]
            d_mat = mat[1:]
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return candidates[:n_results]

            d_norms = np.linalg.norm(d_mat, axis=1)
            d_norms = np.where(d_norms == 0, 1.0, d_norms)
            sims = d_mat.dot(q_vec) / (q_norm * d_norms)

            reranked = sorted(zip(sims, candidates), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in reranked[:n_results]]

        except Exception as e:
            logger.debug(f"Rerank failed, using BM25 order: {e}")
            return candidates[:n_results]

    # ── Public search ─────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        bm25_candidates: int = 30,
        franchise_filter: str = "",
        entity_filter: str = "",
    ) -> List[Dict]:
        """
        Two-stage retrieval:
          1. Metadata pre-filter (franchise / entity)
          2. BM25 recall (top bm25_candidates)
          3. TF-IDF cosine rerank with global IDF (top n_results)
        """
        if not self.documents:
            return []

        pool = self.documents
        if franchise_filter:
            ff = franchise_filter.lower()
            filtered = [
                d for d in pool
                if ff in d.get("metadata", {}).get("franchise", "").lower()
            ]
            if filtered:
                pool = filtered

        if entity_filter:
            ef = entity_filter.lower()
            filtered = [
                d for d in pool
                if ef in d.get("metadata", {}).get("entity", "").lower()
                or ef in d.get("metadata", {}).get("character", "").lower()
            ]
            if filtered:
                pool = filtered

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return pool[:n_results]

        # Stage 1: BM25
        scored = [
            (self._bm25_score(query_tokens, doc["text"]), doc)
            for doc in pool
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [doc for score, doc in scored[:bm25_candidates] if score > 0]

        if not candidates:
            return []
        if len(candidates) <= n_results:
            return candidates

        # Stage 2: global-IDF cosine rerank
        return self._tfidf_cosine_rerank(query, candidates, n_results)


# ─────────────────────────────────────────────
# 3. Knowledge Builder
# ─────────────────────────────────────────────

class KnowledgeBuilder:
    """
    Integrates WikipediaFetcher + SimpleVectorStore.
    Handles sentence-aware chunking, metadata tagging, noise filtering,
    entity-aware list-page processing, and character profile generation.
    """

    CHUNK_SIZE = 150    # target words per chunk
    CHUNK_OVERLAP = 20  # overlap words between adjacent chunks

    def __init__(self, store_path: str, language: str = "en"):
        self.fetcher = WikipediaFetcher(language=language)
        self.store = SimpleVectorStore(store_path)
        self.store_path = store_path
        self.language = language

    # ── Fix 1: word-boundary safe truncation ─────────────

    @staticmethod
    def _safe_truncate(text: str, max_chars: int) -> str:
        """Truncate to max_chars at the nearest word boundary (no mid-word cuts)."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        boundary = truncated.rfind(" ")
        if boundary > max_chars // 2:          # only step back if there's a real word
            truncated = truncated[:boundary]
        return truncated + "..."

    # ── Sentence-aware chunking ───────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into ~CHUNK_SIZE word chunks at sentence boundaries.
        Hard-splits sentences that exceed CHUNK_SIZE by themselves.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks: List[str] = []
        current: List[str] = []

        for sent in sentences:
            words = sent.split()
            if not words:
                continue

            if len(words) > self.CHUNK_SIZE:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                for i in range(0, len(words), self.CHUNK_SIZE):
                    part = words[i: i + self.CHUNK_SIZE]
                    if len(part) >= 5:
                        chunks.append(" ".join(part))
                continue

            if len(current) + len(words) > self.CHUNK_SIZE:
                if len(current) >= 10:
                    chunks.append(" ".join(current))
                    current = current[-self.CHUNK_OVERLAP:]
            current.extend(words)

        if len(current) >= 5:
            chunks.append(" ".join(current))

        return chunks

    # ── Trait extraction ──────────────────────────────────

    def _extract_traits(self, summary: str, sections: Dict) -> List[str]:
        text = summary
        for key in ("Personality", "Character", "Description", "Overview",
                    "Appearances", "Characterization", "Abilities"):
            if key in sections:
                text += " " + sections[key]
        text_lower = text.lower()
        found = [t for t in TRAIT_KEYWORDS if t in text_lower]
        return list(dict.fromkeys(found))[:8]   # deduplicate, cap at 8

    # ── Character Profile chunk ─────────────────

    def _make_character_profile(
        self,
        character_name: str,
        franchise: str,
        page_data: Dict,
        char_section_text: str = "",   # content from list-page section
    ) -> Optional[Dict]:
        """
        Build a structured Character Profile chunk.
        Fix 1: uses _safe_truncate instead of [:N] hard slices.
        Fix 2: accepts char_section_text for list-page character sections.
        """
        summary = page_data.get("summary", "")
        sections = page_data.get("sections", {})

        # If we have a dedicated section from a list page, use it as synthetic summary
        if char_section_text and len(char_section_text) > len(summary):
            summary = char_section_text

        if not summary and not sections:
            return None

        traits = self._extract_traits(summary, sections)

        priority_order = [
            "Personality", "Character", "Overview", "Description",
            "Appearance", "Abilities", "Background", "History", "Relationships",
        ]
        profile_parts: List[str] = []
        for key in priority_order:
            if key in sections:
                snippet = self._safe_truncate(sections[key], 500).strip()
                if snippet:
                    profile_parts.append(f"{key}: {snippet}")
            if len(profile_parts) >= 3:
                break

        lines = [
            f"[Character Profile: {character_name} | {franchise}]",
            f"Entity: {character_name}",
            f"Franchise: {franchise}",
        ]
        if traits:
            lines.append(f"Key Traits: {', '.join(traits)}")
        if summary:
            # Fix 1: safe truncate
            lines.append(f"Description: {self._safe_truncate(summary, 600)}")
        lines.extend(profile_parts)

        return {
            "text": "\n".join(lines),
            "metadata": {
                "source": "wikipedia",
                "title": page_data.get("title", character_name),
                "franchise": franchise,
                "entity": character_name,
                "type": "character_profile",
                "section": "profile",
                "traits": traits,
            },
        }

    # ── Utility ───────────────────────────────────────────

    def clear(self):
        self.store.clear()

    def get_document_count(self) -> int:
        return len(self.store.documents)

    # ── Add Wikipedia page ────────────────────────────────

    def add_wikipedia_page(
        self,
        page_data: Dict,
        source_type: str = "general",
        franchise: str = "",
    ):
        """
        Add one Wikipedia page to the knowledge base.

        Fix 2: Detects "List of X characters" pages and uses the section
               title as the entity tag for each chunk, rather than the
               requested character name, avoiding mass entity mislabeling.
        Fix 6: Section noise filter is now case-insensitive.
        """
        if not page_data:
            return

        title = page_data.get("title", "Unknown")
        char_name = page_data.get("character_name", "")
        entity = char_name if char_name else title

        # detect list/character-aggregate pages
        is_list_page = (
            "list of" in title.lower()
            or ("characters" in title.lower() and len(title.split()) > 2)
        )

        texts: List[str] = []
        metas: List[Dict] = []

        base_meta = {
            "source": "wikipedia",
            "title": title,
            "franchise": franchise,
            "entity": entity,
            "type": source_type,
        }

        # 1. Summary / Overview (entity = requested char or title)
        summary = page_data.get("summary", "").strip()
        if summary:
            for chunk in self._chunk_text(summary):
                texts.append(f"[{title} - Overview]\n{chunk}")
                metas.append({**base_meta, "section": "overview"})

        # 2. Content sections — noise filtered, entity-aware
        char_section_text = ""   # collect the char's own section
        for sec_title, sec_text in page_data.get("sections", {}).items():
            # case-insensitive noise check
            if sec_title.lower() in _NOISE_LOWER:
                continue

            # on list pages, each section is a different character
            if is_list_page and 1 <= len(sec_title.split()) <= 4:
                current_entity = sec_title
                # Capture the specific section for this character's profile
                if char_name and char_name.lower() in sec_title.lower():
                    char_section_text = sec_text
            else:
                current_entity = entity

            is_priority = any(kw in sec_title.lower() for kw in PRIORITY_SECTIONS)
            prefix = f"[{title} - {sec_title}]\n"
            for chunk in self._chunk_text(sec_text):
                texts.append(prefix + chunk)
                metas.append({
                    **base_meta,
                    "entity": current_entity,   # Fix 2: per-section entity
                    "section": sec_title,
                    "priority": is_priority,
                })

        if texts:
            self.store.add_documents(texts, metas)
            logger.info(f"   Added Wikipedia page: {title} ({len(texts)} chunks)")

        # 3. Character Profile chunk
        if source_type == "character":
            profile = self._make_character_profile(
                character_name=entity,
                franchise=franchise,
                page_data=page_data,
                char_section_text=char_section_text,   # Fix 2
            )
            if profile:
                self.store.add_documents([profile["text"]], [profile["metadata"]])
                logger.info(f"   Added character profile: {entity}")

    # ── Add local documents ───────────────────────────────

    def add_local_documents(self, folder_path: str, franchise: str = ""):
        folder = Path(folder_path)
        if not folder.exists():
            logger.warning(f"Local document folder not found: {folder_path}")
            return

        supported_ext = {".txt", ".md", ".json"}
        files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix in supported_ext
        ]
        if not files:
            logger.warning(f"No supported documents found in: {folder_path}")
            return

        logger.info(f"Loading local documents ({len(files)} files)...")
        for fp in files:
            try:
                content = fp.read_text(encoding="utf-8")
                if fp.suffix == ".json":
                    try:
                        content = json.dumps(
                            json.loads(content), ensure_ascii=False, indent=2
                        )
                    except json.JSONDecodeError:
                        pass

                chunks = self._chunk_text(content)
                meta_list = [{
                    "source": "local", "filename": fp.name,
                    "franchise": franchise, "entity": "", "type": "document",
                }] * len(chunks)
                self.store.add_documents(chunks, meta_list)
                logger.info(f"   Loaded: {fp.name} ({len(chunks)} chunks)")

            except Exception as e:
                logger.warning(f"   Failed to read: {fp.name} — {e}")

    # ── Build from franchise ──────────────────────────────

    def build_from_franchise(
        self,
        franchise: str,
        characters: List[str] = None,
    ) -> int:
        logger.info(f"\nBuilding Wikipedia knowledge base (franchise: {franchise})")
        initial_count = self.get_document_count()

        franchise_page = self.fetcher.fetch_franchise(franchise)
        if franchise_page:
            self.add_wikipedia_page(franchise_page, source_type="franchise", franchise=franchise)
        else:
            logger.warning(f"Franchise page not found: {franchise}; adding placeholder.")
            self.store.add_documents(
                [f"Franchise: {franchise}\nThis fan-fiction visual novel is based on {franchise}."],
                [{"source": "manual", "type": "franchise",
                  "franchise": franchise, "entity": franchise, "title": franchise}],
            )

        for char_name in (characters or []):
            char_page = self.fetcher.fetch_character(char_name, franchise)
            if char_page:
                self.add_wikipedia_page(char_page, source_type="character", franchise=franchise)
            else:
                logger.warning(f"Character page not found: {char_name}; adding placeholder.")
                self.store.add_documents(
                    [f"Character: {char_name}\nThis character is from {franchise}."],
                    [{"source": "manual", "type": "character",
                      "franchise": franchise, "entity": char_name, "title": char_name}],
                )

        added = self.get_document_count() - initial_count
        logger.info(f"Knowledge base built — {added} new chunks added.")
        return added

    # ── Search ────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        franchise_filter: str = "",
        entity_filter: str = "",
    ) -> List[Dict]:
        """BM25 recall + global-IDF cosine rerank with optional metadata filters."""
        return self.store.search(
            query,
            n_results=n_results,
            franchise_filter=franchise_filter,
            entity_filter=entity_filter,
        )
