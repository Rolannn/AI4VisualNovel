"""
RAG Agent
~~~~~~~~~
Retrieval-Augmented Generation agent for fan-fiction visual novel creation.

Responsibilities:
1. Manage knowledge base construction and caching per franchise
2. Retrieve relevant IP world / character knowledge by query
3. Format and inject retrieved context into DesignerAgent / WriterAgent prompts

Usage example:
    from agents.rag_agent import RAGAgent

    rag = RAGAgent(franchise="Chainsaw Man")
    rag.build_knowledge_base(
        franchise="Chainsaw Man",
        characters=["Makima", "Denji", "Power"],
        docs_dir="data/rag_docs"   # optional local documents
    )

    augmented_req = rag.build_requirements_with_rag(
        user_requirements="A dramatic confrontation scene at dawn",
        franchise="Chainsaw Man",
        characters=["Makima", "Denji"]
    )
"""

import os
import logging
from typing import List, Dict, Optional

from .knowledge_builder import KnowledgeBuilder
from .config import PathConfig

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG Retrieval Agent.

    Maintains a separate knowledge base file per franchise
    to prevent cross-project knowledge contamination.
    """

    DEFAULT_N_RESULTS = 5

    def __init__(self, franchise: str = "", language: str = "en"):
        """
        Args:
            franchise: IP name, e.g. "Chainsaw Man"
            language:  Wikipedia language, "en" or "zh"
        """
        self.franchise = franchise
        self.language = language

        # Knowledge base path: data/rag/knowledge_<safe_name>.json
        safe_name = (
            "".join(c if c.isalnum() or c == "_" else "_" for c in franchise)
            if franchise
            else "default"
        )
        rag_dir = os.path.join(PathConfig.DATA_DIR, "rag")
        os.makedirs(rag_dir, exist_ok=True)
        store_path = os.path.join(rag_dir, f"knowledge_{safe_name}.json")

        self.kb = KnowledgeBuilder(store_path=store_path, language=language)
        logger.info(
            f"RAG Agent initialized\n"
            f"  Franchise : {franchise or '(unspecified)'}\n"
            f"  KB path   : {store_path}"
        )

    # ─────────────────────────────────────────────
    # Knowledge base construction
    # ─────────────────────────────────────────────

    def build_knowledge_base(
        self,
        franchise: str,
        characters: List[str] = None,
        docs_dir: str = None,
        force_rebuild: bool = False,
    ) -> int:
        """
        Build / update the knowledge base.
        Skips construction if the KB already exists unless force_rebuild=True.

        Args:
            franchise:     IP name
            characters:    List of character names to fetch
            docs_dir:      Optional local document directory
            force_rebuild: Clear existing KB and rebuild from scratch

        Returns:
            Total document count in the knowledge base after construction.
        """
        logger.info("=" * 55)
        logger.info(f"RAG Knowledge Base: {franchise}")
        logger.info("=" * 55)

        if force_rebuild:
            logger.info("Force rebuild: clearing existing knowledge base...")
            self.kb.clear()

        existing = self.kb.get_document_count()
        if existing > 0 and not force_rebuild:
            logger.info(
                f"Knowledge base already exists ({existing} documents). "
                "Pass force_rebuild=True to rebuild."
            )
            return existing

        # Fetch from Wikipedia
        if franchise:
            self.kb.build_from_franchise(
                franchise=franchise,
                characters=characters or [],
            )

        # Load local documents (optional)
        if docs_dir:
            logger.info(f"Loading local documents: {docs_dir}")
            self.kb.add_local_documents(docs_dir, franchise=franchise)

        total = self.kb.get_document_count()
        logger.info(f"Knowledge base ready — {total} document chunks total.")
        return total

    # ─────────────────────────────────────────────
    # Retrieval interface
    # ─────────────────────────────────────────────

    def retrieve_character_context(
        self,
        character_name: str,
        n_results: int = DEFAULT_N_RESULTS,
    ) -> str:
        """
        Retrieve relevant knowledge for a specific character.
        Uses entity_filter + franchise_filter to focus the search pool.

        Returns:
            Formatted character knowledge string for prompt injection.
        """
        # Prioritize character profile chunks, then personality/appearance
        query = (
            f"{character_name} character personality appearance "
            f"traits background abilities relationships"
        )
        results = self.kb.search(
            query,
            n_results=n_results,
            franchise_filter=self.franchise,
            entity_filter=character_name,
        )

        # Fallback: search without entity filter if no results
        if not results:
            results = self.kb.search(
                query,
                n_results=n_results,
                franchise_filter=self.franchise,
            )

        if not results:
            return ""

        parts = [f"[Character Knowledge: {character_name}]"]
        seen: set = set()
        for doc in results:
            text = doc["text"].strip()
            if text not in seen:
                parts.append(text)
                seen.add(text)

        return "\n\n".join(parts)

    def retrieve_world_context(
        self,
        topic: str = "",
        n_results: int = DEFAULT_N_RESULTS,
    ) -> str:
        """
        Retrieve world-building / lore knowledge for the franchise.

        Returns:
            Formatted world knowledge string for prompt injection.
        """
        query = topic or f"{self.franchise} world setting lore history power system"
        results = self.kb.search(
            query,
            n_results=n_results,
            franchise_filter=self.franchise,
        )

        if not results:
            return ""

        parts = [f"[World Knowledge: {self.franchise}]"]
        seen: set = set()
        for doc in results:
            text = doc["text"].strip()
            if text not in seen:
                parts.append(text)
                seen.add(text)

        return "\n\n".join(parts)

    def get_franchise_overview(self, n_results: int = 3) -> str:
        """
        Retrieve franchise overview (summary / overview sections).
        Used as base background for game design.
        """
        results = self.kb.search(
            f"{self.franchise} overview plot summary introduction setting",
            n_results=n_results,
            franchise_filter=self.franchise,
        )

        if not results:
            return f"This is a fan-fiction visual novel based on {self.franchise}."

        parts: List[str] = []
        seen: set = set()
        for doc in results:
            text = doc["text"].strip()
            if text not in seen:
                parts.append(text)
                seen.add(text)

        return "\n\n".join(parts)

    # ─────────────────────────────────────────────
    # Prompt augmentation
    # ─────────────────────────────────────────────

    def build_requirements_with_rag(
        self,
        user_requirements: str,
        franchise: str,
        characters: List[str],
    ) -> str:
        """
        Merge the user's creation brief with RAG-retrieved IP knowledge
        into an augmented requirements string for DesignerAgent.

        Args:
            user_requirements: Raw user brief (may be empty)
            franchise:         IP name
            characters:        List of characters to feature

        Returns:
            Augmented requirements string with world + character knowledge injected.
        """
        sep = "=" * 50
        sections: List[str] = []

        # ── 0. Fan-fiction directive ─────────────────────
        char_list = ", ".join(characters) if characters else "(free choice)"
        sections.append(
            f"[Fan-Fiction Directive]\n"
            f"This game is a fan-fiction visual novel based on the IP '{franchise}'.\n"
            f"The following characters MUST appear as main cast "
            f"(personality, appearance, and backstory must match canon):\n"
            f"  -> {char_list}\n"
            f"The plot may be original, but character behavior and traits "
            f"must strictly follow the source material."
        )

        # ── 1. User brief ────────────────────────────────
        if user_requirements and user_requirements.strip():
            sections.append(f"[User Requirements]\n{user_requirements.strip()}")

        # ── 2. Franchise world overview ──────────────────
        world_ctx = self.get_franchise_overview()
        if world_ctx:
            sections.append(f"[World Knowledge: {franchise}]\n{world_ctx}")

        # ── 3. Per-character knowledge ───────────────────
        for char_name in characters:
            char_ctx = self.retrieve_character_context(char_name, n_results=4)
            if char_ctx:
                sections.append(char_ctx)
            else:
                sections.append(
                    f"[Character Knowledge: {char_name}]\n"
                    f"{char_name} is a character from {franchise}.\n"
                    f"Please design dialogue and personality consistent with "
                    f"their canonical portrayal."
                )

        # ── 4. Format reminder ───────────────────────────
        sections.append(
            "[Important Format Requirements]\n"
            "- Use character names consistent with the source material.\n"
            "- The 'appearance' field MUST reference the Character Knowledge above,\n"
            "  including hair color, hairstyle, eye color, and clothing details,\n"
            "  so that accurate character sprites can be generated.\n"
            "- The 'personality' and 'background' fields must align with canon."
        )

        result = f"\n{sep}\n" + f"\n\n{sep}\n".join(sections) + f"\n{sep}\n"
        logger.info(
            f"RAG-augmented requirements built "
            f"({len(result)} chars, {len(sections)} knowledge blocks)"
        )
        return result

    # ─────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Returns True if the knowledge base has content."""
        return self.kb.get_document_count() > 0

    def get_stats(self) -> Dict:
        """Return knowledge base statistics."""
        docs = self.kb.store.documents
        type_counts: Dict[str, int] = {}
        franchise_counts: Dict[str, int] = {}

        for doc in docs:
            meta = doc.get("metadata", {})
            t = meta.get("type", "unknown")
            f = meta.get("franchise", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
            franchise_counts[f] = franchise_counts.get(f, 0) + 1

        return {
            "franchise": self.franchise,
            "total_documents": len(docs),
            "type_distribution": type_counts,
            "franchise_distribution": franchise_counts,
            "store_path": self.kb.store_path,
        }
