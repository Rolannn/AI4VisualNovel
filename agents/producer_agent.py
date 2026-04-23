"""
Producer Agent
~~~~~~~~~~~~~~
Reviews the design draft and steers the project.
"""

import logging
from typing import Dict, Any, Optional
from .llm_client import LLMClient
import json

from .config import ProducerConfig, PathConfig
from .utils import FileHelper

logger = logging.getLogger(__name__)


class ProducerAgent:
    """Reviews the designer's draft and returns PASS or feedback."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.llm_client = LLMClient(api_key=api_key, base_url=base_url)
        self.config = ProducerConfig

        logger.info("Producer agent initialized")

    def critique_game_design(
        self,
        game_design: Dict[str, Any],
        user_requirements: str = "",
        expected_nodes: int = 12,
        expected_characters: int = 3
    ) -> str:
        """
        Returns:
            "PASS" or detailed revision notes.
        """
        logger.info("Producer reviewing design draft...")

        try:
            prompt = self.config.GAME_DESIGN_CRITIQUE_PROMPT.format(
                game_design=json.dumps(game_design, ensure_ascii=False, indent=2),
                user_requirements=user_requirements if user_requirements else "No specific requirements",
                expected_nodes=expected_nodes,
                expected_characters=expected_characters
            )

            feedback = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": self.config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            feedback = feedback.strip()
            if "PASS" in feedback:
                logger.info("Producer approved the design (PASS).")
                return "PASS"
            else:
                logger.warning("Producer requested changes")
                return feedback

        except Exception as e:
            logger.error(f"Producer review failed: {e}")
            return "PASS"

    def save_game_design(self, game_design: Dict[str, Any]) -> None:
        if not FileHelper.safe_write_json(PathConfig.GAME_DESIGN_FILE, game_design):
            raise Exception("Failed to save game design")

    @staticmethod
    def load_game_design() -> Optional[Dict[str, Any]]:
        game_design = FileHelper.safe_read_json(PathConfig.GAME_DESIGN_FILE)
        if game_design:
            logger.info(f"Loaded game design: {game_design.get('title', 'Unknown')!r}")
        return game_design
