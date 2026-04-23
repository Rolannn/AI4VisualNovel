"""
Designer Agent
~~~~~~~~~~~~~~
Drafts the overall game design document.
"""

import logging
from typing import Dict, Any, Optional
from .llm_client import LLMClient
import json

from .config import DesignerConfig, PathConfig
from .utils import JSONParser, FileHelper

logger = logging.getLogger(__name__)


class DesignerAgent:
    """Designer Agent — drafts the game design JSON."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the designer agent."""
        self.llm_client = LLMClient(api_key=api_key, base_url=base_url)
        self.config = DesignerConfig

        logger.info("Designer Agent initialized successfully")

    def generate_game_design(
        self,
        character_count: int = None,
        requirements: str = "",
        feedback: str = None,
        previous_game_design: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate or revise the game design.

        Args:
            character_count: Number of characters (including protagonist).
            requirements: User requirements text.
            feedback: Producer feedback (optional, revision mode).
            previous_game_design: Prior design (optional, revision mode).
        """
        character_count = character_count or self.config.DEFAULT_CHARACTER_COUNT

        logger.info("Generating game design...")

        try:
            user_prompt = self.config.GAME_DESIGN_PROMPT.format(
                character_count=character_count,
                total_nodes=self.config.TOTAL_NODES,
                requirements=requirements if requirements else "(none — create freely in English per system rules)"
            )

            if feedback and previous_game_design:
                logger.info("Revision mode: applying producer feedback...")
                user_prompt += (
                    "\n\n[Previous game design]\n"
                    f"{json.dumps(previous_game_design, ensure_ascii=False, indent=2)}\n\n"
                    f"[Producer feedback]\n{feedback}\n\n"
                    "Update the design to address the feedback. Keep the same JSON structure; "
                    "only change content. All in-game text must remain in English."
                )

            content = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": self.config.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.TEMPERATURE,
                json_mode=True
            )

            game_design = JSONParser.parse_ai_response(content)

            required_fields = ["title", "background", "story_graph", "characters", "scenes"]
            if not JSONParser.validate_required_fields(game_design, required_fields):
                raise ValueError("Generated design is missing required fields")

            sg = game_design.get("story_graph", {})
            raw_nodes = sg.get("nodes", {})
            actual_count = len(raw_nodes) if isinstance(raw_nodes, dict) else len(raw_nodes)
            target = self.config.TOTAL_NODES
            if actual_count > target + 1:
                logger.warning(
                    f"LLM produced {actual_count} nodes, above target {target}±1. "
                    f"Consider adjusting GAME_TOTAL_NODES in .env or regenerating."
                )
            else:
                logger.info(f"Node count within range: {actual_count}/{target}")

            logger.info(f"Game design complete: {game_design['title']!r}")
            return game_design

        except Exception as e:
            logger.error(f"Game design generation failed: {e}")
            raise
