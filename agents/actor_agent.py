"""
Actor Agent
~~~~~~~~~~~
In-character performance and sprite critique.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from .llm_client import LLMClient

from .config import ActorConfig

logger = logging.getLogger(__name__)


class ActorAgent:
    """Plays a character and reviews generated sprites."""

    def __init__(self, character_info: Dict[str, Any], api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Args:
            character_info: Dict with name, personality, background, etc.
        """
        self.llm_client = LLMClient(api_key=api_key, base_url=base_url)
        self.config = ActorConfig
        self.character_info = character_info
        self.name = character_info.get('name', 'Unknown')
        self.is_protagonist = character_info.get('is_protagonist', False)

        logger.info(f"Actor agent ({self.name}) initialized")

    def perform_plot(
        self,
        plot_summary: str,
        other_characters: List[Dict[str, Any]],
        story_context: str,
        character_expressions: List[str] = []
    ) -> str:
        """Generate the next in-character lines for the current beat."""
        logger.info(f"Actor {self.name} performing a beat...")

        script_label = "I" if self.is_protagonist else self.name

        other_chars_info = "\n".join([
            f"- {char.get('name', 'Unknown')} ({char.get('gender', '')}, {char.get('personality', '')}): "
            f"{char.get('appearance', '')}. Background: {char.get('background', '')[:80]}..."
            for char in other_characters
        ])

        prompt = self.config.PERFORM_PROMPT.format(
            name=self.name,
            script_label=script_label,
            plot_summary=plot_summary,
            other_characters=other_chars_info,
            story_context=story_context,
            character_expressions=", ".join(character_expressions)
        )

        system_prompt = self.config.SYSTEM_PROMPT.format(
            name=self.name,
            personality=self.character_info.get('personality', ''),
            background=self.character_info.get('background', '')
        )

        try:
            return self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9
            )
        except Exception as e:
            logger.error(f"Performance failed: {e}")
            return ""

    def critique_visual(
        self,
        image_path: str,
        expression: str = "neutral",
        reference_image_path: Optional[str] = None,
        story_background: Optional[str] = None,
        art_style: Optional[str] = None
    ) -> str:
        """
        Review a sprite in character.

        Returns:
            "PASS" or feedback text
        """
        logger.info(f"Actor {self.name} reviewing sprite: {image_path} (expression: {expression})...")

        system_prompt = self.config.SYSTEM_PROMPT.format(
            name=self.name,
            personality=self.character_info.get('personality', ''),
            background=self.character_info.get('background', '')
        )

        user_prompt = self.config.IMAGE_CRITIQUE_PROMPT.format(
            story_background=story_background or "A visual novel game",
            art_style=art_style or "Japanese anime style",
            appearance=self.character_info.get('appearance', ''),
            expression=expression
        )

        try:
            content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_path}}
            ]

            if reference_image_path and expression != "neutral":
                content.insert(1, {
                    "type": "text",
                    "text": "Your standard neutral sprite for reference:"
                })
                content.insert(2, {"type": "image_url", "image_url": {"url": reference_image_path}})

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]

            feedback = self.llm_client.chat_completion(
                messages=messages,
                temperature=self.config.TEMPERATURE
            )

            feedback = feedback.strip()

            logger.info(f"Actor {self.name} review:\n{feedback}")

            if "PASS" in feedback:
                logger.info(f"Actor {self.name} approved the sprite")
                return "PASS"
            else:
                logger.warning(f"Actor {self.name} requested changes")
                return feedback

        except Exception as e:
            logger.error(f"Sprite critique failed for {self.name}: {str(e)}")
            return "PASS"

    def generate_expression_description(self, expression_name: str) -> str:
        """Describe a facial expression in words (for image generation)."""
        prompt = self.config.EXPRESSION_DESCRIPTION_PROMPT.format(
            name=self.name,
            expression=expression_name,
            character_info=json.dumps(self.character_info, ensure_ascii=False, indent=2)
        )

        system_prompt = self.config.SYSTEM_PROMPT.format(
            name=self.name,
            personality=self.character_info.get('personality', ''),
            background=self.character_info.get('background', '')
        )

        try:
            description = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return description.strip()
        except Exception as e:
            logger.error(f"Expression description failed ({expression_name}): {e}")
            return f"{self.name} with {expression_name} expression"
