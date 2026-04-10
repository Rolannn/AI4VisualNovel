"""
Writer Agent
~~~~~~~~~~~~
Writer Agent - generates detailed story content.
Uses the configured LLM to create dialogue and events based on the game design and character state.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional
from .llm_client import LLMClient

from .config import APIConfig, WriterConfig, PathConfig, ArtistConfig
from .utils import JSONParser, FileHelper, TextProcessor

logger = logging.getLogger(__name__)


class WriterAgent:
    """Writer Agent - Story generator"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the Writer Agent.
        
        Args:
            api_key: API key
            base_url: API base URL
        """
        self.llm_client = LLMClient(api_key=api_key, base_url=base_url)
        self.config = WriterConfig
        
        logger.info("Writer Agent initialized successfully")
    
    def split_node_into_plots(
        self,
        node_summary: str,
        long_term_memory: str,
        available_scenes: List[str] = [],
        available_characters: List[Dict[str, Any]] = [],
        segment_count: int = 3
    ) -> List[Dict[str, Any]]:
        """Split the node summary into plot segments"""
        logger.info(f"Splitting plot segments (target segments: {segment_count})...")
        
        # 根据 segment_count 生成不同的指令
        if segment_count == 1:
            split_instruction = "Keep as a single complete scene. Do not split."
        else:
            split_instruction = "Each segment should be a small scene or event with a clear conflict or action."
        
        scenes_str = ", ".join(available_scenes) if available_scenes else "unspecified; choose based on plot needs"
        # Build character details
        characters_info = "\n".join([
            f"- {char.get('name', 'Unknown')} ({char.get('gender', '')},{char.get('personality', '')}): {char.get('appearance', '')}. Background: {char.get('background', '')[:80]}..."
            for char in available_characters
        ]) if available_characters else "no characters specified"

        prompt = self.config.PLOT_SPLIT_PROMPT.format(
            segment_count=segment_count,
            split_instruction=split_instruction,
            node_summary=node_summary,
            previous_story_summary=long_term_memory,
            available_scenes=scenes_str,
            available_characters=characters_info
        )
        try:
            # Use a dedicated system prompt to ensure JSON format
            system_prompt = "You are a plot structure assistant. Split the plot summary into structured segments and output strictly in JSON."
            
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return JSONParser.parse_ai_response(response)
        except Exception as e:
            logger.error(f"Split plot failed: {e}")
            return []

    def synthesize_script(
        self,
        plot_performances: List[Dict[str, Any]],
        choices: List[Dict[str, Any]] = [],
        story_context: str = "",
        available_scenes: List[str] = [],
        available_characters: List[Dict[str, Any]] = []
    ) -> str:
        """Synthesize actor performances into a script"""
        logger.info("Synthesizing script (from structured data)...")
        
        # 将结构化数据转换为 JSON 字符串供 LLM 阅读
        performances_json = json.dumps(plot_performances, ensure_ascii=False, indent=2)
        choices_json = json.dumps(choices, ensure_ascii=False, indent=2)
        scenes_str = ", ".join(available_scenes) if available_scenes else "unspecified"
        # Build character details
        characters_info = "\n".join([
            f"- {char.get('name', 'Unknown')} ({char.get('gender', '')},{char.get('personality', '')}): {char.get('appearance', '')}. Background: {char.get('background', '')[:80]}..."
            for char in available_characters
        ]) if available_characters else "unspecified"
        
        prompt = self.config.PLOT_SYNTHESIS_PROMPT.format(
            plot_performances=performances_json,
            choices=choices_json,
            story_context=story_context,
            available_scenes=scenes_str,
            available_characters=characters_info
        )
        try:
            return self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": self.config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"Synthesize script failed: {e}")
            return str(plot_performances)

    def decide_next_speaker(
        self,
        plot_summary: str,
        characters: List[Dict[str, Any]],
        story_context: str
    ) -> tuple[str, str]:
        """
        Decide the next speaker and brief guidance.
        
        Returns:
            (character_name, guidance) or ("STOP", "")
        """
        # Build character details
        characters_info = "\n".join([
            f"- {char.get('name', 'Unknown')} ({char.get('gender', '')},{char.get('personality', '')}): {char.get('appearance', '')}. Background: {char.get('background', '')[:80]}..."
            for char in characters
        ])
        
        prompt = self.config.NEXT_SPEAKER_PROMPT.format(
            plot_summary=plot_summary,
            characters=characters_info,
            story_context=story_context
        )
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a directing assistant that helps coordinate plot generation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3 # low temperature for more deterministic results
            )
            response = response.strip()
            
            # Parse response
            import re
            
            # Extract <character> tag
            char_match = re.search(r'<character>(.+?)</character>', response, re.DOTALL)
            if not char_match:
                logger.warning("Director response format error: missing <character> tag")
                return "STOP", ""
            
            speaker = char_match.group(1).strip()
            
            # Check STOP
            if speaker.upper() == "STOP":
                return "STOP", ""
            
            # Extract <advice> tag
            advice_match = re.search(r'<advice>(.+?)</advice>', response, re.DOTALL)
            guidance = advice_match.group(1).strip() if advice_match else ""
            
            logger.debug(f"Parsed result: character={speaker}, guidance={guidance}")
            return speaker, guidance
            
        except Exception as e:
            logger.error(f"Decide next speaker failed: {e}")
            return "STOP", ""
    
    def append_story(self, story_text: str) -> None:
        """
        Append new story content to story.txt
        
        Args:
            story_text: story text to append
        """
        if not FileHelper.safe_append_text(PathConfig.STORY_FILE, story_text):
            raise Exception("Failed to append story")
    
    @staticmethod
    def load_story() -> str:
        """
        Load full story.txt
        
        Returns:
            Story text; returns empty string if file does not exist
        """
        try:
            with open(PathConfig.STORY_FILE, 'r', encoding='utf-8') as f:
                story = f.read()
            
            logger.info(f"Story file loaded: {len(story)} characters")
            return story
            
        except FileNotFoundError:
            logger.warning("Story file not found; a new file will be created on first write")
            return ""
        except Exception as e:
            logger.error(f"Load story file failed: {e}")
            return ""
    

    
    def parse_story_for_ui(self, story_text: str) -> List[Dict[str, Any]]:
        """
        Parse story text into a UI-friendly structure
        
        Args:
            story_text: raw story text
            
        Returns:
            List of story segments; each contains dialogue, images, choices, etc.
        """
        logger.info("Parsing story text...")
        
        segments = []
        current_location = None
        current_time = None
        
        lines = story_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Parse scene title (## Location or ## Location - Time)
            scene_match = re.match(r'##\s*(.+)', line)
            
            if scene_match:
                content = scene_match.group(1).strip()
                if '-' in content:
                    parts = content.split('-', 1)
                    new_location = parts[0].strip()
                    new_time = parts[1].strip()
                else:
                    new_location = content
                    new_time = "Day" # default time
                
                # If scene changes, add scene switch marker
                if new_location != current_location:
                    segments.append({
                        "type": "scene",
                        "location": new_location,
                        "time": new_time
                    })
                
                current_time = new_time
                current_location = new_location
                continue
            
            # Parse image tag <image id="Character">expression</image>
            image_match = re.match(r'<image\s+id="([^"]+)">([^<]+)</image>', line)
            if image_match:
                character = image_match.group(1).strip()
                expression = image_match.group(2).strip()
                
                segments.append({
                    "type": "image",
                    "character": character,
                    "expression": expression,
                    "location": current_location,
                    "time": current_time
                })
                continue
            
            # Parse dialogue (Character: "text")
            dialogue_match = re.match(r'([^:]+):\s*"?(.+?)"?$', line)
            if dialogue_match:
                speaker = dialogue_match.group(1).strip()
                text = dialogue_match.group(2).strip()
                segments.append({
                    "type": "dialogue",
                    "speaker": speaker if speaker != "NARRATOR" else None,
                    "text": text,
                    "location": current_location,
                    "time": current_time
                })
                continue
            
            # Parse choice start ([CHOICE])
            if line == '[CHOICE]':
                segments.append({
                    "type": "choice_start",
                    "location": current_location,
                    "time": current_time
                })
                continue
            
            # Parse choice option (Option1: "text" → [effects])
            choice_match = re.match(r'选项(\d+):\s*"(.+?)"\s*→\s*\[(.+?)\]', line)
            if choice_match:
                choice_num = int(choice_match.group(1))
                choice_text = choice_match.group(2).strip()
                effects_str = choice_match.group(3).strip()
                
                # Parse effects
                effects = self._parse_choice_effects(effects_str)
                
                segments.append({
                    "type": "choice_option",
                    "number": choice_num,
                    "text": choice_text,
                    "effects": effects
                })
                continue
        
        logger.info(f"Parsing complete: {len(segments)} segments")
        return segments
    
    def summarize_story(self, story_content: str) -> str:
        """
        Generate story summary
        
        Args:
            story_content: story content
            
        Returns:
            Story summary
        """
        logger.info("Generating story summary...")
        
        try:
            prompt = self.config.SUMMARY_PROMPT.format(story_content=story_content)
            
            summary = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an assistant skilled at summarizing stories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Generate summary failed: {str(e)}")
            return story_content[-500:]  # fallback to last section on failure
