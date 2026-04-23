import json
import re
from typing import Dict, List, Optional
from .config import DataPaths


class GameDataLoader:
    """Loads `game_design.json` and `story.txt`."""

    @staticmethod
    def load_game_design() -> Optional[Dict]:
        if not DataPaths.GAME_DESIGN_FILE.exists():
            print(f"Game design file not found: {DataPaths.GAME_DESIGN_FILE}")
            return None

        with open(DataPaths.GAME_DESIGN_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_story() -> Optional[str]:
        if not DataPaths.STORY_FILE.exists():
            print(f"Story file not found: {DataPaths.STORY_FILE}")
            return None

        with open(DataPaths.STORY_FILE, 'r', encoding='utf-8') as f:
            return f.read()


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith('=== End'):
        return True
    # Legacy / bilingual boilerplate from some model outputs
    if s.startswith("\u8fd9\u91cc\u4e3a\u60a8\u751f\u6210") or s.lower().startswith("here is the generated"):
        return True
    return False


class StoryParser:
    """Parses AI script text into per-node line lists (DAG / node blocks)."""

    @staticmethod
    def parse_story(story_text: str) -> Dict[str, List[Dict]]:
        """
        Returns:
            { "node_id": [ {type, ...}, ... ], ... }
        """
        nodes = {}
        current_node_id = None
        current_lines = []

        lines = story_text.strip().split('\n')

        for line in lines:
            line = line.strip()

            if _is_noise_line(line):
                continue

            node_match = re.match(r'===\s*Node:\s*(.+?)\s*===', line, re.IGNORECASE)
            if node_match:
                if current_node_id:
                    nodes[current_node_id] = current_lines

                current_node_id = node_match.group(1).strip()
                current_lines = []
                print(f"Parsing node: {current_node_id}")
                continue

            if current_node_id:
                parsed = StoryParser._parse_line(line)
                if parsed:
                    current_lines.append(parsed)

        if current_node_id:
            nodes[current_node_id] = current_lines

        return nodes

    @staticmethod
    def _parse_line(line: str) -> Optional[Dict]:
        scene_tag_match = re.match(r'<scene>(.+?)</scene>', line)
        if scene_tag_match:
            return {"type": "scene", "value": scene_tag_match.group(1).strip()}

        if_match = re.match(r'\[IF: (.+?) >= (\d+)\]', line)
        if if_match:
            return {
                "type": "if",
                "condition_role": if_match.group(1),
                "condition_level": int(if_match.group(2))
            }

        if line == '[ELSE]':
            return {"type": "else"}

        if line == '[ENDIF]':
            return {"type": "endif"}

        image_match = re.match(r'<image\s+id="([^"]+)">([^<]+)</image>', line)
        if image_match:
            char_name = image_match.group(1)
            expression = image_match.group(2).strip()
            return {"type": "image", "value": f"{char_name}-{expression}"}

        content_match = re.match(r'<content\s+id="([^"]+)">([^<]+)</content>', line)
        if content_match:
            speaker = content_match.group(1).strip()
            text = content_match.group(2).strip()

            s_low = speaker.lower()
            if s_low in ("narration", "narrator", "\u65c1\u767d"):
                return {"type": "narrator", "text": text}
            else:
                return {"type": "dialogue", "speaker": speaker, "text": text, "emotion": "neutral"}

        jump_match = re.match(r'\[JUMP: (.+?)\]', line)
        if jump_match:
            return {"type": "jump", "target": jump_match.group(1)}

        if line == '[CHOICE]':
            return {"type": "choice_start"}

        xml_choice_match = re.match(r'<choice\s+target="([^"]+)">(.+?)</choice>', line)
        if xml_choice_match:
            return {
                "type": "choice_option",
                "index": None,
                "text": xml_choice_match.group(2).strip(),
                "target": xml_choice_match.group(1).strip()
            }

        return None
