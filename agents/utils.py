"""
Agent Utils
~~~~~~~~~~~
Shared utilities for all agents.
"""

import json
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class JSONParser:
    """JSON parsing with common LLM output fixes."""

    @staticmethod
    def parse_ai_response(content: str, save_on_fail: bool = True) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Direct JSON parse failed: {e}")
            logger.info("Attempting to repair JSON...")

            try:
                fixed_content = JSONParser.fix_json_format(content)
                result = json.loads(fixed_content)
                logger.info("JSON repair succeeded")
                return result
            except json.JSONDecodeError as e2:
                logger.error(f"JSON repair failed: {e2}")

                if save_on_fail:
                    JSONParser._save_failed_response(content, e2)

                logger.error("Tip: see logs/failed_json*.txt for the raw model output")
                logger.error("Common cause: unescaped quotes or newlines inside strings")
                raise

    @staticmethod
    def _save_failed_response(content: str, error: Exception) -> None:
        import os
        from datetime import datetime

        try:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
            os.makedirs(log_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(log_dir, f'failed_json_{timestamp}.txt')

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"=== JSON parse failed ===\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"Error: {error}\n")
                f.write(f"\n=== Raw content ===\n")
                f.write(content)

            logger.info(f"Saved failed response to {file_path}")
        except Exception as e:
            logger.error(f"Could not save failed response: {e}")

    @staticmethod
    def fix_json_format(content: str) -> str:
        original_content = content

        # 1. Strip markdown code fences
        content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'```', '', content)

        # 2. BOM and trim
        content = content.strip('\ufeff').strip()

        # 3. Extract JSON object or array
        start_obj = content.find('{')
        start_arr = content.find('[')

        is_object = False
        if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
            start = start_obj
            open_char = '{'
            close_char = '}'
            is_object = True
        elif start_arr != -1:
            start = start_arr
            open_char = '['
            close_char = ']'
        else:
            return content

        if start != -1:
            count = 0
            end = -1
            in_string = False
            escape = False

            for i in range(start, len(content)):
                char = content[i]

                if escape:
                    escape = False
                    continue

                if char == '\\':
                    escape = True
                    continue

                if char == '"':
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == open_char:
                        count += 1
                    elif char == close_char:
                        count -= 1
                        if count == 0:
                            end = i
                            break

            if end != -1:
                content = content[start:end+1]
            else:
                end = content.rfind(close_char)
                if end != -1 and start < end:
                    content = content[start:end+1]

        # 4. Strip // and /* */ comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # 5. Trailing commas
        content = re.sub(r',\s*]', ']', content)
        content = re.sub(r',\s*}', '}', content)

        # 6-7. Smart quotes to ASCII
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace(''', "'").replace(''', "'")

        logger.debug(f"fix_json: {len(original_content)} -> {len(content)} chars")

        return content

    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: list) -> bool:
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            logger.error(f"Missing required fields: {', '.join(missing_fields)}")
            return False

        return True


class PromptBuilder:
    """Format prompt templates with safe defaults for missing keys."""

    @staticmethod
    def format_with_fallback(template: str, **kwargs) -> str:
        placeholders = re.findall(r'\{(\w+)\}', template)

        defaults = {
            'game_type': 'school romance',
            'game_style': 'light and warm',
            'character_count': 3,
            'name': 'Character',
            'appearance': 'anime style character',
            'personality': 'friendly',
            'expression': 'neutral',
            'color': [100, 149, 237]
        }

        for placeholder in placeholders:
            if placeholder not in kwargs and placeholder in defaults:
                kwargs[placeholder] = defaults[placeholder]

        return template.format(**kwargs)


class FileHelper:
    """File read/write helpers."""

    @staticmethod
    def safe_write_json(file_path: str, data: Dict[str, Any]) -> bool:
        try:
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Wrote JSON: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write JSON: {e}")
            return False

    @staticmethod
    def safe_read_json(file_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Read failed: {e}")
            return None

    @staticmethod
    def safe_append_text(file_path: str, text: str) -> bool:
        try:
            import os
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'a', encoding='utf-8') as f:
                f.write("\n" + text + "\n")

            logger.info(f"Appended text to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Append failed: {e}")
            return False


class TextProcessor:
    """Light text cleanup for model output."""

    @staticmethod
    def clean_ai_text(text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text

    @staticmethod
    def extract_json_from_text(text: str) -> Optional[str]:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return None


__all__ = [
    'JSONParser',
    'PromptBuilder',
    'FileHelper',
    'TextProcessor'
]
