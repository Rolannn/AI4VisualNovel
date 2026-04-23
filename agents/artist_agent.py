import os
import logging
import base64
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from PIL import Image, ImageOps
from rembg import remove, new_session

from .config import APIConfig, ArtistConfig, PathConfig

logger = logging.getLogger(__name__)


class ArtistAgent:
    """Generates character sprites (OpenAI Images or Google Imagen)."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = APIConfig.IMAGE_PROVIDER.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.config = ArtistConfig
        self.client = None
        self.available = False

        self._initialize_client()

    def _initialize_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            self.api_key = self.api_key or APIConfig.OPENAI_API_KEY
            self.base_url = self.base_url or APIConfig.OPENAI_BASE_URL

            if not self.api_key:
                logger.warning("OpenAI API key not set; image generation disabled")
            else:
                try:
                    self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                    self.available = True
                    logger.info("Artist agent initialized (OpenAI)")
                except Exception as e:
                    logger.error(f"Artist agent init failed: {e}")

        elif self.provider == "google":
            try:
                from google import genai
                self.api_key = self.api_key or APIConfig.GOOGLE_API_KEY
                self.base_url = self.base_url or APIConfig.GOOGLE_BASE_URL

                if not self.api_key:
                    logger.warning("Google API key not set; image generation disabled")
                else:
                    client_kwargs = {"api_key": self.api_key}
                    if self.base_url:
                        client_kwargs["http_options"] = {"base_url": self.base_url}

                    self.client = genai.Client(**client_kwargs)
                    self.available = True
                    logger.info("Artist agent initialized (Google Imagen)")
            except ImportError:
                logger.error("google-genai is not installed")
            except Exception as e:
                logger.error(f"Artist agent init failed: {e}")
        else:
            logger.error(f"Unsupported image provider: {self.provider}")

    def generate_character_images(
        self,
        character: Dict[str, Any],
        expressions: Optional[List[str]] = None,
        feedback: Optional[str] = None,
        reference_image_paths: Optional[List[str]] = None,
        story_background: Optional[str] = None,
        art_style: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate one sprite per expression for a character.

        Returns:
            Map expression name -> file path
        """
        character_name = character.get('name', 'unknown')
        character_id = character.get('id', character_name)

        expressions = expressions or self.config.STANDARD_EXPRESSIONS

        logger.info(f"Generating sprites for [{character_name}], expressions: {expressions}")
        if feedback:
            logger.info(f"   With critique feedback: {feedback}")
        if art_style:
            logger.info(f"   Art style: {art_style}")

        character_dir = os.path.join(PathConfig.CHARACTERS_DIR, character_id)
        os.makedirs(character_dir, exist_ok=True)

        image_paths = {}

        if not reference_image_paths:
            neutral_path = os.path.join(character_dir, "neutral.png")
            if os.path.exists(neutral_path):
                reference_image_paths = [neutral_path]
                logger.info(f"   Auto-loaded reference: {neutral_path}")

        sorted_expressions = sorted(expressions, key=lambda x: 0 if x == 'neutral' else 1)

        for expression in sorted_expressions:
            try:
                filename = f"{expression}.png"
                expected_image_path = os.path.join(character_dir, filename)

                if os.path.exists(expected_image_path) and not feedback:
                    logger.info(f"   [{expression}] already exists, skip")
                    image_paths[expression] = expected_image_path
                    if expression == 'neutral':
                        reference_image_paths = [expected_image_path]
                    continue

                if feedback and os.path.exists(expected_image_path):
                    logger.info(f"   [{expression}] exists but feedback present, regenerating...")

                image_path = self._generate_single_image(
                    character=character,
                    expression=expression,
                    output_dir=character_dir,
                    reference_image_paths=reference_image_paths,
                    feedback=feedback,
                    story_background=story_background,
                    art_style=art_style
                )

                if image_path:
                    image_paths[expression] = image_path
                    logger.info(f"   [{expression}] sprite OK")
                    if expression == 'neutral':
                        reference_image_paths = [image_path]
                else:
                    logger.warning(f"   [{expression}] sprite failed")

            except Exception as e:
                logger.error(f"   [{expression}] error: {e}")

        logger.info(f"Finished [{character_name}]: {len(image_paths)} image(s)")

        return image_paths

    def _build_prompt(
        self,
        character: Dict[str, Any],
        expression_type: str,
        description: Optional[str] = None,
        feedback: Optional[str] = None,
        story_background: Optional[str] = None,
        art_style: Optional[str] = None
    ) -> str:
        """Build the image generation prompt."""
        appearance = character.get('appearance', 'anime style character')
        personality = character.get('personality', 'Unknown')

        if expression_type == "custom" and description:
            base_prompt = self.config.IMAGE_PROMPT_TEMPLATE.format(
                story_background=story_background or "A visual novel game",
                art_style=art_style or "Japanese anime style",
                appearance=appearance,
                personality=personality,
                expression=description
            )
        else:
            base_prompt = self.config.IMAGE_PROMPT_TEMPLATE.format(
                story_background=story_background or "A visual novel game",
                art_style=art_style or "Japanese anime style",
                appearance=appearance,
                personality=personality,
                expression=expression_type
            )

        if feedback:
            base_prompt += f"\n\nIMPORTANT CORRECTIONS FROM CHARACTER REVIEW:\n{feedback}\n\nSTRICT: Keep the same face, hair, clothes, and style while applying the fixes."

        return base_prompt

    def _call_image_api(self, prompt: str, reference_image_paths: Optional[List[str]] = None) -> Optional[bytes]:
        if self.provider == "openai":
            try:
                short_prompt = prompt[:1000]
                model_name = APIConfig.IMAGE_MODEL

                if reference_image_paths and any(os.path.exists(p) for p in reference_image_paths):
                    ref_path = next(p for p in reference_image_paths if os.path.exists(p))
                    with open(ref_path, "rb") as f:
                        img_bytes = f.read()

                    logger.info(f"Calling OpenAI images.edit (model: {model_name})")
                    response = self.client.images.edit(
                        model=model_name,
                        image=("reference.png", img_bytes, "image/png"),
                        prompt=short_prompt,
                        n=1,
                        size=self.config.IMAGE_SIZE
                    )
                else:
                    logger.info(f"Calling OpenAI images.generate (model: {model_name})")
                    response = self.client.images.generate(
                        model=model_name,
                        prompt=short_prompt,
                        n=1,
                        size=self.config.IMAGE_SIZE,
                        response_format="b64_json"
                    )

                data = response.data[0]
                if hasattr(data, 'b64_json') and data.b64_json:
                    return base64.b64decode(data.b64_json)
                elif hasattr(data, 'url') and data.url:
                    import requests
                    return requests.get(data.url).content

                return None

            except Exception as e:
                logger.error(f"OpenAI image API failed: {e}")
                return None

        elif self.provider == "google":
            contents = [prompt]
            if reference_image_paths:
                for path in reference_image_paths:
                    if not path or not os.path.exists(path):
                        continue
                    try:
                        ref_img = Image.open(path)
                        contents.append(ref_img)
                    except Exception as e:
                        logger.warning(f"   Could not load reference [{path}]: {e}")

                if len(contents) > 1:
                    contents[0] = f"Generate a variation of the character in the attached images, maintaining visual consistency: {prompt}"

            response = self.client.models.generate_content(
                model=APIConfig.IMAGE_MODEL,
                contents=contents
            )

            if hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        return part.inline_data.data
                    try:
                        img = part.as_image()
                        import io
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        return buf.getvalue()
                    except Exception:
                        pass
            return None
        return None

    def _save_image(self, image_data: bytes, filepath: Path) -> None:
        with open(filepath, 'wb') as f:
            f.write(image_data)
        logger.info(f"   Saved image: {filepath}")

    def _remove_background(self, filepath: Path) -> None:
        try:
            logger.info(f"   Removing background: {filepath.name} (isnet-anime)...")
            input_image = Image.open(filepath)

            session = new_session("isnet-anime")

            output_image = remove(
                input_image,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=200,
                alpha_matting_background_threshold=20,
                alpha_matting_erode_size=10,
                alpha_matting_base_size=0
            )

            output_image.save(filepath)
            logger.info(f"   Background removed")
        except Exception as e:
            logger.error(f"   Background removal failed: {e}")

    def _generate_single_image(
        self,
        character: Dict[str, Any],
        expression: str,
        output_dir: str,
        reference_image_paths: Optional[List[str]] = None,
        feedback: Optional[str] = None,
        story_background: Optional[str] = None,
        art_style: Optional[str] = None
    ) -> Optional[str]:
        if not self.available:
            return None

        try:
            name = character.get('name', 'Character')
            prompt = self._build_prompt(
                character,
                expression,
                feedback=feedback,
                story_background=story_background,
                art_style=art_style
            )

            logger.info(f"   Generating [{name}] / {expression}...")

            image_data = self._call_image_api(prompt, reference_image_paths)

            if image_data:
                filename = f"{expression}.png"
                filepath = Path(output_dir) / filename
                self._save_image(image_data, filepath)

                self._remove_background(filepath)

                return str(filepath)
            return None

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None

    def generate_background(
        self,
        location: str,
        time_of_day: str = "",
        atmosphere: str = "peaceful",
        story_background: str = "",
        art_style: str = ""
    ) -> Optional[str]:
        """Generate a single background CG."""
        logger.info(f"Background CG: {location}")

        import re
        safe_location = re.sub(r'[^\w\s-]', '', location).strip().replace(' ', '_')

        if time_of_day:
            filename = f"{safe_location}_{time_of_day}.png"
        else:
            filename = f"{safe_location}.png"

        file_path = os.path.join(PathConfig.BACKGROUNDS_DIR, filename)

        if os.path.exists(file_path):
            logger.info(f"   Background exists, skip: {file_path}")
            return file_path

        if not self.available or not self.client:
            logger.warning("Image generation not available")
            return None

        try:
            prompt = self.config.BACKGROUND_PROMPT_TEMPLATE.format(
                location=location,
                time_of_day=time_of_day,
                atmosphere=atmosphere,
                story_background=story_background,
                art_style=art_style
            )

            logger.info(f"   Rendering: {location}...")
            logger.debug(f"   Prompt: {prompt[:150]}...")

            image_data = self._call_image_api(prompt)

            if image_data:
                self._save_image(image_data, Path(file_path))
                return file_path
            return None

        except Exception as e:
            logger.error(f"Background failed: {e}")
            return None

    def generate_all_backgrounds(self, locations: List[str], story_background: str = "", art_style: str = "") -> Dict[str, str]:
        """Generate backgrounds for every location name in the list."""
        logger.info(f"Generating {len(locations)} background(s)")

        background_images = {}

        for i, location in enumerate(locations, 1):
            logger.info(f"\n[{i}/{len(locations)}] {location}")

            try:
                bg_path = self.generate_background(
                    location=location,
                    time_of_day="",
                    atmosphere="peaceful",
                    story_background=story_background,
                    art_style=art_style
                )
                if bg_path:
                    background_images[location] = bg_path
            except Exception as e:
                logger.error(f"Location {location} failed: {e}")

        logger.info(f"\nBackgrounds done: {len(background_images)}/{len(locations)}")

        return background_images

    def generate_title_image(self, title: str, background_desc: str, character_images: List[str] = None) -> Optional[str]:
        """Generate the title screen key art."""
        logger.info(f"Title screen: {title}")

        filename = "title_screen.png"
        file_path = os.path.join(PathConfig.IMAGES_DIR, filename)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"   Removed old title art for refresh...")
            except Exception:
                pass

        if not self.available or not self.client:
            logger.warning("Image generation not available")
            return None

        try:
            prompt = self.config.TITLE_IMAGE_PROMPT_TEMPLATE.format(
                title=title,
                background=background_desc
            )

            logger.info(f"   Generating title art...")

            image_data = self._call_image_api(prompt, reference_image_paths=character_images)

            if image_data:
                self._save_image(image_data, Path(file_path))
                return file_path
            return None

        except Exception as e:
            logger.error(f"Title image failed: {e}")
            return None
