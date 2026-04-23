"""
LLM Client
~~~~~~~~~~
Unified LLM client for OpenAI and Google Gemini.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Union
from .config import APIConfig

logger = logging.getLogger(__name__)

class LLMClient:
    """Unified LLM client wrapper."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = APIConfig.TEXT_PROVIDER.lower()
        self.api_key = api_key
        self.base_url = base_url

        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Create the provider-specific client."""
        if self.provider == "openai":
            from openai import OpenAI
            self.api_key = self.api_key or APIConfig.OPENAI_API_KEY
            self.base_url = self.base_url or APIConfig.OPENAI_BASE_URL

            if not self.api_key:
                logger.warning("OpenAI API key is not set")
            else:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        elif self.provider == "google":
            try:
                from google import genai
                self.api_key = self.api_key or APIConfig.GOOGLE_API_KEY
                self.base_url = self.base_url or APIConfig.GOOGLE_BASE_URL

                if not self.api_key:
                    logger.warning("Google API key is not set")
                else:
                    client_kwargs = {"api_key": self.api_key}
                    if self.base_url:
                        client_kwargs["http_options"] = {"base_url": self.base_url}
                        logger.info(f"Google client initialized (endpoint: {self.base_url})")

                    self.client = genai.Client(**client_kwargs)

            except ImportError:
                logger.error("google-genai is not installed; run: pip install google-genai")

        else:
            logger.error(f"Unknown LLM provider: {self.provider}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        json_mode: bool = False,
        max_retries: int = 5
    ) -> str:
        """
        Chat completion with retries.

        Args:
            messages: OpenAI-style messages
            temperature: sampling temperature
            json_mode: request JSON output when supported
            max_retries: max attempts

        Returns:
            Assistant text content
        """
        import time

        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    return self._chat_openai(messages, temperature, json_mode)
                elif self.provider == "google":
                    return self._chat_google(messages, temperature, json_mode)
                else:
                    raise ValueError(f"Unsupported LLM provider: {self.provider}")
            except Exception as e:
                err_str = str(e)
                is_rate_limit = any(code in err_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                if attempt < max_retries - 1:
                    wait_time = (10 * (2 ** attempt)) if is_rate_limit else (2 ** attempt)
                    logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    raise

    def _chat_openai(self, messages: List[Dict[str, Any]], temperature: float, json_mode: bool) -> str:
        if not self.client:
            raise ValueError("OpenAI client is not initialized")

        import base64
        import mimetypes

        processed_messages = []
        for msg in messages:
            new_msg = msg.copy()
            if isinstance(msg.get("content"), list):
                new_content = []
                for item in msg["content"]:
                    if item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if os.path.exists(url):
                            mime_type, _ = mimetypes.guess_type(url)
                            if not mime_type:
                                mime_type = "image/png"

                            try:
                                with open(url, "rb") as image_file:
                                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                                    new_item = item.copy()
                                    new_item["image_url"] = {
                                        "url": f"data:{mime_type};base64,{encoded_string}"
                                    }
                                    new_content.append(new_item)
                            except Exception as e:
                                logger.error(f"Failed to read image: {e}")
                                new_content.append(item)
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
                new_msg["content"] = new_content
            processed_messages.append(new_msg)

        response_format = {"type": "json_object"} if json_mode else None

        try:
            response = self.client.chat.completions.create(
                model=APIConfig.MODEL,
                messages=processed_messages,
                temperature=temperature,
                response_format=response_format,
                timeout=90
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _chat_google(self, messages: List[Dict[str, Any]], temperature: float, json_mode: bool) -> str:
        if not self.client:
            raise ValueError("Google client is not initialized")

        from google.genai import types

        system_instruction = None
        contents = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                parts = []
                if isinstance(content, str):
                    parts.append(types.Part.from_text(text=content))
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(text=item["text"]))
                        elif item.get("type") == "image_url":
                            image_path = item["image_url"]["url"]
                            try:
                                if os.path.exists(image_path):
                                    with open(image_path, "rb") as f:
                                        image_data = f.read()
                                    parts.append(types.Part.from_bytes(data=image_data, mime_type="image/png"))
                                else:
                                    logger.warning(f"Google client: network image URLs are not supported: {image_path}")
                            except Exception as e:
                                logger.error(f"Failed to read image: {e}")

                contents.append(types.Content(role="user", parts=parts))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction
        )

        if json_mode:
            config.response_mime_type = "application/json"

        import concurrent.futures
        API_TIMEOUT = 240

        def _call():
            return self.client.models.generate_content(
                model=APIConfig.MODEL,
                contents=contents,
                config=config
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call)
                try:
                    response = future.result(timeout=API_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError(
                        f"Google Gemini API did not respond within {API_TIMEOUT}s"
                    )
            return response.text

        except Exception as e:
            logger.error(f"Google Gemini API call failed: {e}")
            raise
