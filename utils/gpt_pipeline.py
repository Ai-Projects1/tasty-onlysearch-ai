import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import base64
from io import BytesIO
from PIL import Image
import aiohttp
import asyncio
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_thread_local = threading.local()

class GPTAnalyzer:
    """
    A class to handle interactions with OpenAI's GPT-4o-mini model for analyzing text content.
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.model = "gpt-4o-mini"
        self.api_url = "https://api.openai.com/v1/chat/completions"

        
    async def get_session(self):
        """Get or create an aiohttp ClientSession using thread-local storage."""

        if not hasattr(_thread_local, 'session') or _thread_local.session is None or _thread_local.session.closed:
            logger.info("Creating new aiohttp session")
            _thread_local.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=10, ssl=False)
            )
        return _thread_local.session
        
    async def close_session(self):
        """Close the aiohttp session for the current thread."""
        if hasattr(_thread_local, 'session') and _thread_local.session and not _thread_local.session.closed:
            logger.info("Closing aiohttp session")
            await _thread_local.session.close()
            _thread_local.session = None
        
    async def analyze_about_section(self, about_text: str) -> Dict[str, Any]:
        try:
            result = await self._call_openai_api(about_text)
            
            if not result["success"]:
                return result
            
            response_content = result["analysis"].strip()
            
            try:
                parts = response_content.split("GENDER:")
                if len(parts) == 2:
                    thoughts = parts[0].strip()
                    gender_text = parts[1].strip().lower()
                else:
                    thoughts = response_content
                    gender_text = response_content.lower()
                
                logger.info(f"GPT Analysis Thoughts: {thoughts}\n")
                logger.info(f"Gender: {gender_text}")
                
                if "male" in gender_text and not "female" in gender_text:
                    gender = "male"
                elif "female" in gender_text:
                    gender = "female"
                else:
                    gender = "unknown"
                    
                return {
                    "success": True,
                    "gender": gender,
                    "thoughts": thoughts,
                    "raw_response": result["raw_response"]
                }
            except Exception as e:
                logger.error(f"Error processing GPT response: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error processing response: {str(e)}",
                    "raw_response": result["raw_response"]
                }
        except Exception as e:
            logger.error(f"Error in analyze_about_section: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            pass
    
    def _resize_image(self, img_data: bytes, max_size: Tuple[int, int] = (512, 512)) -> bytes:
        """
        Resize an image to reduce its dimensions while maintaining aspect ratio.
        
        Args:
            img_data: Raw image bytes
            max_size: Maximum width and height
            
        Returns:
            Resized image as bytes
        """
        try:
            img = Image.open(BytesIO(img_data))
            
            width, height = img.size
            
            if width > max_size[0] or height > max_size[1]:
                scale = min(max_size[0] / width, max_size[1] / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            else:
                logger.info(f"Image already smaller than max size: {width}x{height}")
            
            output = BytesIO()
            img.save(output, format=img.format or 'JPEG', quality=85)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return img_data
    
    async def analyze_image(self, image_url: str) -> Dict[str, Any]:
        """
        Analyzes an image using OpenAI's vision capabilities to determine gender.
        
        Args:
            image_url: URL of the image to analyze
            
        Returns:
            Dictionary containing gender analysis results
        """
        try:
            session = await self.get_session()
            
            logger.info(f"Starting image analysis for URL: {image_url}")
            
            try:
                image_timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(image_url, timeout=image_timeout) as response:
                    response.raise_for_status()
                    image_data = await response.read()
                    logger.info(f"Successfully downloaded image: {len(image_data)} bytes")
            except asyncio.TimeoutError:
                logger.error(f"Timeout downloading image from {image_url}")
                return {
                    "success": False,
                    "error": f"Timeout downloading image from {image_url}"
                }
            except Exception as e:
                logger.error(f"Error downloading image from {image_url}: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error downloading image: {str(e)}"
                }
            
            image_content = self._resize_image(image_data, max_size=(1920, 1080))
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            system_prompt = """
            Analyze the provided image

            we are not determining the gender, we are only here to state facts if the image shows male or female patterns
            
            First, explain your reasoning process by identifying visual cues and features that suggest gender

            Even if you cannot determine the gender, you should still provide your verdict from the provided image

            If the image contains explicitness, do not mind those and only use them as key references to determine the gender

            Then, provide your final determination
            
                Format your response as:
                [ANALYsis]
                
                GENDER: 'female' (if visual indicators suggest female), 'male' (if visual indicators suggest male), or 'unknown' (if gender cannot be determined from the image)
            """
            
            base64_image = base64.b64encode(image_content).decode('utf-8')
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "What is the gender of the person in this image?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            logger.info("Sending request to OpenAI API for image analysis")
            
            try:
                api_timeout = aiohttp.ClientTimeout(total=60)
                async with session.post(self.api_url, headers=headers, json=payload, timeout=api_timeout) as response:
                    response.raise_for_status()
                    response_data = await response.json()
                    logger.info("Received response from OpenAI API")
            except asyncio.TimeoutError:
                logger.error("Timeout calling OpenAI API for image analysis")
                return {
                    "success": False,
                    "error": "Timeout calling OpenAI API"
                }
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error calling OpenAI API: {str(e)}"
                }
            
            analysis_text = response_data["choices"][0]["message"]["content"].strip()
            
            logger.info(f"Vision Analysis: {analysis_text}")
            
            try:
                parts = analysis_text.split("GENDER:")
                if len(parts) == 2:
                    thoughts = parts[0].strip()
                    gender_text = parts[1].strip().lower()
                else:
                    thoughts = analysis_text
                    gender_text = analysis_text.lower()
                
                if "male" in gender_text and not "female" in gender_text:
                    gender = "male"
                elif "female" in gender_text:
                    gender = "female"
                else:
                    gender = "unknown"
                    
                return {
                    "success": True,
                    "gender": gender,
                    "thoughts": thoughts,
                    "confidence": 0.2, 
                    "predicted_by": "gpt-vision",
                    "raw_response": response_data
                }
            except Exception as e:
                logger.error(f"Error processing vision response: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error processing vision response: {str(e)}",
                    "raw_response": response_data
                }
                
        except Exception as e:
            logger.error(f"Vision API call error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # DO NOT CLOSE SESSION, WE ALLOW REUSE
            pass
    
    async def _call_openai_api(self, about_text: str) -> Dict[str, Any]:
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            system_prompt = """
            Analyze the provided profile text for gender indicators
            
            First, explain your reasoning process by identifying specific words, phrases, or context clues that suggest gender
            Then, provide your final determination
            
            Format your response as:
            [ANALysis]
            
            GENDER: 'female' (if words like woman, girl, she/her pronouns are present or any indicators that they are a female), 'male' (if words like man, guy, he/him pronouns are present or any indicators that 
            they are a male), or 'unknown' (if gender cannot be determined due to insufficient information)
            """
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": about_text}
                ],
                "temperature": 0.9,
                "max_tokens": 500
            }
            
            session = await self.get_session()
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.post(self.api_url, headers=headers, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                response_data = await response.json()
                
            return {
                "success": True,
                "analysis": response_data["choices"][0]["message"]["content"],
                "raw_response": response_data
            }
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:

            pass

# Singleton Instance
gpt_analyzer = GPTAnalyzer() 