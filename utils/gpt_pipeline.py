import os
import requests
import logging
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
    def analyze_about_section(self, about_text: str) -> Dict[str, Any]:
        result = self._call_openai_api(about_text)
        
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
    
    def _call_openai_api(self, about_text: str) -> Dict[str, Any]:
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
        
        GENDER: 'female' (if words like woman, girl, she/her pronouns are present or any indicators that they are a female), 'male' (if words like man, guy, he/him pronouns are present or any indicators that they are a male), or 'unknown' (if gender cannot be determined due to insufficient information)
        """
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": about_text}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return {
                "success": True,
                "analysis": response.json()["choices"][0]["message"]["content"],
                "raw_response": response.json()
            }
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Singleton Instance
gpt_analyzer = GPTAnalyzer() 