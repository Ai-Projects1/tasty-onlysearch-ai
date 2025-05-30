import os
import requests
from typing import Dict, Any, Optional, List

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

        prompt_template = """
You are a gender classification assistant. You are based on OnlySearch which is an OnlyFans search engine where user's can join and create their own profile

Your job is to determine the gender of the user based on the {about_text} given. Their {about_text} is their bio, explaining themselves and their preferences, marketing themselves on OnlySearch

Since we're linked to OnlyFans, most likely majority of the {about_text} will be female gender. But you should still account for the possibility of male gender based on their {about_text}

I will be giving a list of possible keywords that are most likely to be used by the user to describe themselves below:

    <female_possible_keywords>
    - Blowjob
    - Creampie
    - Watch you cum
    - Watch me cum my ass out
    - Watch me seduce you
    - Looking for BBC/Male here
    - Feet pics, panties, dickrates, pussy pics
    - Anything related to tits and camming
    - horny MILF here
    - Dildo blowjob videos
    </female_possible_keywords>

    <male_possible_keywords>
    - BBC Male here
    - Rate my dick
    - Cum ride this 🍆
    - I eat pussies
    </male_possible_keywords>


'''
{about_text}
'''

In some instance, the {about_text} may contain insufficient information for you to determine the gender of the user. In this case, you should respond with "unknown" to let our fallback model take over

Based on the text above, is the user male or female? Answer with only one word: "male", "female", or "unknown"
        """
        
        prompt = prompt_template.format(about_text=about_text)
        
        result = self._call_openai_api(prompt)
        
        if not result["success"]:
            return result
            
        analysis = result["analysis"].strip().lower()
        
        if "male" in analysis:
            gender = "male"
        elif "female" in analysis:
            gender = "female"
        else:
            gender = "unknown"
            
        return {
            "success": True,
            "gender": gender,
            "raw_response": result["raw_response"]
        }
    
    def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a gender classification assistant. You only respond with 'male', 'female', or 'unknown'."},
                {"role": "user", "content": prompt}
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
            return {
                "success": False,
                "error": str(e)
            }

# Singleton Instance
gpt_analyzer = GPTAnalyzer() 