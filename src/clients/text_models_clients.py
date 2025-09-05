import os
from typing import Optional
import requests

from dotenv import load_dotenv
load_dotenv()

class DeepSeekAPIClient:
    def __init__(self, 
            api_key: str = os.getenv("DEEPSEEK_API"), 
            base_url: str = "https://api.deepseek.com/v1/chat/completions",
            model: str = "deepseek-chat"):
        
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def generate_text(self, prompt: str, max_tokens: int) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional YouTube scriptwriter. Generate engaging, coherent, and concise video script content containing only the spoken dialogue for the narrator. Exclude any non-spoken elements such as scene descriptions, music cues, or meta-commentary about the script itself."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Error parsing API response: {e}")
            return None
