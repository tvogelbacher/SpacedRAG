from google import genai
from google.genai import types
import os

from .Model import Model

class Gemini(Model):
    def __init__(self, config):
        super().__init__(config)
        
        api_key = os.environ.get('GEMINI_API_KEY')
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = genai.Client(api_key=api_key)

    def query(self, msg):
        try:
            response = self.client.models.generate_content(
                model=self.name,
                contents=msg,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens)
            )
            return response.text
        except Exception as e:
            print(e)
            return ""