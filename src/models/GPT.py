from openai import OpenAI
from .Model import Model
import time
import os


class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        api_key = os.environ.get('OPENAI_API_KEY')
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = OpenAI(api_key=api_key)

    def query(self, msg):
        retries = 0
        while retries < 5:
            try:
                completion = self.client.chat.completions.create(
                    model=self.name,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": msg}
                    ],
                )
                response = completion.choices[0].message.content
                break  # Exit loop on success
            except Exception as e:
                print(e)
                retries += 1
                print(f"Retrying... ({retries}/5)")
                time.sleep(20)
                continue
        else:
            print("All retries failed. Returning empty response.")
            response = ""
        return response