import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
import os

from .Model import Model


class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        hf_token = os.environ.get('HF_ACCESS_TOKEN')

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, token=hf_token) #LlamaTokenizer.from_pretrained(self.name, token=hf_token)
        self.model = LlamaForCausalLM.from_pretrained(self.name, torch_dtype=torch.float16, token=hf_token).to(self.device)

    def query(self, msg):
        inputs = self.tokenizer(msg, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,  # Pass attention mask
            temperature=self.temperature,
            max_new_tokens=int(self.max_output_tokens),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_tokens = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return result