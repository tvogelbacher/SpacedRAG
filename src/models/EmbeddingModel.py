from src.spaced_utils import get_embedding

class EmbeddingModel:
    def __init__(self, identity, tokenizer, model, c_model, get_emb):
        self.identity = identity
        self.tokenizer = tokenizer
        self.model = model
        self.c_model = c_model
        self.get_emb = get_emb

    def embed(self, text):
        return get_embedding(text, self.tokenizer, self.c_model, self.get_emb, self.identity)