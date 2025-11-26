from typing import Literal
from openai import OpenAI
from ollama import Client as Ollama
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available as cuda_available
from torch.mps import is_available as mps_available

class Encoder:
    
    def __init__(
        self, 
        types: Literal["openai", "huggingface", "ollama"], 
        model: str,
        api_key: str | None = None,
        api_base: str | None = None
    ) -> None:
        self.types = types
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self._initialize_model()
        
    def _set_device(self) -> None:
        if cuda_available():
            self.device = "cuda"
            return
        if mps_available():
            self.device = "mps"
            return
        self.device = "cpu"
        
    def _initialize_model(self) -> None:
        self._set_device()
        if self.types == "openai":
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        if self.types == "huggingface":
            self.client = SentenceTransformer(self.model, device=self.device)
        if self.types == "ollama":
            self.client = Ollama(host=self.api_base)
        
    
    def get_sentence_embedding_dimension(self) -> int:
        if self.types == "openai":
            return len(self.client.embeddings.create(input="Hello, world!", model=self.model).data[0].embedding)
        if self.types == "huggingface":
            return self.client.get_sentence_embedding_dimension() or 0
        if self.types == "ollama":
            return len(self.client.embed(self.model, "Hello, world!").embeddings[0])
        
    def encode(self, data: any) -> list[float]:
        data = str(data)
        if self.types == "openai":
            return self.client.embeddings.create(input=data, model=self.model).data[0].embedding
        if self.types == "huggingface":
            return self.client.encode(data).tolist()
        if self.types == "ollama":
            return self.client.embed(self.model, data).embeddings[0]