from src.service.ChatService.ollamaService import OllamaService
from src.service.ChatService.openaiService import OpenaiService
from src.component.typing import BaseChatService
from dotenv import load_dotenv
import os

load_dotenv('config/.env')

class llamaFactory:
    
    def __init__(self) -> None:
        self.llm_type = os.getenv("LLM_TYPE",'openai').lower()
        self.vlm_type = os.getenv("VLM_TYPE",'openai').lower()
        self.llm_model = os.getenv("LLM_MODEL",'gpt-3.5-turbo')
        self.vlm_model = os.getenv("VLM_MODEL",'llava')
        self.llm_host = os.getenv("LLM_URL")
        self.vlm_host = os.getenv("VLM_URL")
        self.llm_api_key = os.getenv("LLM_API_KEY")
        self.vlm_api_key = os.getenv("VLM_API_KEY")

    def get_llm(self) -> BaseChatService:
        if self.llm_type == 'openai':
            return OpenaiService(model=self.llm_model, host=self.llm_host, api_key=self.llm_api_key)
        elif self.llm_type == 'ollama':
            return OllamaService(model=self.llm_model, host=self.llm_host, api_key=self.llm_api_key)
        else:
            return OpenaiService(model=self.llm_model, host=self.llm_host, api_key=self.llm_api_key)
        
    def get_vlm(self) -> BaseChatService:
        if self.vlm_type == 'openai':
            return OpenaiService(model=self.vlm_model, host=self.vlm_host, api_key=self.vlm_api_key)
        elif self.vlm_type == 'ollama':
            return OllamaService(model=self.vlm_model, host=self.vlm_host, api_key=self.vlm_api_key)
        else:
            return OpenaiService(model=self.vlm_model, host=self.vlm_host, api_key=self.vlm_api_key)