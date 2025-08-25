from src.service.ChatService.ollamaService import OllamaService
from src.service.ChatService.openaiService import OpenaiService
from src.component.typing import BaseChatService
from dotenv import load_dotenv
import os

load_dotenv('config/.env')

class llamaFactory:

    def get_llm(self) -> BaseChatService:
        types = os.getenv("LLM_TYPE",'openai').lower()
        if types == 'openai':
            return OpenaiService()
        elif types == 'ollama':
            return OllamaService()
        else:
            return OpenaiService()