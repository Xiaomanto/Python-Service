from src.service.ChatService import llamaFactory
from src.service.WebService import WebService
from typing import Literal

class Service:

    def get_service(self, name:Literal['chat','web']='chat'):
        if name == 'chat':
            return llamaFactory().get_llm()
        if name == 'web':
            return WebService()