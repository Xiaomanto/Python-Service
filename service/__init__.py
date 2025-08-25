from src.service.ChatService import llamaFactory
from typing import Literal
class Service:

    def get_service(self, name:Literal['chat']='chat'):
        if name == 'chat':
            return llamaFactory().get_llm()