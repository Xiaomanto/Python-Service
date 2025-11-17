from src.service.ChatService import llamaFactory
from src.service.WebService import WebService
from src.service.VectorService import VectorFactory
from typing import Literal
class Service:

    def get_service(self, name:Literal['chat','web','vector','vision']='chat'):
        if name == 'chat':
            return llamaFactory().get_llm()
        if name == 'web':
            return WebService()
        if name == 'vector':
            return VectorFactory().get_vector()
        if name == 'vision':
            return llamaFactory().get_vlm()