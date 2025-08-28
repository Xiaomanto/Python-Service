from src.component.typing import searchResultType
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os

load_dotenv('config/.env')

class BaseSearchService(ABC):
    
    def __init__(self) -> None:
        self.api_key = os.getenv("SEARCH_API_KEY")
    
    @abstractmethod
    def search(self, query: str, limit: int=10) -> searchResultType:
        pass