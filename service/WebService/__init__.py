from src.component.typing import *
from src.service.WebService.SearchService import SearchFactory

class WebService:
    
    def __init__(self) -> None:
        self.search_engine = SearchFactory().get_search()
    
    def search(self, query: str, **kwargs) -> searchResultType:
        result = self.search_engine.search(query, **kwargs)
        return result