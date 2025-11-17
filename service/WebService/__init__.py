from src.component.typing import *
from src.service.WebService.SearchService import SearchFactory

class WebService:
    
    def __init__(self) -> None:
        self.search_engine = SearchFactory().get_search()
    
    def search_web(self, query: str, *args,**kwargs) -> searchResultType:
        result = self.search_engine.search_web(query, *args,**kwargs)
        return result