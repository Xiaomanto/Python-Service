from typing import Literal
from src.service.WebService.SearchService.base import BaseSearchService
from src.service.WebService.SearchService.SerpSearchService import SerpSearchService

class SearchFactory:
    
    def __init__(self) -> None:
        pass
    
    def get_search(self, searchType: Literal['serp']='serp') -> BaseSearchService:
        if searchType == 'serp':
            return SerpSearchService()