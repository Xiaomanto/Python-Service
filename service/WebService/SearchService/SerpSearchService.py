from serpapi.google_search import GoogleSearch
from src.component.typing import SerpApiConfig,SerpResult
from src.service.WebService.SearchService.base import BaseSearchService
from typing_extensions import override

class SerpSearchService(BaseSearchService):
    def __init__(self):
        super().__init__()
    
    @override
    def search_web(self, query: str, limit: int = 10, *args, **kwargs) -> list[SerpResult]:
        params = SerpApiConfig(
            q=query,
            num=limit,
            api_key=self.api_key
        )

        search = GoogleSearch(params.model_dump())
        results = search.get_dict()
        result = []
        
        for r in results['organic_results']:
            result.append(SerpResult(
                title=r['title'],
                link=r['link'],
                snippet=r['snippet'],
                position=r['position']
            ))

        return result
    