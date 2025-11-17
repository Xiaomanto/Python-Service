from dotenv import load_dotenv
import os

from src.component.typing.vectorbase import BaseVectorService
from src.service.VectorService.WeaviateService import WeaviateService

load_dotenv('config/.env')

class VectorFactory:

    def get_vector(self) -> BaseVectorService:
        types = os.getenv("VECTOR_TYPE",'weaviate').lower()
        if types == 'weaviate':
            return WeaviateService()
        else:
            return WeaviateService()