from dotenv import load_dotenv
import os

from src.component.typing.vectorbase import BaseVectorService
from src.service.VectorService.WeaviateService import WeaviateService
from src.service.VectorService.QdrantService import QdrantService
from src.service.VectorService.ChromadbService import ChromadbService

load_dotenv('config/.env')

class VectorFactory:

    def get_vector(self) -> BaseVectorService:
        types = os.getenv("VECTOR_TYPE",'weaviate').lower()
        if types == 'weaviate':
            return WeaviateService()
        elif types == 'qdrant':
            return QdrantService()
        elif types == 'chromadb':
            return ChromadbService()
        else:
            return WeaviateService()