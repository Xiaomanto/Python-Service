from abc import ABC
import json
import os
from typing import Literal
from pydantic import BaseModel


class Document(BaseModel):
    content:str
    metadata: dict

class BaseVectorService(ABC):
    def __init__(self) -> None:
        self.host = os.getenv("VECTOR_HOST")
        self.api_key = os.getenv("VECTOR_API_KEY")
        self.port = os.getenv("VECTOR_PORT")
        self.model = os.getenv("VECTOR_MODEL")
        self.baseUrl = os.getenv("VECTOR_MODEL_BASE_URL")
        self.model_type = os.getenv("VECTOR_MODEL_TYPE","ollama").lower()
        self.config_path = os.getenv("CONFIG_PATH","config/config.json")
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump({}, f)
        self.headers = self._get_headers()
        self.is_need_recreate = self._check_config_state()
        self.table_database_name = "TableCollection"
        self.image_database_name = "ImageCollection"
        self.label_database_name = "LabelCollection"
    
    def _save_config(self, data: dict):
        with open(self.config_path, "w") as f:
            json.dump(data, f)
    
    def _check_config_state(self) -> bool:
        if not os.path.exists(self.config_path):
            return True
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
            return (
                self.config.get("vector_config_type") != self.model_type
                or self.config.get("vector_config_model") != self.model
            )
    
    def _get_headers(self) -> dict | None:
        if self.model_type == "openai":
            return {
                "X-OpenAI-Api-Key": self.api_key
            }
        if self.model_type == "huggingface":
            return {
                "X-HuggingFace-Api-Key": self.api_key
            }
        return None
    
    def connect(self):
        """
        Connect to the vector collection
        """
        pass
    
    def insert(self, data: dict, collection_name: str):
        """
        Insert data into the vector collection
        """
        pass
    
    def search_knowledge(self, query: str, collection_name: str, mode: Literal["bm25", "similarity", "multi"]="multi", limit: int=3) -> list[Document]:
        """
        Search the vector collection
        Args:
            query: The query to search for
            collection_name: The name of the collection to search in
            mode: The mode to use for the search, "bm25" for BM25, "similarity" for similarity search, "multi" for both
            limit: The limit of the search, default is 3, if used multi mode, the limit is the limit of each mode
        Returns:
            A list of documents
        can be used database names are [TableCollection, ImageCollection, LabelCollection]
        """
        pass
    
    def update(self, data: dict):
        """
        Update data in the vector collection
        """
        pass
    
    def delete(self, uid: str):
        """
        Delete data from the vector collection by uuid
        """
        pass
    
    def create_collection(self, name: str):
        """
        Create a new collection
        """
        pass
    
    def delete_collection(self, name: str):
        """
        Delete a collection
        """
        pass
    
    def list_collections(self) -> list[str]:
        """
        List all collections
        """
        pass