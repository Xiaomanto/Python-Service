from typing import Literal
from uuid import UUID
from typing_extensions import override
from src.component.typing.vectorbase import BaseVectorService, Document
from qdrant_client import QdrantClient, models
from qdrant_client.models import  PointStruct, VectorParams, Distance

from src.component.utils import Encoder

class QdrantService(BaseVectorService):
    
    def __init__(self) -> None:
        super().__init__()
        self.connect()
        self.collections = [self.table_database_name, self.image_database_name, self.label_database_name]
        self.backup_data = None
        if self.is_need_recreate:
            print("start to recreate database")
            self._backup_data()
            for collection in self.collections:
                self.delete_collection(collection)
        for collection in self.collections:
            self.create_collection(collection, exist_ok=True)
        if self.backup_data:
            for collection in self.collections:
                for item in self.backup_data[collection]:
                    self.insert(item, collection)
        self.config["chromadb"]["vector_config_type"] = self.model_type
        self.config["chromadb"]["vector_config_model"] = self.model
        self._save_config(self.config)
        print("ChromadbServiceImpl initialized.")
    
    def _backup_data(self):
        try:
            data = {}
            for collection in self.collections:
                data[collection] = []
                offset = 0
                try:
                    while offset is not None:
                        obj, offset = self.client.scroll(collection_name=collection, limit=1000, offset=offset)
                        data[collection] += [self._parse_result(o.payload) for o in obj if o.payload is not None]
                except Exception as e:
                    print(f"Failed to get data from collection: {e}")
                    print(f"skip this collection {collection} no data found")
                    continue
            print("backup data completed.")
            self.backup_data = data
        except Exception as e:
            print(f"Failed to backup data: {e}")
            raise e
    
    def _get_encoder(self) -> Encoder:
        if self.model_type == "openai":
            return Encoder(
                types=self.model_type,
                api_key=self.api_key,
                model_name=self.model or "text-embedding-3-small",
                api_base=self.baseUrl
            )
        if self.model_type == "huggingface":
            print(f"using huggingface embedding function: {self.model}")
            return Encoder(
                types=self.model_type,
                api_key=self.api_key,
                model=self.model or "sentence-transformers/all-MiniLM-L6-v2"
            )
        if self.model_type == "ollama":
            return Encoder(
                types=self.model_type,
                model=self.model,
                api_base=self.baseUrl or "http://localhost:11434"
            )
        return Encoder(
            types=self.model_type,
            model=self.model,
            api_base=self.baseUrl or "http://localhost:11434"
        )
    
    def _parse_result(self, result: dict) -> list[Document]:
        docs = []
        docId = result.pop("docId")
        pageId = result.pop("pageId")
        content = result.pop("content")
        return Document(
            docId=UUID(docId),
            pageId=UUID(pageId),
            content=str(content),
            metadata=result
        )
    @override
    def connect(self):
        self.client  = QdrantClient(host=self.host or "localhost", port=self.port or 6333)
    
    @override
    def create_collection(self, name: str, exist_ok: bool=False):
        encoder = self._get_encoder()
        try:
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=encoder.get_sentence_embedding_dimension(), 
                        distance=Distance.COSINE
                    ),
                )
                return
            raise ValueError(f"Collection {name} already exists")
        except Exception as e:
            print(f"Failed to create collection: {e}")
            if exist_ok:
                return
            raise e
        
    @override
    def delete_collection(self, name: str):
        self.client.delete_collection(name)
        
    @override
    def list_collections(self) -> list[str]:
        return self.client.get_collections().collections
    
    @override
    def insert(self, data: Document | list[Document], collection_name: str):
        if isinstance(data, Document):
            data = [data]
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=d.pageId,
                    vector=self._get_encoder().encode(d.content),
                    payload={
                        **d.metadata,
                        "docId": d.docId.hex,
                        "pageId": d.pageId.hex,
                        "content": d.content
                    }
                )
                for d in data
            ],
        )
        
    @override
    def delete(self, collection_name: str, uid: UUID):
        self.client.delete(
            collection_name,
            points_selector=models.PointIdsList(
                points=[uid]
            )
        )
    
    @override
    def update(self, data: Document, collection_name: str):
        """
        Use insert to update data
        """
        self.insert(data, collection_name)
    
    @override
    def search_knowledge(self, query: str, collection_name: str, mode: Literal["bm25", "similarity", "multi"] = "multi", limit: int = 3) -> list[Document]:
        """
        Search the knowledge base
        Args:
            query: The query to search for
            collection_name: The name of the collection to search in
            mode: this vector service only support similarity search
            limit: The limit of the search, default is 3, if used multi mode, the limit is the limit of each mode
        Returns:
            A list of documents
        """
        try:
            # 尝试获取集合，如果不存在则列出可用的集合
            try:
                collection = self.client.get_collection(collection_name)
            except Exception as collection_error:
                # 如果集合不存在，列出所有可用的集合
                available_collections = self.list_collections()
                error_msg = f"找不到集合 '{collection_name}'。"
                if available_collections:
                    error_msg += f" 可用的集合有：{', '.join(available_collections)}。"
                    error_msg += " 请使用正确的集合名称重试。"
                else:
                    error_msg += " 目前没有任何可用的集合。"
                print(f"Collection error: {error_msg}")
                raise ValueError(error_msg) from collection_error
            qembed = self._get_encoder().encode(query)
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=qembed,
                query_filter=None,
                limit=limit
            ).points
            return self._parse_result(search_result)
        except ValueError:
            # 重新抛出 ValueError（集合不存在）
            raise
        except Exception as e:
            print(f"Failed to search data: {e}")
            raise e