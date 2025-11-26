from typing import Literal
from uuid import UUID
from typing_extensions import override
from src.component.typing.vectorbase import BaseVectorService, Document
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction, OpenAIEmbeddingFunction, HuggingFaceEmbeddingFunction

class ChromadbService(BaseVectorService):
    
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
                coll = self.client.get_collection(collection)
                data[collection] = []
                try:
                    obj = coll.get()
                    for key in obj.keys():
                        obj[key] = [obj[key]]
                    parsed_data = self._parse_result(obj)
                    data[collection] = parsed_data
                except Exception as e:
                    print(f"Failed to get data from collection: {e}")
                    print(f"skip this collection {collection} no data found")
                    continue
            print("backup data completed.")
            self.backup_data = data
        except Exception as e:
            print(f"Failed to backup data: {e}")
            raise e
    
    def _get_embedding_function(self):
        if self.model_type == "openai":
            return OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name=self.model or "text-embedding-3-small",
                api_base=self.baseUrl
            )
        if self.model_type == "huggingface":
            print(f"using huggingface embedding function: {self.model}")
            return HuggingFaceEmbeddingFunction(
                api_key=self.api_key,
                model_name=self.model or "sentence-transformers/all-MiniLM-L6-v2"
            )
        if self.model_type == "ollama":
            return OllamaEmbeddingFunction(
                model=self.model,
                url=self.baseUrl or "http://localhost:11434"
            )
        return OllamaEmbeddingFunction(
            model=self.model, 
            url=self.baseUrl or "http://localhost:11434"
        )
    
    def _parse_result(self, result: dict) -> list[Document]:
        docs = []
        print(result)
        for i in range(len(result.get("ids",[]))):
            for j in range(len(result.get("ids")[i] or [])):
                docId = result.get("metadatas")[i][j].get("docId")
                result.get("metadatas")[i][j].pop("docId")
                docs.append(Document(
                    docId=UUID(docId),
                    pageId=UUID(result.get("ids")[i][j]),
                    content=result.get("documents")[i][j],
                    metadata=result.get("metadatas")[i][j]
                ))
        
    @override
    def connect(self):
        self.client  = chromadb.HttpClient(host=self.host or "localhost", port=self.port or 8000, headers=self.headers)
    
    @override
    def create_collection(self, name: str, exist_ok: bool=False):
        try:
            if exist_ok:
                self.client.get_or_create_collection(
                    name=name,
                    embedding_function=self._get_embedding_function()
                )
                return
            self.client.create_collection(
                name=name,
                embedding_function=self._get_embedding_function()
            )   
        except Exception as e:
            print(f"Failed to create collection: {e}")
            raise e
        
    @override
    def delete_collection(self, name: str):
        self.client.delete_collection(name)
        
    @override
    def list_collections(self) -> list[str]:
        return [item.name for item in self.client.list_collections()]
    
    @override
    def insert(self, data: Document, collection_name: str):
        metadata = {
            **data.metadata,
            "docId": data.docId.hex
        }
        self.client.get_collection(collection_name).upsert(
            documents=[data.content],
            metadatas=[metadata],
            ids=[data.pageId.hex]
        )
    
    @override
    def delete(self, collection_name: str, uid: UUID):
        self.client.get_collection(collection_name).delete(ids=[uid.hex])
    
    @override
    def update(self, data: Document, collection_name: str):
        """
        Use insert to update data
        """
        pass
    
    @override
    def search_knowledge(self, query: str, collection_name: str, mode: Literal["bm25", "similarity", "multi"] = "multi", limit: int = 3) -> list[Document]:
        """
        Search the knowledge base
        Args:
            query: The query to search for
            collection_name: The name of the collection to search in
            mode: The mode to use for the search, "bm25" for BM25, "similarity" for similarity search, "multi" for both
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
            
            if mode == "bm25":
                bm25 = collection.query(n_results=limit , query_texts=[query], where_document={"$contains":query})
                bm25.pop("included")
                return self._parse_result(bm25)
            if mode == "similarity":
                similar = collection.query(n_results=limit , query_texts=[query])
                return [self._parse_result(result) for result in similar]
            if mode == "multi":
                bm25_results = self.search_knowledge(query, collection_name, "bm25", limit=limit)
                return bm25_results
        except ValueError:
            # 重新抛出 ValueError（集合不存在）
            raise
        except Exception as e:
            print(f"Failed to search data: {e}")
            raise e