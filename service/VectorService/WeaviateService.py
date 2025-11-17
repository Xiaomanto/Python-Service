from typing import Literal
from uuid import UUID
from typing_extensions import override
from weaviate.collections.classes.internal import Object
from weaviate.collections.classes.types import WeaviateProperties
from weaviate.collections.classes.config_vectors import _VectorConfigCreate
from src.component.typing import BaseVectorService, Document
import weaviate as wc


class WeaviateService(BaseVectorService):
    
    def __init__(self) -> None:
        super().__init__()
        self.collections = [self.table_database_name, self.image_database_name, self.label_database_name]
        self.backup_data = None
        if self.is_need_recreate:
            print("start to recreate database")
            self._backup_data()
            for collection in self.collections:
                self.delete_database(collection)
        for collection in self.collections:
            self.create_database(collection, exist_ok=True)
        if self.backup_data:
            for collection in self.collections:
                for item in self.backup_data[collection]:
                    self.insert(item, collection)
        self.config["vector_config_type"] = self.model_type
        self.config["vector_config_model"] = self.model
        self._save_config(self.config)
        print("WeaviateServiceImpl initialized.")
        
    def _get_vectorizer(self) -> _VectorConfigCreate:
        if self.model_type == "openai":
            return wc.classes.config.Configure.Vectors.text2vec_openai(
                model=self.model,
                base_url=self.baseUrl
            )
        if self.model_type == "huggingface":
            return wc.classes.config.Configure.Vectors.text2vec_huggingface(
                model=self.model,
                endpoint_url=self.baseUrl
            )
        return wc.classes.config.Configure.Vectors.text2vec_ollama(
            model=self.model,
            api_endpoint=self.baseUrl
        )
        
    def _parse_data(self, data: dict):
        try:
            pageNumber = str(data.get("docPage")) or "0"
            return {
                "name": data.get("labelName") or data.get("imageName") or data.get("tableName") or "No Name",
                "content": data.get("content") or "No Content",
                "PageNumber": int(pageNumber if pageNumber.isdigit() else "0"),
                "docId": data.get("docId")
            }
        except Exception as e:
            print(f"Failed to parse data: {e}")
            raise e
        
    def _parse_result(self, result: Object[WeaviateProperties, None]) -> Document:
        try:
            return Document(
                content=result.properties.get("content"), 
                metadata={
                    "name": result.properties.get("name"), 
                    "PageNumber": result.properties.get("PageNumber"), 
                    "docId": result.properties.get("docId")
                }
            )
        except Exception as e:
            print(f"Failed to parse result: {e}")
            raise e
        
    def _backup_data(self):
        try:
            with self.connect() as conn:
                data = {}
                for collection in self.collections:
                    coll = conn.collections.get(collection)
                    data[collection] = []
                    try:
                        for item in coll.iterator():
                            data[collection].append(item.properties)
                    except Exception as e:
                        print(f"Failed to get data from collection: {e}")
                        print(f"skip this collection {collection} no data found")
                        continue
                print("backup data completed.")
                self.backup_data = data
        except Exception as e:
            print(f"Failed to backup data: {e}")
            raise e
        
    @override
    def connect(self):
        try:
            return wc.connect_to_local(host=self.host, port=self.port or 8080, headers=self.headers)
        except Exception as e:
            print(f"Failed to connect to Weaviate: {e}")
            raise e
        
    @override
    def create_database(self, name: str, exist_ok: bool=False):
        try:
            with self.connect() as conn:
                conn.collections.create(
                    name, 
                    vector_config=self._get_vectorizer(),
                    properties=[
                        wc.classes.config.Property(name="name", data_type=wc.classes.config.DataType.TEXT),
                        wc.classes.config.Property(name="content", data_type=wc.classes.config.DataType.TEXT),
                        wc.classes.config.Property(name="PageNumber", data_type=wc.classes.config.DataType.INT),
                        wc.classes.config.Property(name="docId", data_type=wc.classes.config.DataType.UUID)
                    ]
                )
        except Exception as e:
            print(f"Failed to create database: {e}")
            if exist_ok:
                return
            raise e
        
    @override
    def list_databases(self) -> list[str]:
        try:
            with self.connect() as conn:
                return [item.name for item in conn.collections.list_all(simple=True).values()]
        except Exception as e:
            print(f"Failed to list databases: {e}")
            raise e
        
    @override
    def delete_database(self, name: str):
        try:
            with self.connect() as conn:
                conn.collections.delete(name)
            print(f"database {name} deleted")
        except Exception as e:
            print(f"Failed to delete database: {e}")
            raise e
        
    @override
    def insert(self, data: dict, collection_name: str) -> UUID:
        try:
            with self.connect() as conn:
                collection = conn.collections.get(collection_name)
                uid = collection.data.insert(self._parse_data(data))
                return uid
        except Exception as e:
            print(f"Failed to insert data: {e}")
            raise e
        
    @override
    def search_knowledge(self, query: str, collection_name: str, mode: Literal["bm25", "similarity", "multi"]="multi", limit: int=3) -> list[Document]:
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
            with self.connect() as conn:
                # 尝试获取集合，如果不存在则列出可用的集合
                try:
                    collection = conn.collections.get(collection_name)
                except Exception as collection_error:
                    # 如果集合不存在，列出所有可用的集合
                    available_collections = self.list_databases()
                    error_msg = f"找不到集合 '{collection_name}'。"
                    if available_collections:
                        error_msg += f" 可用的集合有：{', '.join(available_collections)}。"
                        error_msg += " 请使用正确的集合名称重试。"
                    else:
                        error_msg += " 目前没有任何可用的集合。"
                    print(f"Collection error: {error_msg}")
                    raise ValueError(error_msg) from collection_error
                
                if mode == "bm25":
                    bm25 = collection.query.bm25(query, limit=limit).objects
                    return [self._parse_result(result) for result in bm25]
                if mode == "similarity":
                    similar = collection.query.near_text(query, limit=limit).objects
                    return [self._parse_result(result) for result in similar]
                if mode == "multi":
                    bm25_results = self.search_knowledge(query, collection_name, "bm25", limit=limit)
                    similar_results = self.search_knowledge(query, collection_name, "similarity", limit=limit)
                    return list[Document](bm25_results + similar_results)
        except ValueError:
            # 重新抛出 ValueError（集合不存在）
            raise
        except Exception as e:
            print(f"Failed to search data: {e}")
            raise e
    
    @override
    def update(self, data: dict, collection_name: str, uid: UUID):
        try:
            with self.connect() as conn:
                collection = conn.collections.get(collection_name)
                collection.data.update(uuid=uid, properties=self._parse_data(data))
        except Exception as e:
            print(f"Failed to update data: {e}")
            raise e
        
    @override
    def delete(self, collection_name: str, uid: UUID):
        try:
            with self.connect() as conn:
                collection = conn.collections.get(collection_name)
                collection.data.delete_by_id(uid)
        except Exception as e:
            print(f"Failed to delete data: {e}")
            raise e