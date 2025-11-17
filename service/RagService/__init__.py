from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
import json
from uuid import uuid4

from PIL.Image import Image
from src.service import Service
from src.service.RagService.FileManagerServiceImpl import FileManageService


class RagService:
    def __init__(self) -> None:
        self.file_manager = FileManageService()
        self.vector_service = Service().get_service('vector')
        self.vlm_template = """
            你是一個擅長從一張圖片中分類出裡面包含圖片、表格、文字三大類並提供區域座標的助手，使用者會提供圖片，你的任務是抓出該三大類的座標，並回傳一個 JSON。
            
            規範：
            1. 不可以輸出多餘的文字符號等。
            2. JSON 文字的 " 不可以加 \\ 。
            3. 輸出的內容必須正確。
            4. 輸出的 JSON 欄位必須符合下方範例格式及規定。
            5. 輸出的表格內容必須是 markdown 表格格式，不要輸出其他文字。
            6. 若表格過於複雜，可以適當拆解表格，拆解後的表格需要保留其在原始表格中的結構。
            7. 輸出前請自行再三確認輸出的內容是否符合 JSON 格式規範。

            範例如下：
            {
                "tables":[
                    {
                        "tableName":string, # if document is not have this table name, you can generate by yourself.
                        "docPage":int, # document page number, start with 0.
                        "content":string # markdown table content , don't input table description or other text, just the table content.
                        "xy":array # (x1, y1, x2, y2) => (left, top, right, bottom).
                    },
                    ... or more.
                ],
                "images":[
                    {
                        "imageName":string, # if document is not have this image name, you can generate by yourself.
                        "docPage":int, # document page number, start with 0.
                        "content":string, # image description or content texts.
                        "xy":array # (x1, y1, x2, y2) => (left, top, right, bottom).
                    },
                    ... or more.
                ],
                "labels":[
                    {
                        "labelName":string, # label title or other, you can generate by yourself.
                        "docPage":int, # document page number, start with 0.
                        "content":string # markdown text.
                        "xy":array # (x1, y1, x2, y2) => (left, top, right, bottom).
                    },
                    ... or more (A paragraph is divided into an object).
                ]
            }
        """

    def invoke(self, file_path:str) -> tuple[str,list[dict]]:
        """
        這個用於調用整個 RAG 流程，回傳 doc_id UUID 和 圖片 list[Image]。
        """
        doc_id = uuid4().hex
        self.file_manager.file_to_image(file_path)
        images = self.file_manager.get_images()
        self.insert_images(doc_id, images)
        return doc_id, images
        
    def process_image(self, image:Image | str, idx:int, retried:int = 0) -> dict:
        if retried > 3:
            return {}
        client = Service().get_service('vision')
        if isinstance(image, Image):
            image_base64 = self.file_manager.parse_Image_to_base64(image)
        else:
            image_base64 = image
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.vlm_template + f"\n現在是第 {idx+1} 頁"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }]
        try:
            response = client.chat(message)
            image_content = response.choices[0].message.content.replace("```json","").replace("```","").replace("\\\"","\"")
            data = json.loads(image_content)
            return data
        except Exception as e:
            print(f"Failed to get image content: {str(e)[:300]}")
            # print(f"vlm output: {image_content or ''}")
            print("skipping page...")
            return self.process_image(image, idx, retried + 1)
    
    def insert_images(self, doc_id:str, images:list[Image | str]):
        objects:list[dict] = []
        with ThreadPoolExecutor(max_workers=3) as executer:
            futures = [executer.submit(self.process_image, image, idx) for idx, image in enumerate(images)]
            for result in as_completed(futures):
                objects.append(result.result())
        for obj in objects:
            for table in obj.get("tables", []):
                table["docId"] = doc_id
                self.vector_service.insert(table, self.vector_service.table_database_name)
            for image in obj.get("images", []):
                image["docId"] = doc_id
                self.vector_service.insert(image, self.vector_service.image_database_name)
            for label in obj.get("labels", []):
                label["docId"] = doc_id
                self.vector_service.insert(label, self.vector_service.label_database_name)
        print(f"inserted {len(objects)} objects to vector database")