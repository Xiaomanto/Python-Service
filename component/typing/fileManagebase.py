from abc import ABC, abstractmethod
import base64
from io import BytesIO
import os
from typing import Union
from PIL.Image import Image,open as open_image

class BaseFileManageService(ABC):
    
    def __init__(self):
        self.soffice_path = os.getenv("SOFFICE_PATH")
        if not self.soffice_path:
            raise ValueError("SOFFICE_PATH is not set, please set it in the environment variables.")
        self.images:list[Image] = []
    
    @abstractmethod
    def file_to_image(self,input_file: str) -> None:
        pass

    def get_images(self) -> list[Image]:
        try:
            return self.images
        except Exception as e:
            print(f"Failed to get images: {e}")
            return []
        finally:
            self.images = []
    
    def get_base64_list(self) -> list[str]:
        if not self.images:
            raise ValueError("Images Not Found Error: Please use file_to_image first!!!")
        image_b64_list = []
        for idx, img in enumerate(self.images):
            b64 = self.parse_Image_to_base64(img)
            image_b64_list.append(b64)
        return image_b64_list

    def parse_Image_to_base64(self, image:Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    
    def parse_base64_to_Image(self, b64:str) -> Image:
        return open_image(BytesIO(base64.decodebytes(b64.encode("utf-8"))))

    @abstractmethod
    def _convert_pdf_to_image(self,pdf: Union[bytes, str]) -> list[Image]:
        pass

    @abstractmethod
    def _convert_office_to_image(self,office: Union[bytes, str]) -> list[Image]:
        pass