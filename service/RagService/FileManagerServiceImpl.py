import os
import subprocess
import mimetypes
import tempfile
from typing import Literal, Union
from io import BytesIO
from typing_extensions import override
from PIL.Image import Image,open as open_image, new as new_image
from PIL import ImageFont, ImageDraw
from pdf2image import convert_from_bytes

from src.component.typing.fileManagebase import BaseFileManageService

class FileManageService(BaseFileManageService):
    
    def __init__(self):
        super().__init__()

    @override
    def file_to_image(self,input_file: str) -> None:
        ext = os.path.splitext(input_file)[1].lower()
        _ = mimetypes.guess_type(input_file)

        if ext in [".pdf"]:
            self.images = self._convert_pdf_to_image(input_file)
            return
        
        if ext in [".xls", ".xlsx"]:
            self.images = self._convert_office_to_image(input_file,types="calc")
            return
        
        if ext in [".doc", ".docx", ".odt", ".ods", ".odp",".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".yaml", ".yml"]:
            self.images = self._convert_office_to_image(input_file,types="writer")
            return
        
        if ext in [".ppt", ".pptx"]:
            self.images = self._convert_office_to_image(input_file,types="impress")
            return
        
        if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".ico", ".webp"]:
            self.images = [open_image(input_file)]
            return
        
        raise ValueError(f"不支援的格式: {ext}")

    @override
    def _convert_pdf_to_image(self, pdf: bytes | str) -> list[Image]:
        """將 PDF bytes 或圖片 bytes 轉換為 Image 列表"""
        try:
            # 嘗試作為 PDF 處理
            if isinstance(pdf, str):
                with open(pdf, "rb") as f:
                    pdf = f.read()
            images = convert_from_bytes(pdf)
            return images
        except Exception:
            try:
                # 如果 PDF 轉換失敗，嘗試作為圖片處理
                image = open_image(BytesIO(pdf))
                return [image]
            except Exception as e:
                print(f"無法將資料轉換為圖片: {e}")
                raise ValueError("無法處理的檔案格式")

    @override
    def _convert_office_to_image(self, office: Union[bytes, str], types: Literal["calc", "impress", "writer"]) -> list[Image]:
        """Word / PowerPoint → 圖片"""
        methods = {
            "calc":"calc_pdf_Export",
            "impress":"impress_pdf_Export",
            "writer":"writer_pdf_Export",
        }
        method = methods.get(types)
        if not method:
            raise ValueError(f"不支援的格式: {types}")
        with tempfile.TemporaryDirectory() as tmpdir:
            # 儲存檔案
            if isinstance(office, bytes):
                input_file = os.path.join(tmpdir, "input_file.docx")
                with open(input_file, "wb") as f:
                    f.write(office)
            else:
                input_file = office
            # 用 LibreOffice 轉 PDF
            subprocess.run([
                self.soffice_path, "--headless", "--norestore", "--convert-to", f"pdf:{method}",
                "--outdir", tmpdir, input_file
            ], check=True)

            print(f"saved to temp directory: {tmpdir}")
            print("temp path files: ", os.listdir(tmpdir))
            pdf_file = os.path.splitext(os.path.basename(input_file))[0] + ".pdf"
            pdf_path = os.path.join(tmpdir, pdf_file)
            # 轉成圖片
            images = self._convert_pdf_to_image(pdf_path)
            return images
