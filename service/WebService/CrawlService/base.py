from abc import ABC, abstractmethod
from io import BufferedReader
import requests
import magic

class BaseCrawlService(ABC):
    
    def __init__(self) -> None:
        super().__init__()


    @abstractmethod
    def get_context(self):
        ...
    
    @abstractmethod
    def goto(self, url):
        ...

    @abstractmethod
    def click(self, element):
        ...

    @abstractmethod
    def get_links(self):
        ...

    @abstractmethod
    def get_text(self):
        ...

    def get_file_types(self, byte:bytes) -> str:
        return magic.from_buffer(byte, False).split(" ")[0].lower()

    def download(self,link:str) -> tuple[BufferedReader, str]:
        res = requests.get(link)
        res.raise_for_status()

        byte = res.content
        buffer = BufferedReader(byte)

        try:
            file_type = self.get_file_types(byte)
        except Exception as e:
            print(f"got an Error in :{e}")
            file_type = "unknown"

        return buffer, file_type