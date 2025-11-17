from pydantic import BaseModel
from .llmbase import *
from .vectorbase import *
from .fileManagebase import *
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config/.env')

class SerpApiConfig(BaseModel):
    q:str
    api_key:str
    hl:str = 'zh-tw'
    gl:str = 'tw'
    num: int = 10
    location: str = 'Taiwan'

class SerpResult(BaseModel):
    title:str
    link:str
    snippet:str
    position:int

searchResultType = list[SerpResult]