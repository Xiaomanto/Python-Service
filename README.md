# Python-Service

### 這是一個用來快速開發 Python LLM Service 的模組，他可以讓你輕鬆建立一個 LLM 對話、向量、SQL、Web Services等。

## 功能

- 快速建立 LLM 對話、向量、SQL、Web 相關服務等。
- 支援多種 LLM 模型，包含：Ollama、OpenAI、以及任何與 OpenAI API 相容的模型 API。
- 支援多種向量庫，包含：Weaviate 等。
- 支援多種數據庫，包含：PostgreSQL、MySQL 等。
- 支援多種 Web 使用，包含：Search、Crawl 等。

## 日誌

 - 2025-08-25: 初始版本，僅支援 LLM 快速開發。
 - 2025-08-28: 新增 Web 模組，僅支援 Google Search (SerpAPI)。

## 安裝

### 在你的專案資料夾中，執行以下指令：
```bash
git clone https://github.com/Xiaomanto/Python-Service.git src
```

### 在你的專案根目錄中，安裝環境，請執行它：
``` bash
uv init --bare
uv venv
source .venv/bin/activate
mv src/requirements.txt requirements-service.txt
uv add -r requirements-service.txt
```
## 使用

### 先在專案主程式或專案跟目錄中建立 config/.env 檔案，並填入以下內容：
``` env
########## LLM Config ##########
LLM_URL=http://xxx.xxx.xxx.xxx:12345/v1/
# example LLM_URL=http://xxx.xxx.xxx.xxx:12345/v1/

LLM_TYPE=Company
# example LLM_TYPE=OpenAI
# Options: OpenAI, Ollama

LLM_API_KEY=your-api-key
# example LLM_API_KEY=sk-proj-xxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

LLM_MODEL=ModelName 
# example LLM_MODEL=gpt-3.5-turbo

########## Search Config ##########
SEARCH_API_KEY=your-api-key
# only use serpapi api key
```

### 在你需要使用聊天或其他本模組支援之功能的 Python 檔中引入本模組：
``` python
from src.service import Service

chatService = Service().get_service('chat')

# example
response = chatService.chat({"role":"user","content":"你好"})

print(response)

webService = Service().get_service('web')
# example
searched = webService.search("NBA")

print(searched)
```

## 給使用者們的話

### 這是一個快速開發 LLM Service 的模組，希望對你開發 AI 應用上能有所幫助，你可以隨意擴充他以便你可以運用在任何的環境，如果使用上有任何問題，歡迎建立 Issues 詢問。
