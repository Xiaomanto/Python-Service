from typing import Literal
from typing_extensions import override
from ollama import Client
from openai.types.chat import ChatCompletion, ChatCompletionToolParam,ChatCompletionMessageParam
from src.component.typing import BaseChatService

class OllamaService(BaseChatService):

    def __init__(self, model:str = "llama3.3", host:str = None, api_key:str = None) -> None:
        super().__init__(model=model, host=host, api_key=api_key)
        self.client = Client(host=self.host or None)
        self.system_prompt = "你是一個有用的 AI 助手，請友善且準確地回答用戶的問題。優先以繁體中文回應。"

    @override
    def chat(self, prompt: list[ChatCompletionMessageParam], tools: list[ChatCompletionToolParam] | None = None) -> ChatCompletion:
        prompt.insert(0, {"role": "system", "content": self.system_prompt})
        config = self._parse_prompt(prompt, tools)
        response = self.client.chat(
            **config.model_dump(),
            stream=False
        )
        return self._parse_response(response)