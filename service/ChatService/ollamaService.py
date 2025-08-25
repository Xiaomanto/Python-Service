from typing_extensions import override
from ollama import Client
from openai.types.chat import ChatCompletion, ChatCompletionToolParam,ChatCompletionMessageParam
from src.component.typing import BaseChatService

class OllamaService(BaseChatService):

    def __init__(self) -> None:
        super().__init__()
        self.client = Client(host=self.host or None)

    @override
    def chat(self, prompt: list[ChatCompletionMessageParam], tools: list[ChatCompletionToolParam] | None = None) -> ChatCompletion:
        config = self._parse_prompt(prompt, tools)
        response = self.client.chat(
            **config.model_dump(),
            stream=False
        )
        return self._parse_response(response)