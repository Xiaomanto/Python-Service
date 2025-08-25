from typing_extensions import override
from openai import OpenAI
from src.component.typing import BaseChatService
from openai.types.chat import ChatCompletion,ChatCompletionMessageParam,ChatCompletionToolParam
from openai._types import NotGiven

class OpenaiService(BaseChatService):
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(api_key=self.api_key, base_url=self.host)

    @override
    def chat(self, prompt: list[ChatCompletionMessageParam], tools: list[ChatCompletionToolParam] | None = None) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            tools=tools or NotGiven()
        )