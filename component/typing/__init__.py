from typing import Literal
from pydantic import BaseModel
from abc import ABC, abstractmethod
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall
)
from openai.types.chat.chat_completion_message_tool_call import Function as FunctionParam
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage
from ollama._types import Message,Tool,ChatResponse
from dotenv import load_dotenv
from datetime import datetime
from uuid import uuid4
import json
import os

# Load environment variables
load_dotenv('config/.env')

# Response Type
ToolCall = Message.ToolCall
ToolCallFunction = Message.ToolCall.Function

# Request Type
Function = Tool.Function
Parameters = Tool.Function.Parameters

class llmConfig(BaseModel):
    model:str
    messages:list[Message] | list[dict]
    tools:list[ToolCall] | list[dict] | None = None

class BaseChatService(ABC):
    
    def __init__(self) -> None:
        super().__init__()
        self.host = os.getenv("LLM_URL")
        self.model = os.getenv("LLM_MODEL","llama3.3")
        self.api_key = os.getenv("LLM_API_KEY")

    def _parse_prompt(
            self, 
            prompt:list[ChatCompletionMessageParam], 
            tools:list[ChatCompletionToolParam] | None = None
    ) -> llmConfig:
        messages:list[Message] = []
        toolList:list[Tool] = []

        for message in prompt:
            content = ""
            images = []
            promptContent = message.get("content","")
            if isinstance(promptContent,list):
                for c in promptContent:
                    types = c.get("type","")
                    if types == "image_url":
                        image_url = c.get("image_url",{"url":""}).get("url","")
                        if image_url.startswith("http"):
                            print("image url is not base64 encoded or path skipping...")
                            continue
                        if image_url.startswith("data:image"):
                            images.append(image_url.split(",")[-1])
                        else:
                            images.append(image_url)
                    if types == "text" and not content:
                        content = c.get("text","")
                    if types == "input_audio":
                        print("audio not supported skipping...")
            else:
                content = promptContent or ""
            role = message.get("role","user")
            messages.append(Message(role=role, content=content, images=images or None))
        
        if tools is not None:
            toolList = self._parse_tool(tools)
        
        config = llmConfig(
            model=self.model,
            messages=messages,
            tools=tools
        )
        return config
    
    def _parse_tool(
            self, 
            tool_list:list[ChatCompletionToolParam]
    ) -> list[Tool]:
        tools = []
        for tool in tool_list:
            functions = tool.get("function")
            if functions is None:
                continue
            parameters = functions.get("parameters")
            if parameters is None:
                continue
            tools.append(
                Tool(
                    type=tool.get("type","function"),
                    function=Function(
                        name=functions.get("name"),
                        description=functions.get("description"),
                        parameters=Parameters(
                            type=parameters.get("type","object"),
                            properties=parameters.get("properties",{}),
                            required=parameters.get("required",[])
                        )
                    ) 
                )
            )
        return tools
    
    def _get_reason_state(
        self,
        message:Message
    ) -> Literal["stop", "length", "tool_calls", "content_filter", "function_call"]:
        state: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] = "stop"
        if message.tool_calls is not None:
            state = "tool_calls"
        return state
    
    def _parse_tool_calls(
        self,
        tool_calls: list[ToolCall]
    ) -> list[ChatCompletionMessageToolCall]:
        tools:list[ChatCompletionMessageToolCall] = []
        if tool_calls is not None:
            for tool in tool_calls:
                tools.append(
                    ChatCompletionMessageToolCall(
                        id=uuid4().hex,
                        function=FunctionParam(
                            arguments=str(tool.function.arguments),
                            name=tool.function.name
                        ),
                        type="function"
                    )
                )
        return tools

    def _parse_message(
        self,
        message:Message
    ) -> list[Choice]:
        choices:list[Choice] = []
        choices.append(
            Choice(
                finish_reason=self._get_reason_state(message),
                index=0,
                message=ChatCompletionMessage(
                    content=message.content,
                    role=message.role,
                    tool_calls=self._parse_tool_calls(message.tool_calls),
                )
            )
        )
        return choices

    def _parse_response(self, response:ChatResponse) -> ChatCompletion:
        return ChatCompletion(
            id=uuid4().hex,
            choices=self._parse_message(response.message),
            created=int(datetime.now().timestamp()),
            model=response.model,
            object='chat.completion',
            usage=CompletionUsage(
                prompt_tokens=response.prompt_eval_count or 0,
                completion_tokens=response.eval_count or 0,
                total_tokens=(response.prompt_eval_count or 0) + (response.eval_count or 0)
            )
        )

    @abstractmethod
    def chat(self, prompt:list[ChatCompletionMessageParam], tools:list[ChatCompletionToolParam] | None = None) -> ChatCompletion:
        pass

class InputDataObject(BaseModel):
    points:list[list[float]]
    polygonlabels:list[str]
    original_width:int
    original_height:int

class InputData(BaseModel):
    ocr:str
    id: int
    label: list[InputDataObject] | None = None
    transcription:str | list[str] | None = None
    shape:str
    scoreline:str
    annotator: str
    annotation_id:int
    created_at: str
    updated_at: str
    lead_time: float

    @classmethod
    def from_json(cls, data:dict | str):
        if isinstance(data, str):
            data = json.loads(data)
        transcription = data.get('transcription')
        if isinstance(transcription, list):
            temp = []
            for t in transcription:
                if isinstance(t, dict):
                    tlist = t.get('text')
                    if isinstance(tlist, list):
                        temp.extend(tlist)
                    if isinstance(tlist, str):
                        temp.append(tlist)
                if isinstance(t, str):
                    temp.append(t)
            data['transcription'] = temp   
        return cls(**data)
    
