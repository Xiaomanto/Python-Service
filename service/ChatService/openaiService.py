from typing import Literal
from typing_extensions import override
from openai import OpenAI
from src.component.typing import BaseChatService
from openai.types.chat import ChatCompletion,ChatCompletionMessageParam,ChatCompletionToolParam
from openai._types import NotGiven

class OpenaiService(BaseChatService):
    def __init__(self, model:str = "gpt-3.5-turbo", host:str = None, api_key:str = None) -> None:
        super().__init__(model=model, host=host, api_key=api_key)
        self.client = OpenAI(api_key=self.api_key, base_url=self.host)

    def _get_tool_info(self, tool: ChatCompletionToolParam) -> tuple[str, str]:
        """获取工具的名称和描述"""
        if isinstance(tool, dict):
            func_info = tool.get('function', {})
            return func_info.get('name', 'unknown'), func_info.get('description', '')
        
        try:
            if hasattr(tool, 'get'):
                func_info = tool.get('function', {})
                return func_info.get('name', 'unknown'), func_info.get('description', '')
            elif hasattr(tool, 'function'):
                func = tool.function
                return getattr(func, 'name', 'unknown'), getattr(func, 'description', '')
        except (AttributeError, TypeError):
            pass
        
        return 'unknown', ''

    def _get_system_prompt(self, tools: list[ChatCompletionToolParam] | None = None) -> str:
        """生成系统提示，引导模型使用工具"""
        if not tools or len(tools) == 0:
            return "你是一個有用的 AI 助手，請友善且準確地回答用戶的問題。"
        
        # 获取工具名称列表
        tool_names = []
        for tool in tools:
            tool_name, _ = self._get_tool_info(tool)
            if tool_name != 'unknown':
                tool_names.append(tool_name)
        
        tools_list_str = "、".join(tool_names) if tool_names else "可用工具"
        
        # 强化的系统提示，明确要求使用工具
        system_prompt = f"""你是 AI 助手，可以使用工具获取资料。可用工具：{tools_list_str}

规则：
- 当用户问题需要查询、搜索、获取资料时，必须使用工具
- 先使用工具获取资料，再根据结果回答
- 不要仅凭知识回答，要主动使用工具

记住：需要实际资料时必须使用工具。"""
        
        return system_prompt

    def _has_system_message(self, messages: list[ChatCompletionMessageParam]) -> bool:
        """检查消息列表中是否已有系统消息"""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                return True
            if hasattr(msg, 'get') and msg.get("role") == "system":
                return True
        return False

    def _print_tool_info(self, tools: list[ChatCompletionToolParam]) -> None:
        """打印工具信息"""
        for i, tool in enumerate(tools, 1):
            tool_name, tool_desc = self._get_tool_info(tool)
            if tool_desc:
                print(f"Tool {i+1}: {tool_name} - {tool_desc[:50]}...")
            else:
                print(f"Tool {i+1}: {tool_name}")

    def _validate_and_clean_messages(self, messages: list[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
        """验证和清理消息列表"""
        cleaned = []
        for msg in messages:
            msg_dict = None
            
            # 尝试将消息转换为字典格式
            if isinstance(msg, dict):
                msg_dict = msg.copy()
            elif hasattr(msg, 'model_dump'):
                # Pydantic v2
                msg_dict = msg.model_dump()
            elif hasattr(msg, 'dict'):
                # Pydantic v1
                msg_dict = msg.dict()
            elif isinstance(msg, (list, tuple)):
                # 如果是列表/元组，尝试构建字典（某些 SDK 可能这样返回）
                continue
            else:
                # 尝试通过属性访问获取字段（TypedDict 或其他类型）
                try:
                    msg_dict = {
                        "role": getattr(msg, 'role', None),
                        "content": getattr(msg, 'content', None)
                    }
                except Exception:
                    continue
            
            if not isinstance(msg_dict, dict):
                continue
            
            # 确保必要的字段存在
            role = msg_dict.get("role")
            content = msg_dict.get("content")
            
            if not role or content is None:
                continue
            
            # 允许字符串和列表类型的 content（列表用于多模态消息，如文本+图片）
            if not isinstance(content, (str, list, tuple)):
                content = str(content)
            
            # 构建清理后的消息
            cleaned_msg = {
                "role": role,
                "content": content
            }
            
            cleaned.append(cleaned_msg)
        
        return cleaned

    def _ensure_tools_format(self, tools: list[ChatCompletionToolParam]) -> list[dict]:
        """确保工具格式正确，转换为字典格式以确保兼容性"""
        valid_tools = []
        for tool in tools:
            if tool is None:
                continue
            try:
                # 如果是字典，直接使用
                if isinstance(tool, dict):
                    # 验证字典结构
                    if 'type' in tool and 'function' in tool:
                        valid_tools.append(tool)
                        continue
                
                # 尝试转换为字典
                tool_dict = None
                
                # 方法1: 使用 model_dump (Pydantic v2)
                if hasattr(tool, 'model_dump'):
                    tool_dict = tool.model_dump()
                
                # 方法2: 使用 dict() (Pydantic v1)
                elif hasattr(tool, 'dict'):
                    tool_dict = tool.dict()
                
                # 方法3: 手动构建
                else:
                    tool_dict = {"type": "function", "function": {}}
                    if hasattr(tool, 'type'):
                        tool_dict['type'] = tool.type if isinstance(tool.type, str) else 'function'
                    
                    if hasattr(tool, 'function'):
                        func = tool.function
                        func_dict = {}
                        if hasattr(func, 'name'):
                            func_dict['name'] = func.name
                        if hasattr(func, 'description'):
                            func_dict['description'] = func.description
                        if hasattr(func, 'parameters'):
                            # parameters 应该是字典
                            params = func.parameters
                            if isinstance(params, dict):
                                func_dict['parameters'] = params
                            elif hasattr(params, 'model_dump'):
                                func_dict['parameters'] = params.model_dump()
                            elif hasattr(params, 'dict'):
                                func_dict['parameters'] = params.dict()
                        tool_dict['function'] = func_dict
                
                if tool_dict and 'type' in tool_dict and 'function' in tool_dict:
                    # 验证 function 字段
                    func_info = tool_dict['function']
                    if isinstance(func_info, dict) and 'name' in func_info and 'parameters' in func_info:
                        valid_tools.append(tool_dict)
                    else:
                        print(f"Warning: Invalid tool function structure: {func_info}")
                else:
                    print(f"Warning: Invalid tool structure: {tool_dict}")
                    
            except Exception as e:
                print(f"Warning: Failed to convert tool {type(tool)}: {e}")
                continue
        
        return valid_tools

    @override
    def chat(self, prompt: list[ChatCompletionMessageParam], tools: list[ChatCompletionToolParam] | None = None) -> ChatCompletion:
        print("chat with openai")
        print(f"tools count: {len(tools) if tools else 0}")
        
        try:
            # 验证和清理消息列表
            messages = self._validate_and_clean_messages(prompt)
            
            if not messages:
                raise ValueError("No valid messages in prompt")
            
            print(f"Messages after validation: {len(messages)}")
            
            # 准备参数
            kwargs = {
                "model": self.model,
                "messages": messages,
            }
            
            # 如果有工具，添加系统提示引导使用工具
            if tools and len(tools) > 0:
                self._print_tool_info(tools)
                
                # 确保工具格式正确，转换为字典格式
                valid_tools = self._ensure_tools_format(tools)
                
                if valid_tools:
                    # 调试：打印工具结构
                    if valid_tools:
                        print(f"Tool structure sample: type={valid_tools[0].get('type')}, has_function={('function' in valid_tools[0])}")
                        if 'function' in valid_tools[0]:
                            func = valid_tools[0]['function']
                            print(f"Function: name={func.get('name')}, has_params={('parameters' in func)}")
                    
                    # 如果没有系统消息，添加简化的系统提示
                    if not self._has_system_message(messages):
                        system_prompt = self._get_system_prompt(tools)
                        messages.insert(0, {"role": "system", "content": system_prompt})
                        kwargs["messages"] = messages  # 更新消息列表
                        print("Added system prompt to guide tool usage")
                    
                    # 使用转换后的字典格式工具
                    kwargs["tools"] = valid_tools
                    # 注意：不设置 tool_choice，让模型自己决定，某些 API 可能不支持此参数
                    print(f"Passing {len(valid_tools)} tool(s) to OpenAI API")
                else:
                    print("Warning: No valid tools to pass, continuing without tools")
            
            # 调试：打印最终参数（不包含敏感信息）
            print(f"API call params: model={kwargs.get('model')}, messages_count={len(kwargs.get('messages', []))}, tools_count={len(kwargs.get('tools', []))}")
            
            # 尝试调用 API，如果失败则重试或降级
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    return self.client.chat.completions.create(**kwargs)
                except Exception as api_error:
                    error_type = type(api_error).__name__
                    # 如果是 500 错误且是最后一次尝试，尝试不带工具调用
                    if attempt == max_retries - 1 and 'InternalServerError' in error_type and 'tools' in kwargs:
                        print("API call failed with tools, attempting without tools as fallback...")
                        # 创建不带工具的副本
                        fallback_kwargs = {k: v for k, v in kwargs.items() if k != 'tools'}
                        try:
                            return self.client.chat.completions.create(**fallback_kwargs)
                        except Exception as fallback_error:
                            print(f"Fallback without tools also failed: {fallback_error}")
                            raise api_error  # 抛出原始错误
                    elif attempt < max_retries - 1:
                        print(f"API call failed (attempt {attempt + 1}/{max_retries}): {api_error}, retrying...")
                        import time
                        time.sleep(0.5)  # 短暂等待后重试
                        continue
                    else:
                        raise
            
        except Exception as e:
            import traceback
            print(f"Error in chat: {e}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            if 'messages' in locals():
                print(f"Messages count: {len(messages)}")
                if messages:
                    print(f"First message: {messages[0]}")
            if tools:
                print(f"Tools count: {len(tools)}")
                if tools:
                    print(f"First tool type: {type(tools[0])}")
            raise