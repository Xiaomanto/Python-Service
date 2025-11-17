import inspect
import json
from typing import Any, Callable, Literal

from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolMessageParam, ChatCompletionToolParam
from openai.types.chat.chat_completion_tool_param import FunctionDefinition
from openai.types.shared_params.function_parameters import FunctionParameters

from src.service import Service


class ToolService:
    def __init__(self) -> None:
        self.tools = []
        self.tool_map = {}
        self.client = Service().get_service('chat')
        
    def _get_function_params(self, tool: Callable[..., Any], function_source: str) -> FunctionParameters:
        """
        生成函数参数的 JSON Schema 定义
        
        Args:
            tool: 要分析的工具函数
            function_source: 函数的源代码字符串
            
        Returns:
            FunctionParameters: OpenAI 函数参数定义
        """
        function_params = {"type": "object", "properties": {}, "required": []}
        
        try:
            param_info = self._extract_param_info(tool)
            self._build_param_definitions(function_params, param_info)
            self._enhance_descriptions(function_params, function_source, param_info)
        except Exception as e:
            print(f"生成函数参数时出错: {e}")
            
        return function_params
    
    def _extract_param_info(self, tool: Callable[..., Any]) -> list:
        """提取函数参数信息"""
        signature = inspect.signature(tool)
        param_info = []
        
        for param in signature.parameters.values():
            if param.name == 'self':
                continue
                
            param_type = param.annotation
            type_name = self._get_type_name(param_type)
            
            param_info.append({
                "name": param.name,
                "type": type_name,
                "annotation": param_type.__name__ if param_type is not inspect.Parameter.empty else "Any",
                "param_type": param_type  # 保存原始类型，用于处理 Literal
            })
            
        return param_info
    
    def _build_param_definitions(self, function_params: dict, param_info: list) -> None:
        """构建参数定义"""
        for param in param_info:
            param_name = param["name"]
            type_name = param["type"]
            annotation = param["annotation"]
            param_type = param.get("param_type")
            
            # 处理 Literal 类型
            if type_name == "__LITERAL__":
                literal_values = self._extract_literal_values(param_type)
                function_params["properties"][param_name] = {
                    "type": "string",
                    "enum": literal_values,
                    "description": f"参数 {param_name} ({annotation})"
                }
            else:
                function_params["properties"][param_name] = {
                    "type": type_name,
                    "description": f"参数 {param_name} ({annotation})"
                }
            function_params["required"].append(param_name)
    
    def _enhance_descriptions(self, function_params: dict, function_source: str, param_info: list) -> None:
        """增强参数描述"""
        if len(param_info) <= 1:
            return
            
        try:
            batch_description = self._generate_batch_param_descriptions(function_source, param_info)
            if batch_description:
                for param_name, description in batch_description.items():
                    if param_name in function_params["properties"]:
                        function_params["properties"][param_name]["description"] = description
        except Exception as e:
            print(f"批量生成参数描述失败，使用默认描述: {e}")
    
    def _get_type_name(self, param_type: Any) -> str:
        """
        获取参数类型的名称，转换为标准的 JSON Schema 类型
        
        Args:
            param_type: 参数类型注解
            
        Returns:
            str: JSON Schema 类型名称，特殊值 "__LITERAL__" 表示需要处理为 enum
        """
        # 处理 Literal 类型（枚举值）
        if hasattr(param_type, '__origin__') and param_type.__origin__ is Literal:
            return "__LITERAL__"
        
        if param_type is inspect.Parameter.empty or param_type is Any:
            return "string"  # 默认使用 string
    
        # Python 类型到 JSON Schema 类型的映射
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        
        # 处理泛型类型
        if hasattr(param_type, '__origin__'):
            origin = param_type.__origin__
            if origin in type_mapping:
                return type_mapping[origin]
            # 对于其他泛型类型，默认返回 string
            return "string"
        
        # 直接映射 Python 类型到 JSON Schema 类型
        if param_type in type_mapping:
            return type_mapping[param_type]
        
        # 默认返回 string
        return "string"
    
    def _extract_literal_values(self, param_type: Any) -> list:
        """从 Literal 类型中提取枚举值"""
        if param_type and hasattr(param_type, '__args__'):
            return [str(arg) for arg in param_type.__args__]
        return []
    
    def _generate_batch_param_descriptions(self, function_source: str, param_info: list) -> dict:
        """
        批量生成参数描述，减少 API 调用次数

        Args:
            function_source: 函数源代码
            param_info: 参数信息列表
            
        Returns:
            dict: 参数名称到描述的映射
        """
        try:
            param_list = ", ".join([f"{p['name']} ({p['annotation']})" for p in param_info])
            
            prompt = f"""
                請根據以下函數源碼，為每個參數生成簡潔的描述（每個描述不超過20個字）：
                規則：
                1. 描述必須要保留該參數的限制值的語言 如： mode: Literal["bm25", "similarity", "multi"]="multi" ，應保留 bm25, similarity, multi 三個值的原文，不可以擅自翻譯，避免模型判斷錯誤。
                
                源碼：
                {function_source}

                參數列表：{param_list}

                請以 JSON 格式返回，格式如下：
                {{
                    "參數名1": "描述1",
                    "參數名2": "描述2"
                }}
            """

            response = self.client.chat([
                {"role": "user", "content": prompt}
            ]).choices[0].message.content.replace("```json","").replace("```","")
            print(response)
            # 解析 JSON 響應
            import json
            result = json.loads(response)
            result.pop("kwargs", None)
            result.pop("args", None)
            return result
            
        except Exception as e:
            print(f"批量生成参数描述失败: {e}")
            return {}

    def use_tool(self, tool_calls: list[ChatCompletionMessageToolCall], **kwargs) -> list[ChatCompletionToolMessageParam]:
        results = []
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.function.name
                tool_params = json.loads(tool_call.function.arguments or "{}")
                tool_result = self.tool_map[tool_name](**tool_params)
                results.append(ChatCompletionToolMessageParam(
                    content=f"{tool_result}",
                    role="tool",
                    tool_call_id=tool_call.id
                ))
            except Exception as e:
                print(f"使用工具 {tool_name} 時出錯: {e}")
                results.append(ChatCompletionToolMessageParam(
                    content=f"工具 {tool_name} 使用時出錯請根據錯誤訊息修正後再試一次: {e}",
                    role="tool",
                    tool_call_id=tool_call.id
                ))
        return results
    
    def _clean_description(self, description: str) -> str:
        """清理描述文本，移除 markdown 代码块"""
        if not description:
            return ""
        # 移除 markdown 代码块标记
        description = description.replace("```json", "").replace("```", "").strip()
        # 尝试提取 JSON 中的 description 字段（如果有的话）
        try:
            json_obj = json.loads(description)
            if isinstance(json_obj, dict) and "description" in json_obj:
                return json_obj["description"]
            return description
        except (json.JSONDecodeError, ValueError):
            # 如果不是 JSON，直接返回清理后的文本
            return description
    
    def add_tool(self, tool: Callable[..., Any], **kwargs) -> None:
        """
        Registers a callable tool into the service and prepares its OpenAI tool description.
        Expects 'name', 'description', 'parameters' as kwargs (parameters is a JSON schema).
        """
        tool_name = kwargs.get('name', tool.__name__)
        function_source = inspect.getsource(tool)
        function_params = self._get_function_params(tool,function_source)
        self.tool_map[tool_name] = tool
        
        # 生成或获取描述
        raw_description = kwargs.get(
            'description', 
            self.client.chat(
                [
                    {
                        "role":"user",
                        "content":f"請幫我產生以下 Tool Function 源碼的 Tool Description（只返回描述文字，不要用 markdown 格式，不要用代碼塊）： Source_Code: {function_source}"
                    }
                ]
            ).choices[0].message.content
        )
        
        # 清理描述
        cleaned_description = self._clean_description(raw_description)
        
        tool_param = ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=tool_name,
                description=cleaned_description,
                parameters=function_params
            )
        )
        self.tools.append(tool_param)
    
    def list_tools(self) -> list[ChatCompletionToolParam]:
        """
        返回已注册的所有工具列表
        
        Returns:
            list[ChatCompletionToolParam]: 工具参数列表
        """
        return self.tools.copy()
