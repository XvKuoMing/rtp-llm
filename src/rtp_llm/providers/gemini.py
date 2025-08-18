from .base import BaseSTTProvider, Message

from google import genai
from google.genai import types
from typing import Optional, Any, List, AsyncGenerator, Dict, Set

class GeminiProvider(BaseSTTProvider):
    """
    it provides only stt capabilities
    """

    def __init__(self, 
                 api_key: str, 
                 model: str, 
                 base_url: Optional[str] = None, 
                 system_prompt: Optional[str] = None,
                 gen_config: Optional[Dict[str, Any]] = None,
                 ):
        super().__init__(system_prompt, gen_config)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        if self.base_url is None:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = genai.Client(
                api_key=self.api_key, 
                http_options=types.HttpOptions(base_url=self.base_url)
            )
            
    def get_stt_gen_config_info(self) -> Set[str]:
        """
        Get the stt_config info of the provider -> name and default value
        """
        return {"temperature", "top_p", "top_k", "web_search"}
        
    
    def __transform_gen_config(self, gen_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        _gen_config = gen_config or self.stt_gen_config
        web_search = _gen_config.pop("web_search", False)
        build_in_tools = []
        if web_search:
            build_in_tools.append(types.Tool(google_search=types.GoogleSearch()))
        
        if build_in_tools:
            if "tools" in _gen_config:
                if not isinstance(_gen_config["tools"], list):
                    raise ValueError("tools must be a list")
                _gen_config["tools"].extend(build_in_tools)
            else:
                _gen_config["tools"] = build_in_tools
        
        return _gen_config

    
    async def format(self, message: Message) -> Any:
        # Map roles to Gemini accepted values
        role = message.role
        if role == "assistant":
            role = "model"

        if message.data_type == "text":
            return types.Content(role=role, parts=[types.Part(text=message.content)])
        elif message.data_type == "audio":
            return types.Content(role=role,
                                 parts=[types.Part.from_bytes(data=message.content, mime_type="audio/wav")])
        else:
            raise ValueError(f"Unsupported data type: {message.data_type}")
        
    
    async def stt(self, formatted_data: List[types.Content], *, system_prompt: Optional[str] = None, gen_config: Optional[Dict[str, Any]] = None) -> str:
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=formatted_data,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt or self.system_prompt,
                **self.__transform_gen_config(gen_config)
            ),
        )
        if isinstance(response, types.GenerateContentResponse):
            return response.text
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")
    

    async def stt_stream(self, formatted_data: List[types.Content], *, system_prompt: Optional[str] = None, gen_config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        async for chunk in self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=formatted_data,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt or self.system_prompt,
                **self.__transform_gen_config(gen_config)
            ),
        ):
            if isinstance(chunk, types.GenerateContentResponse):
                yield chunk.text
            else:
                raise ValueError(f"Unsupported response type: {type(chunk)}")






