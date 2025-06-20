from .base import BaseSTTProvider, Message

from google import genai
from google.genai import types
from typing import Optional, Any, List, AsyncGenerator, Dict




class GeminiSTTProvider(BaseSTTProvider):
    """
    it provides only stt capabilities
    """
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, system_prompt: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant."
        if self.base_url is None:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = genai.Client(
                api_key=self.api_key, 
                http_options=types.HttpOptions(base_url=self.base_url)
            )
        
    
    async def format(self, message: Message) -> Any:
        if message.data_type == "text":
            return types.Content(role=message.role, parts=[types.Part(text=message.content)])
        elif message.data_type == "audio":
            return types.Content(role=message.role, 
                                 parts=[types.Part.from_bytes(data=message.content, mime_type="audio/wav")])
        else:
            raise ValueError(f"Unsupported data type: {message.data_type}")
        
    async def stt(self, 
                  formatted_data: List[types.Content],
                  gen_config: Optional[Dict[str, Any]] = None) -> str:
        if gen_config is None:
            gen_config = {}
        # try:
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=formatted_data,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                **gen_config
            ),
        )
        # except Exception as e:
        #     with open("error.txt", "w") as f:
        #         f.write(str(formatted_data))
        #     raise e
        if isinstance(response, types.GenerateContentResponse):
            return response.text
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")
    

    async def stt_stream(self, 
                         formatted_data: List[types.Content],
                         gen_config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        if gen_config is None:
            gen_config = {}
        async for chunk in self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=formatted_data,
            config=types.GenerateContentConfig(
                **gen_config
            ),
        ):
            if isinstance(chunk, types.GenerateContentResponse):
                yield chunk.text
            else:
                raise ValueError(f"Unsupported response type: {type(chunk)}")






