
from .base import BaseTTSProvider, BaseSTTProvider, Message

from openai import AsyncOpenAI
from openai.types.audio import Transcription
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from typing import AsyncGenerator, Any, Optional, Union, List, Dict
import base64


class OpenAIMessage(Message):
    def __init__(self, role: str, content: str | bytes, data_type: str):
        assert data_type in ["text", "audio"]
        super().__init__(role, content, data_type)
    

    def as_json(self) -> Any:
        if self.data_type == "text":
            return {"role": self.role, "content": self.content}
        elif self.data_type == "audio":
            base64_audio = base64.b64encode(self.content).decode("utf-8")
            return {"role": self.role, "content": [{"type": "input_audio", "input_audio": {"data": base64_audio, "format": "wav"}}]}
        else:
            raise ValueError(f"Invalid data type: {self.data_type}")

class TextOpenAIMessage(OpenAIMessage):
    def __init__(self, role: str, content: str):
        super().__init__(role, content, "text")

class AudioOpenAIMessage(OpenAIMessage):
    def __init__(self, role: str, content: bytes):
        super().__init__(role, content, "audio")





class OpenAIProvider(BaseTTSProvider, BaseSTTProvider):

    def __init__(self, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, 
                 stt_model: Optional[str] = "gpt-4o-mini-audio-preview",
                 tts_model: Optional[str] = "gpt-4o-mini-tts",
                 system_prompt: Optional[str] = None,
                 overwrite_stt_model_api_key: Optional[str] = None,
                 overwrite_stt_model_base_url: Optional[str] = None,
                 overwrite_tts_model_api_key: Optional[str] = None,
                 overwrite_tts_model_base_url: Optional[str] = None,
                 ):
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.__system_prompt = system_prompt or "You are a helpful assistant."

        self.stt_api_key = overwrite_stt_model_api_key or api_key
        self.stt_base_url = overwrite_stt_model_base_url or base_url
        self.tts_api_key = overwrite_tts_model_api_key or api_key
        self.tts_base_url = overwrite_tts_model_base_url or base_url

        if self.stt_base_url == self.tts_base_url and self.stt_api_key == self.tts_api_key:
            if self.stt_api_key is None:
                self.tts_client = None
            else:
                self.tts_client = AsyncOpenAI(api_key=self.tts_api_key, base_url=self.tts_base_url)
            self.stt_client = self.tts_client
        else:
            if self.stt_api_key is None:
                self.tts_client = None
            else:
                self.tts_client = AsyncOpenAI(api_key=self.stt_api_key, base_url=self.stt_base_url)
            if self.stt_api_key is None:
                self.stt_client = None
            else:
                self.stt_client = AsyncOpenAI(api_key=self.stt_api_key, base_url=self.stt_base_url)

    @property
    def system_prompt(self) -> Optional[str]:
        return {"role": "system", "content": self.__system_prompt}
    
    @system_prompt.setter
    def system_prompt(self, value: Optional[str]):
        self.system_prompt = value
    
    async def format(self, message: Message) -> Union[TextOpenAIMessage, AudioOpenAIMessage]:
        if message.data_type == "text":
            return TextOpenAIMessage(role=message.role, content=message.content)
        elif message.data_type == "audio":
            return AudioOpenAIMessage(role=message.role, content=message.content)
        else:
            raise ValueError(f"Invalid data type: {message.data_type}")

    async def stt(self, 
                  formatted_data: List[Union[AudioOpenAIMessage, TextOpenAIMessage]],
                  gen_config: Optional[Dict[str, Any]] = None) -> str:
        if not self.stt_model:
            raise ValueError("STT model is not set")
        if gen_config is None:
            gen_config = {}
        response = await self.stt_client.chat.completions.create(
            model=self.stt_model,
            messages=[self.system_prompt] + [message.as_json() for message in formatted_data],
            **gen_config
        )
        if isinstance(response, ChatCompletion):
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")

    async def stt_stream(self, 
                         formatted_data: List[Union[AudioOpenAIMessage, TextOpenAIMessage]],
                         gen_config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        if not self.stt_model:
            raise ValueError("STT model is not set")
        if gen_config is None:
            gen_config = {}
        response = await self.stt_client.chat.completions.create(
            model=self.stt_model,
            messages=[self.system_prompt] + [message.as_json() for message in formatted_data],
            stream=True,
            **gen_config
        )
        async for chunk in response:
            if isinstance(chunk, ChatCompletionChunk):
                yield chunk.choices[0].delta.content
            else:
                raise ValueError(f"Unsupported response type: {type(chunk)}")
    
    async def tts(self, 
                  text: str, 
                  response_format: str = "pcm", 
                  gen_config: Optional[Dict[str, Any]] = None) -> bytes:
        if not self.tts_model:
            raise ValueError("TTS model is not set")
        if gen_config is None:
            gen_config = {"voice": "nova"}
        response = await self.tts_client.audio.speech.create(
            model=self.tts_model,
            input=text,
            response_format=response_format,
            **gen_config
        )
        return response.content

    async def tts_stream(self, 
                         text: str, 
                         response_format: str = "pcm", 
                         gen_config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[bytes, None]:
        if not self.tts_model:
            raise ValueError("TTS model is not set")
        if gen_config is None:
            gen_config = {"voice": "nova"}
        async with self.tts_client.audio.speech.with_streaming_response.create(
            model=self.tts_model,
            input=text,
            response_format=response_format,
            **gen_config
        ) as response:
            async for chunk in response.iter_bytes():
                yield chunk



class AstLLmProvider(OpenAIProvider):
    def __init__(self,  
                 *args,
                 ast_model: Optional[str] = "openai/whisper-large-v3-turbo",
                 overwrite_ast_model_api_key: Optional[str] = None,
                 overwrite_ast_model_base_url: Optional[str] = None,
                 language: Optional[str] = "en",
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.ast_model = ast_model
        self.ast_api_key = overwrite_ast_model_api_key or self.stt_api_key
        self.ast_base_url = overwrite_ast_model_base_url or self.stt_base_url
        if self.stt_api_key == self.ast_api_key and self.stt_base_url == self.ast_base_url:
            self.ast_client = self.stt_client # using transitive logic: stt > tts, tts > ast -> stt > ast
        else:
            if self.ast_api_key is None:
                self.ast_client = None
            else:
                self.ast_client = AsyncOpenAI(api_key=self.ast_api_key, base_url=self.ast_base_url)
        self.ast_language = language if language else "en"


    async def format(self, message: Message) -> Any:
        assert message.data_type in ["text", "audio"], "Invalid data type"
        content = message.content
        if message.data_type == "audio":
            if not self.ast_model:
                raise ValueError("AST model is not set")
            content = await self.ast_client.audio.transcriptions.create(
                model=self.ast_model, 
                file=message.content,
                language=self.ast_language
            )
            if isinstance(content, Transcription):
                content = content.text
            else:
                raise ValueError(f"Invalid content type: {type(content)}")
        return TextOpenAIMessage(role=message.role, content=content)
    

    async def stt(self, 
                  formatted_data: List[TextOpenAIMessage],
                  gen_config: Optional[Dict[str, Any]] = None) -> str:
        if not self.stt_model:
            raise ValueError("LLM model is not set")
        if gen_config is None:
            gen_config = {}
        response = await self.stt_client.chat.completions.create(
            model=self.stt_model,
            messages=[self.system_prompt] + [message.as_json() for message in formatted_data],
            **gen_config
        )
        if isinstance(response, ChatCompletion):
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")
    
    async def stt_stream(self, 
                         formatted_data: List[TextOpenAIMessage],
                         gen_config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        if not self.stt_model:
            raise ValueError("LLM model is not set")
        if gen_config is None:
            gen_config = {}
        response = await self.stt_client.chat.completions.create(
            model=self.stt_model,
            messages=[self.system_prompt] + [message.as_json() for message in formatted_data],
            stream=True,
            **gen_config
        )
        async for chunk in response:
            if isinstance(chunk, ChatCompletionChunk):
                yield chunk.choices[0].delta.content
            else:
                raise ValueError(f"Unsupported response type: {type(chunk)}")





