
from .base import BaseTTSProvider, BaseSTTProvider, Message

from openai import AsyncOpenAI
from openai.types.audio import Transcription
from openai.types.chat import ChatCompletion, ChatCompletionChunk
# from openai.types import AsyncResponseContextManager, AsyncStreamedBinaryAPIResponse, HttpxBinaryResponseContent

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
                 pcm_response_format: Optional[str] = None,
                 response_sample_rate: Optional[int] = None,
                 tts_gen_config: Optional[Dict[str, Any]] = None,
                 stt_gen_config: Optional[Dict[str, Any]] = None,
                 overwrite_stt_model_api_key: Optional[str] = None,
                 overwrite_stt_model_base_url: Optional[str] = None,
                 overwrite_tts_model_api_key: Optional[str] = None,
                 overwrite_tts_model_base_url: Optional[str] = None,
                 ):
        """
        pcm_response_format: The pcm format of the response from the tts_provider; pcm for openai, pcm_24000 for elevenlabs
        response_sample_rate: The sample rate of the response from the tts_provider.
        """
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.pcm_response_format = pcm_response_format
        self.response_sample_rate = response_sample_rate
        self.tts_gen_config = tts_gen_config
        self.stt_gen_config = stt_gen_config
        # super(BaseTTSProvider, self).__init__(pcm_response_format, response_sample_rate, tts_gen_config)
        # super(BaseSTTProvider, self).__init__(system_prompt, stt_gen_config)

        self.stt_api_key = overwrite_stt_model_api_key or api_key
        self.stt_base_url = overwrite_stt_model_base_url or base_url
        self.tts_api_key = overwrite_tts_model_api_key or api_key
        self.tts_base_url = overwrite_tts_model_base_url or base_url


        if self.stt_api_key:
            self.stt_client = AsyncOpenAI(api_key=self.stt_api_key, base_url=self.stt_base_url)
        else:
            self.stt_client = None
        
        if self.tts_api_key == self.stt_api_key and self.tts_base_url == self.stt_base_url:
            self.tts_client = self.stt_client
        else:
            if self.tts_api_key:
                self.tts_client = AsyncOpenAI(api_key=self.tts_api_key, base_url=self.tts_base_url)
            else:
                self.tts_client = None

        if not self.stt_client and not self.tts_client:
            raise ValueError("STT or TTS client is not set")

    @property
    def formatted_system_prompt(self) -> Optional[str]:
        return {"role": "system", "content": self.__system_prompt}
        
    async def format(self, message: Message) -> Union[TextOpenAIMessage, AudioOpenAIMessage]:
        if message.data_type == "text":
            return TextOpenAIMessage(role=message.role, content=message.content)
        elif message.data_type == "audio":
            return AudioOpenAIMessage(role=message.role, content=message.content)
        else:
            raise ValueError(f"Invalid data type: {message.data_type}")

    async def stt(self, 
                  formatted_data: List[Union[AudioOpenAIMessage, TextOpenAIMessage]]) -> str:
        if not self.stt_client:
            raise ValueError("STT client is not set")
        response = await self.stt_client.chat.completions.create(
            model=self.stt_model,
            messages=[self.formatted_system_prompt] + [message.as_json() for message in formatted_data],
            **self.stt_gen_config
        )
        if isinstance(response, ChatCompletion):
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")

    async def stt_stream(self, 
                         formatted_data: List[Union[AudioOpenAIMessage, TextOpenAIMessage]]) -> AsyncGenerator[str, None]:
        if not self.stt_client:
            raise ValueError("STT client is not set")
        response = await self.stt_client.chat.completions.create(
            model=self.stt_model,
            messages=[self.formatted_system_prompt] + [message.as_json() for message in formatted_data],
            stream=True,
            **self.stt_gen_config
        )
        async for chunk in response:
            if isinstance(chunk, ChatCompletionChunk):
                yield chunk.choices[0].delta.content
            else:
                raise ValueError(f"Unsupported response type: {type(chunk)}")
    
    async def tts(self, text: str) -> bytes:
        if not self.tts_client:
            raise ValueError("TTS client is not set")
        response = await self.tts_client.audio.speech.create(
            model=self.tts_model,
            input=text,
            response_format=self.pcm_response_format,
            **self.tts_gen_config
        )
        # if not isinstance(response, HttpxBinaryResponseContent):
            # raise ValueError(f"Unsupported response type: {type(response)}")
        return response.content

    async def tts_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        if not self.tts_client:
            raise ValueError("TTS client is not set")
        async with self.tts_client.audio.speech.with_streaming_response.create(
            model=self.tts_model,
            input=text,
            response_format=self.pcm_response_format,
            **self.tts_gen_config
        ) as response:
            # if not isinstance(response, AsyncResponseContextManager[AsyncStreamedBinaryAPIResponse]):
            #     raise ValueError(f"Unsupported response type: {type(response)}")
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
                  formatted_data: List[TextOpenAIMessage]) -> str:
        if not self.stt_model:
            raise ValueError("LLM model is not set")
        response = await self.stt_client.chat.completions.create(
            model=self.stt_model,
            messages=[self.system_prompt] + [message.as_json() for message in formatted_data],
            **self.stt_gen_config
        )
        if isinstance(response, ChatCompletion):
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")
    
    async def stt_stream(self, 
                         formatted_data: List[TextOpenAIMessage]) -> AsyncGenerator[str, None]:
        if not self.stt_model:
            raise ValueError("LLM model is not set")
        response = await self.stt_client.chat.completions.create(
            model=self.stt_model,
            messages=[self.system_prompt] + [message.as_json() for message in formatted_data],
            stream=True,
            **self.stt_gen_config
        )
        async for chunk in response:
            if isinstance(chunk, ChatCompletionChunk):
                yield chunk.choices[0].delta.content
            else:
                raise ValueError(f"Unsupported response type: {type(chunk)}")





