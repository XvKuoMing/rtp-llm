from abc import ABC, abstractmethod, ABCMeta
from typing import Union, AsyncGenerator, Any, Optional, Dict, Set
from dataclasses import dataclass
import inspect

@dataclass(frozen=True)
class Message:
    role: str
    content: Union[str, bytes]
    data_type: str = "text" # audio, video, image.

    def __post_init__(self):
        if self.data_type not in ["text", "audio", "video", "image"]:
            raise ValueError(f"Invalid data type: {self.data_type}")


class MetaProvider(ABCMeta):
    # Registry to store all provider classes by their name_tag
    _provider_registry = {}
    
    def __new__(metacls, name, bases, attrs):
        # Create the new class first
        new_cls = super().__new__(metacls, name, bases, attrs)
        
        # Only set name_tag if it's defined directly on the class, not inherited
        # This ensures each subclass gets its own tag derived from its own name
        if "name_tag" not in attrs:
            new_cls.name_tag = name.lower().replace("provider", "")
        
        # Register the provider in the registry (skip base classes)
        if name not in ['BaseTTSProvider', 'BaseSTTProvider']:
            MetaProvider._provider_registry[new_cls.name_tag] = new_cls
            
        return new_cls
    
    @classmethod
    def get_provider(cls, name_tag: str):
        """Get a provider class by its name_tag. Returns None if not found."""
        return cls._provider_registry.get(name_tag)
    
    @classmethod
    def provider_exists(cls, name_tag: str) -> bool:
        """Check if a provider with the given name_tag exists."""
        return name_tag in cls._provider_registry
    
    @classmethod
    def list_providers(cls) -> Dict[str, type]:
        """Get all registered providers as a dict of {name_tag: provider_class}."""
        return cls._provider_registry.copy()
    
    @classmethod
    def create_provider_from_config(cls, name_tag: str, config: Dict[str, Any]) -> Any:
        """
        Create a provider from a config.
        """
        provider_class = cls.get_provider(name_tag)
        if provider_class is None:
            raise ValueError(f"Provider {name_tag} not found; available providers: {cls.list_providers()}")
        return provider_class(**config)
    
    @classmethod
    def get_providers_arg_info(cls, name_tag: str) -> Dict[str, Any]:
        """
        Get the argument info of the provider's __init__ method.
        """
        provider_class = cls.get_provider(name_tag)
        if provider_class is None:
            raise ValueError(f"Provider {name_tag} not found")
        gen_config_info = set()
        if issubclass(provider_class, BaseTTSProvider):
            gen_config_info = provider_class.get_tts_gen_config_info()
        elif issubclass(provider_class, BaseSTTProvider):
            gen_config_info = provider_class.get_stt_gen_config_info()
        else:
            raise ValueError(f"Provider {name_tag} is not a TTS or STT provider")
        
        # Get the __init__ method signature
        sig = inspect.signature(provider_class.__init__)
        init_args_info = {}
        
        for param_name, param in sig.parameters.items():
            if param_name not in ["self", "args", "kwargs", "gen_config"]:
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
                init_args_info[param_name] = param_type
        
        # Add gen_config info separately since it contains nested config options
        init_args_info["gen_config"] = gen_config_info
        
        return init_args_info
    
    @classmethod
    def get_all_providers_arg_info(cls) -> Dict[str, Any]:
        """
        Get the config info of all providers.
        """
        all_config_info = {}
        for name_tag in cls._provider_registry:
            all_config_info[name_tag] = cls.get_providers_arg_info(name_tag)
        return all_config_info


class BaseTTSProvider(ABC, metaclass=MetaProvider):
    """
    must produce output in pcm format
    """


    @abstractmethod
    def __init__(self, 
                 *args,
                 gen_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        pcm_response_format: The pcm format of the response from the tts_provider; pcm for openai, pcm_24000 for elevenlabs
        response_sample_rate: The sample rate of the response from the tts_provider.
        """
        self.tts_gen_config = gen_config or {}
    

    @property
    @abstractmethod
    def response_sample_rate(self) -> int:
        """
        Get the sample rate of the response from the tts_provider.
        """
        pass
    

    @property
    def tts_footprint(self) -> str:
        """
        A string that uniquely identifies the tts_provider.
        IMPORTANT: do not include the gen_config in the footprint, since it is per generation config
        """
        return f"{self.response_sample_rate}_{str(self.tts_gen_config)}"

    @abstractmethod
    def get_tts_gen_config_info(self) -> Set[str]:
        """
        Get the tts_config info of the provider -> name and default value
        """
        pass


    @abstractmethod
    async def tts(self, 
                  text: str,
                  *,
                  gen_config: Optional[Dict[str, Any]] = None,
                  ) -> bytes:
        """
        Generate audio from text.
        """
        pass


    async def tts_stream(self, 
                         text: str,
                         *,
                         gen_config: Optional[Dict[str, Any]] = None,
                         ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio from text with streaming.
        """
        raise NotImplementedError("Streaming is not supported for this STT provider")


class BaseSTTProvider(ABC, metaclass=MetaProvider):

    @abstractmethod
    def __init__(self, 
                 *args,
                 system_prompt: Optional[str] = None, 
                 gen_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        self.system_prompt = system_prompt
        self.stt_gen_config = gen_config or {}

    
    @abstractmethod
    def get_stt_gen_config_info(self) -> Set[str]:
        """
        Get the stt_config info of the provider -> name and default value
        """
        ...

    @abstractmethod
    async def format(self, message: Message) -> Any:
        """given a messsage, format it to the provider's format"""
        pass

    @abstractmethod
    async def stt(self, 
                  formatted_wav_audio: Any,
                  *,
                  system_prompt: Optional[str] = None,
                  gen_config: Optional[Dict[str, Any]] = None,
                  ) -> str:
        """
        Generate text from audio message: a message with audio formatted to the provider's format.
        """
        pass


    async def stt_stream(self,
                         formatted_wav_audio: Any,
                         *,
                         system_prompt: Optional[str] = None,
                         gen_config: Optional[Dict[str, Any]] = None,
                         ) -> AsyncGenerator[str, None]:
        """
        Generate text from audio message with streaming.
        """
        raise NotImplementedError("Streaming is not supported for this STT provider")
