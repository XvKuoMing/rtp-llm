"""
external chat history manager, responsible for storing and retrieving chat history, switching between providers
"""

from typing import List, Optional, Awaitable, Callable

from ..providers.base import Message
from .base import BaseChatHistory


class ChatHistoryLimiter(BaseChatHistory):
    def __init__(self, 
                 messages: Optional[List[Message]] = None, 
                 limit: int = 10):
        self.limit = limit
        self.__messages = messages or []
        self.__last_formatted_messages = []
        self._last_formatted_func = None
    
    @property
    def messages(self) -> List[Message]:
        return self.__messages
    
    async def add_message(self, message: Message) -> None:
        """
        add message to the history
        """
        if self.limit <= 0:
            return
        self.__messages.append(message)
        if len(self.__messages) > self.limit:
            self.__messages.pop(0)
        
        if self._last_formatted_func is not None:
            fmt_msg = await self._last_formatted_func(message)
            self.__last_formatted_messages.append(fmt_msg)
            if len(self.__last_formatted_messages) > self.limit:
                self.__last_formatted_messages.pop(0)
    
    async def get_messages(self, formatter: Callable[[Message], Awaitable[Message]]) -> List[Message]:
        assert formatter is not None, "Formatter cannot be None"
        if self.limit <= 0:
            return []
        # Compare the actual formatter function/method, not just the name
        # This ensures we re-format when switching between different provider instances
        if self._last_formatted_func is None or self._last_formatted_func != formatter:
            self._last_formatted_func = formatter
            self.__last_formatted_messages = [await formatter(msg) for msg in self.__messages]
        return self.__last_formatted_messages
    
    def clear(self) -> None:
        self.__messages = []
        self.__last_formatted_messages = []
        self._last_formatted_func = None







