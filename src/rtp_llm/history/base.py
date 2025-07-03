"""
external chat history manager, responsible for storing and retrieving chat history, switching between providers
"""

from typing import List, Awaitable, Callable

from ..providers.base import Message
from abc import ABC, abstractmethod


class BaseChatHistory(ABC):
    
    @abstractmethod
    async def add_message(self, message: Message) -> None:
        """
        must add message to the history
        """
        ...
    
    @abstractmethod
    async def get_messages(self, formatter: Callable[[Message], Awaitable[Message]]) -> List[Message]:
        """
        must return messages in the history
        """
        ...
    

    @abstractmethod
    def clear(self) -> None:
        """
        must clear the history
        """
        ...







