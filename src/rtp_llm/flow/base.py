from ..vad.base import VoiceState
from abc import ABC, abstractmethod


class BaseChatFlowManager(ABC):

    @abstractmethod
    async def run_agent(self, voice_state: VoiceState) -> bool:
        """
        recieves vad voice state and returns if the agent should be run
        """
        ...

    @abstractmethod
    def reset(self):
        """
        reset the flow manager to the start state
        """
        ...

