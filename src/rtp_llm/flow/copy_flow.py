from ..vad.base import VoiceState
from .base import BaseChatFlowManager
import logging

logger = logging.getLogger(__name__)


class CopyFlowManager(BaseChatFlowManager):


    def __init__(self):
        self.last_state = VoiceState.SILENCE
    

    async def run_agent(self, voice_state: VoiceState) -> bool:
        """
        recieves vad voice state and returns if the agent should be run
        """

        if self.last_state == VoiceState.SPEECH and voice_state == VoiceState.SILENCE:
            # if user stopped speaking, we should run the agent
            return True
        self.last_state = voice_state
        return False

    
    def reset(self):
        """
        reset the flow manager to the start state
        """
        self.last_state = VoiceState.SILENCE
