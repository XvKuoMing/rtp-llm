from ..vad.base import VoiceState
from .base import BaseChatFlowManager
import logging

logger = logging.getLogger(__name__)


class CopyFlowManager(BaseChatFlowManager):


    def __init__(self, throttle_windows: int = 3):
        self.last_state = VoiceState.SILENCE
        self.throttle_counter = 0
        self.throttle_windows = throttle_windows
    

    async def run_agent(self, voice_state: VoiceState) -> bool:
        """
        recieves vad voice state and returns if the agent should be run
        """
        if (self.last_state == VoiceState.SPEECH and voice_state == VoiceState.SILENCE):
            self.throttle_counter += 1

        if self.throttle_counter >= self.throttle_windows:
            # if user stopped speaking, we should run the agent
            self.throttle_counter = 0
            return True
        self.last_state = voice_state
        return False

    
    def reset(self):
        """
        reset the flow manager to the start state
        """
        self.last_state = VoiceState.SILENCE
