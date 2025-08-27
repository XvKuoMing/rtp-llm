from ..vad.base import VoiceState
from .base import BaseChatFlowManager
import logging

logger = logging.getLogger(__name__)


class CopyFlowManager(BaseChatFlowManager):


    def __init__(self, throttle_windows: int = 2):
        self.last_state = VoiceState.SILENCE
        self.silence_counter = 0
        self.throttle_windows = throttle_windows
        self.in_throttle_mode = False
    

    async def run_agent(self, voice_state: VoiceState) -> bool:
        """
        Receives vad voice state and returns if the agent should be run.
        
        Logic: 
        - When we transition from SPEECH to SILENCE, enter throttle mode
        - Count consecutive SILENCE states after the transition
        - Only run agent after throttle_windows consecutive SILENCE states
        - Reset if SPEECH resumes during throttle period
        """
        should_run = False
        
        # Detect transition from SPEECH to SILENCE
        if self.last_state == VoiceState.SPEECH and voice_state == VoiceState.SILENCE:
            self.in_throttle_mode = True
            self.silence_counter = 1  # This is the first silence after speech
        
        # If we're in throttle mode and current state is SILENCE
        elif self.in_throttle_mode and voice_state == VoiceState.SILENCE:
            self.silence_counter += 1
            
            # Check if we've reached the required number of consecutive silence windows
            if self.silence_counter >= self.throttle_windows:
                should_run = True
                self.in_throttle_mode = False
                self.silence_counter = 0
        
        # If speech resumes during throttle period, reset throttle
        elif self.in_throttle_mode and voice_state == VoiceState.SPEECH:
            self.in_throttle_mode = False
            self.silence_counter = 0
        
        self.last_state = voice_state
        return should_run

    
    def reset(self):
        """
        reset the flow manager to the start state
        """
        self.last_state = VoiceState.SILENCE
        self.silence_counter = 0
        self.in_throttle_mode = False
