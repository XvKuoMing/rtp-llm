#!/usr/bin/env python3
"""
Example showing how to use the RestCallback feature with the RTP LLM server.

This example demonstrates how to:
1. Configure a RunParams object with RestCallback settings
2. Send the configuration to the server to enable REST callbacks

The RestCallback will send HTTP POST requests to your specified endpoints
when certain events occur in the conversation flow.
"""

import json
import requests
from src.entrypoint.models.server import RunParams, RestCallbackConfig

def create_run_params_with_rest_callback():
    """Create RunParams with RestCallback configuration"""
    
    # Configure REST callback endpoints
    rest_callback_config = RestCallbackConfig(
        base_url="https://your-api.example.com",
        on_response_endpoint="/webhooks/on_response",  # Called when AI responds
        on_start_endpoint="/webhooks/on_start",        # Called when conversation starts
        on_error_endpoint="/webhooks/on_error",        # Called on errors
        on_finish_endpoint="/webhooks/on_finish"       # Called when conversation ends
    )
    
    # Create RunParams with the callback configuration
    run_params = RunParams(
        uid="example_session_123",
        first_message="Hello! How can I help you today?",
        allow_interruptions=True,
        system_prompt="You are a helpful assistant.",
        tts_volume=0.8,
        rest_callback=rest_callback_config  # <-- This is the new feature!
    )
    
    return run_params

def create_run_params_without_callback():
    """Create RunParams without any callback (traditional usage)"""
    
    run_params = RunParams(
        uid="example_session_124",
        first_message="Hello! This session has no callbacks.",
        allow_interruptions=False,
        system_prompt="You are a helpful assistant.",
        tts_volume=1.0,
        rest_callback=None  # <-- No callback, runs normally
    )
    
    return run_params

if __name__ == "__main__":
    # Example 1: With REST callback
    print("=== Example 1: RunParams with RestCallback ===")
    params_with_callback = create_run_params_with_rest_callback()
    print(json.dumps(params_with_callback.model_dump(), indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Without callback (traditional)
    print("=== Example 2: RunParams without callback ===")
    params_without_callback = create_run_params_without_callback()
    print(json.dumps(params_without_callback.model_dump(), indent=2))
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: How to send to your RTP LLM server
    print("=== Example 3: How to use with your server ===")
    print("""
    # To use these RunParams with your RTP LLM server:
    
    import requests
    
    # Assuming your server is running on localhost:8000
    server_url = "http://localhost:8000/server/run"
    
    # Send the RunParams to start a session with callbacks
    response = requests.post(
        server_url, 
        json=params_with_callback.model_dump()
    )
    
    if response.status_code == 200:
        print("Server started successfully with REST callbacks!")
        # Now your callback endpoints will receive webhook calls
        # when events happen in the conversation
    """)
