"""
Example usage of RestCallback with RTP-LLM system
This demonstrates how to integrate REST API callbacks with your voice agent
"""

import asyncio
import logging
from typing import Optional

# Import your RTP-LLM components
from src.rtp_llm.callbacks import RestCallback
from src.rtp_llm.server import Server
# You'd import other necessary components like adapters, agents, etc.

# Setup logging to see the callback activity
logging.basicConfig(level=logging.INFO)

# Example 1: Basic RestCallback Setup
async def basic_rest_callback_example():
    """
    Basic example of creating and using a RestCallback
    """
    print("=== Basic RestCallback Example ===")
    
    # Create the RestCallback with your webhook endpoints
    callback = RestCallback(
        base_url="https://your-webhook-server.com",
        on_stt_endpoint="/webhooks/speech-to-text",
        on_tts_endpoint="/webhooks/text-to-speech", 
        on_start_endpoint="/webhooks/session-start",
        on_error_endpoint="/webhooks/error",
        on_finish_endpoint="/webhooks/session-end"
    )
    
    # Simulate callback events (this is what your RTP server does internally)
    uid = "session_123"
    
    print(f"Starting session {uid}")
    await callback.on_start(uid)
    
    print("Simulating STT event")
    await callback.on_stt(uid, "Hello, how are you today?")
    
    print("Simulating TTS event") 
    await callback.on_tts(uid, "I'm doing well, thank you for asking!")
    
    print("Ending session")
    await callback.on_finish(uid, "Goodbye!")
    
    # Clean up
    await callback.client.aclose()


# Example 2: Selective Endpoints (Only Error Monitoring)
async def error_monitoring_callback_example():
    """
    Example using RestCallback only for error monitoring
    """
    print("\n=== Error Monitoring Example ===")
    
    # Only monitor errors and session lifecycle
    callback = RestCallback(
        base_url="https://monitoring.example.com",
        on_error_endpoint="/api/v1/errors",
        on_start_endpoint="/api/v1/sessions/start",
        on_finish_endpoint="/api/v1/sessions/end"
        # Notice: no STT/TTS endpoints - those events will be ignored
    )
    
    uid = "session_456"
    
    await callback.on_start(uid)
    
    # Simulate an error
    try:
        raise ValueError("Simulated processing error")
    except Exception as e:
        await callback.on_error(uid, e)
    
    await callback.on_finish(uid)
    await callback.client.aclose()


# Example 3: Custom Headers and Authentication
async def authenticated_callback_example():
    """
    Example with authentication headers and custom configuration
    """
    print("\n=== Authenticated Callback Example ===")
    
    # Custom headers for authentication
    custom_headers = {
        "Authorization": "Bearer your-api-token-here",
        "X-Client-Version": "1.0.0"
    }
    
    callback = RestCallback(
        base_url="https://secure-api.example.com",
        on_stt_endpoint="/secure/stt",
        on_tts_endpoint="/secure/tts",
        headers=custom_headers,
        timeout=10.0  # 10 second timeout
    )
    
    uid = "secure_session_789"
    await callback.on_stt(uid, "Authenticated speech recognition result")
    await callback.client.aclose()


# Example 4: Integration with RTP Server (Pseudocode)
async def server_integration_example():
    """
    Example showing how to integrate RestCallback with your RTP server
    NOTE: This is pseudocode - you'd need to configure your actual server components
    """
    print("\n=== Server Integration Example ===")
    
    # Setup your callback
    webhook_callback = RestCallback(
        base_url="https://your-backend.com",
        on_stt_endpoint="/api/speech/transcribed",
        on_tts_endpoint="/api/speech/synthesized", 
        on_start_endpoint="/api/session/started",
        on_error_endpoint="/api/session/error",
        on_finish_endpoint="/api/session/ended"
    )
    
    # This is how you'd use it with your actual server
    # (You'd need to configure adapter, agent, vad, etc.)
    """
    server = Server(
        adapter=your_adapter,
        audio_buffer=your_buffer,
        flow_manager=your_flow_manager,
        vad=your_vad,
        agent=your_agent
    )
    
    # Run the server with the callback
    await server.run(
        callback=webhook_callback,
        uid="user_session_001",
        first_message="Hello! How can I help you today?"
    )
    """
    
    print("Server would be running with webhook callbacks enabled")


# Example 5: Testing Your Webhook Endpoints
async def test_webhook_endpoints():
    """
    Simple test to verify your webhook endpoints are working
    """
    print("\n=== Testing Webhook Endpoints ===")
    
    # Test with a simple local server endpoint
    callback = RestCallback(
        base_url="http://localhost:8000",  # Your local test server
        on_stt_endpoint="/test/stt"
    )
    
    try:
        await callback.on_stt("test_uid", "Test message")
        print("✅ Webhook test successful!")
    except Exception as e:
        print(f"❌ Webhook test failed: {e}")
    
    await callback.client.aclose()


# Example webhook server payload formats
def example_payload_formats():
    """
    Shows the JSON payloads your webhook endpoints will receive
    """
    print("\n=== Example Webhook Payloads ===")
    
    examples = {
        "STT/TTS Event": {
            "uid": "session_123",
            "text": "The transcribed or synthesized text"
        },
        "Start/Finish Event": {
            "uid": "session_123",
            "text": "Optional finish message"  # May be None for start events
        },
        "Error Event": {
            "uid": "session_123", 
            "error": {
                "type": "ValueError",
                "message": "Detailed error message"
            }
        }
    }
    
    for event_type, payload in examples.items():
        print(f"{event_type}: {payload}")


# Main execution
async def main():
    """Run all examples"""
    await basic_rest_callback_example()
    await error_monitoring_callback_example() 
    await authenticated_callback_example()
    await server_integration_example()
    await test_webhook_endpoints()
    example_payload_formats()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 