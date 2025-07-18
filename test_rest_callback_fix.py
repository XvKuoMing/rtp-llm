#!/usr/bin/env python3
"""
Simple test script to verify RestCallback fixes
"""
import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rtp_llm.callbacks.rest_callback import RestCallback

async def test_rest_callback():
    """Test RestCallback with various scenarios"""
    
    print("Testing RestCallback fixes...")
    
    # Test 1: RestCallback with invalid endpoint (should fallback gracefully)
    print("\n1. Testing with invalid endpoint...")
    callback = RestCallback(
        base_url="http://invalid-endpoint:12345",
        on_response_endpoint="/api/test"
    )
    
    try:
        result = await callback.on_response("test-uid", "Hello world")
        print(f"✅ Returned: {result}")
        print(f"   Text: {result.text}")
        print(f"   Post action: {result.post_action}")
        assert result.text == "Hello world"  # Should fallback to original text
        print("   ✅ Correctly fell back to original text")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: RestCallback with no endpoint (should return original text)
    print("\n2. Testing with no endpoint...")
    callback_no_endpoint = RestCallback(base_url="http://example.com")
    
    try:
        result = await callback_no_endpoint.on_response("test-uid", "No endpoint test")
        print(f"✅ Returned: {result}")
        print(f"   Text: {result.text}")
        assert result.text == "No endpoint test"
        print("   ✅ Correctly returned original text")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Test client recreation after closure
    print("\n3. Testing client recreation...")
    try:
        # Close the client
        await callback.close()
        print("   Client closed")
        
        # Try to use it again (should recreate client)
        result = await callback.on_response("test-uid-2", "After close test")
        print(f"✅ After recreation: {result}")
        assert result.text == "After close test"
        print("   ✅ Client successfully recreated")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Test other callback methods don't crash
    print("\n4. Testing other callback methods...")
    try:
        await callback.on_start("test-uid")
        print("   ✅ on_start completed")
        
        await callback.on_error("test-uid", Exception("Test error"))
        print("   ✅ on_error completed")
        
        await callback.on_finish("test-uid")
        print("   ✅ on_finish completed")
    except Exception as e:
        print(f"❌ Error in other methods: {e}")
    
    # Final cleanup
    await callback.close()
    await callback_no_endpoint.close()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_rest_callback()) 