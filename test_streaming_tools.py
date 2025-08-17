#!/usr/bin/env python3
"""Test script to verify streaming tool responses work correctly."""

import asyncio
import json
from openai import AsyncOpenAI

# Configure the client to use brain-proxy
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1/test_tenant",
    api_key="dummy"  # brain-proxy doesn't need a real key
)

# Define a simple tool for testing
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}]

async def test_streaming_with_tools():
    """Test that tool responses are properly streamed to the client."""
    print("Testing streaming with tool calls...")
    print("-" * 50)
    
    # Make a request that should trigger tool usage
    stream = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What time is it right now?"}
        ],
        tools=tools,
        tool_choice="auto",
        stream=True
    )
    
    # Collect all chunks
    full_response = ""
    tool_response_found = False
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
            
            # Check if we received tool response
            if "Tool 'get_current_time' response:" in content:
                tool_response_found = True
    
    print("\n" + "-" * 50)
    print(f"\nTool response found in stream: {tool_response_found}")
    print(f"Full response length: {len(full_response)} characters")
    
    if tool_response_found:
        print("✅ SUCCESS: Tool responses are being streamed correctly!")
    else:
        print("❌ FAILURE: Tool responses are NOT being streamed!")
    
    return tool_response_found

async def main():
    """Run the test."""
    try:
        # First, register the tool with brain-proxy
        import httpx
        async with httpx.AsyncClient() as client:
            # Register our test tool
            response = await client.post(
                "http://localhost:8000/v1/test_tenant/tools",
                json=tools
            )
            print(f"Tool registration response: {response.status_code}")
            
        # Run the streaming test
        success = await test_streaming_with_tools()
        return success
        
    except Exception as e:
        print(f"Error during test: {e}")
        return False

if __name__ == "__main__":
    print("Starting brain-proxy streaming tool test...")
    success = asyncio.run(main())
    exit(0 if success else 1)
