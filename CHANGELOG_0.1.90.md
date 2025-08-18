# Changelog for v0.1.90

## 🚨 CRITICAL FIX: Stream Continuity During Tool Execution

### The Critical Problem
After calling certain tools (especially `get_user_weather`), the agent would go completely silent. The server was generating responses correctly, but clients never displayed them. Once this happened, the agent remained muted for the rest of the session.

### Root Cause Identified
In v0.1.89, we stopped yielding ANY data during tool call processing to avoid confusing clients with tool call information. However, this created a worse problem:

1. Client sends request
2. Server starts streaming
3. Tools are detected → **Server stops yielding completely**
4. Client sees no data → thinks stream has ended/timed out
5. Client disconnects
6. Server tries to send response → **Client is already gone**
7. Agent appears "muted" from this point forward

### The Solution
We now send **keep-alive chunks** (empty content with `finish_reason: null`) during tool processing to maintain the SSE connection:

```javascript
// Keep-alive chunks during tool processing
{ 
  delta: { content: "" },  // Empty content
  finish_reason: null       // Stream continues
}
```

### What's Fixed

1. **During tool call detection**: Send keep-alive chunks for each tool call
2. **When tools are confirmed**: Send a keep-alive chunk
3. **Before each tool execution**: Send a keep-alive chunk
4. **After tool execution**: Send the actual response

This ensures the SSE stream remains active throughout the entire process.

## 🎯 Critical Changes

```python
# Before (v0.1.89 - BROKEN):
for tc in tool_calls:
    # Process tool calls
    # yield nothing - CLIENT DISCONNECTS!

# After (v0.1.90 - FIXED):
for tc in tool_calls:
    # Process tool calls
    yield keep_alive_chunk  # Keep stream alive!
```

## 🚀 Deployment
```bash
pip install --upgrade brain-proxy==0.1.90
```

## ✅ Expected Behavior

With v0.1.90:
- Tools execute normally
- **Stream stays connected** throughout
- **Responses display immediately** after tool execution
- **No more "muted" agent** after weather or other tools
- Session remains fully functional

## 📊 Testing

Test with the weather tool:
1. Ask "what's the weather?"
2. Tool executes
3. **Response appears immediately**: "El clima actual en Santiago..."
4. Continue conversation normally - agent stays responsive!

## 🔍 Debug Output

You'll see keep-alive chunks being sent:
- During tool call processing
- Before tool execution
- Throughout the entire flow

The client connection remains stable and responses display correctly!
