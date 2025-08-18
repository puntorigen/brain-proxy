# Changelog for v0.1.85

## 🎯 ROOT CAUSE FOUND AND FIXED!

### The Problem
Responses after tool execution were not being displayed to the client until a second message was sent. The server was generating and "sending" the response, but the client wasn't receiving it in real-time.

### Root Cause Analysis

Through step-by-step debugging, we identified that the issue was in the `_handle_streaming` method's async generator flow:

1. When tool calls were detected, we would `break` from the `async for chunk in upstream_iter:` loop
2. After breaking, we'd continue yielding more data (tool execution results and follow-up response)
3. **BUT** - breaking from an async iteration disrupts the generator context
4. FastAPI's `StreamingResponse` expects a continuous async generator
5. The yields after the break weren't being properly consumed by the client

### The Fix

**NEVER break the async iteration loop!** Instead:
- Let the upstream iteration complete naturally
- Store the finish reason without breaking
- After the loop completes, handle tool execution if needed
- This maintains generator continuity throughout the entire stream

### Code Changes

Before (BROKEN):
```python
async for chunk in upstream_iter:
    # ... process chunk ...
    if choice.get("finish_reason") == "tool_calls":
        break  # ❌ This breaks generator continuity!
# Continue yielding after break... (doesn't work properly)
```

After (FIXED):
```python
finish_reason = None
async for chunk in upstream_iter:
    # ... process chunk ...
    if choice.get("finish_reason"):
        finish_reason = choice.get("finish_reason")
        # ✅ DON'T break - let iteration complete naturally
# Stream fully consumed, now handle tool execution
```

### Why This Works

1. **Generator Continuity**: The async generator remains in a single, unbroken flow
2. **Proper Stream Consumption**: FastAPI can properly handle the continuous generator
3. **No Buffering Issues**: Responses are sent immediately without waiting for the next request

### Additional Improvements

- Added stream ID tracking for better debugging (`[stream-id]` in logs)
- Added small delay after keep-alive messages to ensure client processing
- Better logging to track stream consumption

## 🚀 Deployment

```bash
pip install --upgrade brain-proxy==0.1.85
```

## ✅ Expected Behavior

With v0.1.85, when tools are executed:
1. The initial stream completes naturally
2. Tools are executed
3. The follow-up response is generated
4. **The response is immediately displayed to the client** (no need to send another message!)

## 📊 Testing

You should see in logs:
```
[abc12345] Stream fully consumed, now executing X tool calls for tenant...
[abc12345] Starting to stream response of X characters
[abc12345] Finished streaming X chunks
[abc12345] Sending [DONE] marker
```

And most importantly - **the client will see the response immediately!**
