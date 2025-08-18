# Changelog for v0.1.84

## 🐛 Critical Bug Fixes for SSE Streaming After Tool Execution

### Problems Identified

1. **Missing Flag**: When `finish_reason == "tool_calls"`, we were breaking from the stream without setting `tool_calls_detected = True`, causing premature `[DONE]` messages
2. **SSE Connection Timeout**: No keep-alive messages during tool execution could cause client disconnection
3. **Stream Identification**: Multiple concurrent streams were hard to track in logs
4. **SSE Format Issues**: Yielding empty strings for flushing was breaking SSE format

## 🔧 Solutions Implemented

### 1. Fixed Tool Detection Flag
```python
if choice.get("finish_reason") == "tool_calls":
    tool_calls_detected = True  # CRITICAL: Set this flag!
```
- Ensures we don't send `[DONE]` prematurely when tools are detected

### 2. Added SSE Keep-Alive Messages
```python
# Send keep-alive comments during tool execution
yield f": Executing {len(tool_call_parts)} tool(s)...\n\n"
yield f": Processing tool results...\n\n"
```
- SSE comments (lines starting with `:`) maintain connection without affecting data
- Prevents client timeout during tool execution

### 3. Stream Session Tracking
```python
stream_id = str(uuid.uuid4())[:8]
self._log(f"[{stream_id}] Starting stream for tenant {tenant}")
```
- Each stream gets a unique ID for tracking in logs
- Helps identify concurrent stream issues

### 4. Removed Format-Breaking Empty Yields
- Removed `yield ""` that was meant to force flushing but broke SSE format
- SSE clients expect proper `data: ` prefixed lines

### 5. Enhanced Headers
- Already had proper cache control headers from v0.1.83
- Headers prevent proxy/server buffering

## 📊 Testing

With v0.1.84, you should see:
- `[stream-id] Starting stream for tenant...`
- `[stream-id] Proceeding to execute X tool calls...`
- SSE comments during tool execution
- `[stream-id] Starting to stream response of X characters`
- `[stream-id] Finished streaming X chunks`
- `[stream-id] Sending [DONE] marker`

The client should now properly receive and display responses after tool execution!

## 🚀 Deployment

```bash
pip install --upgrade brain-proxy==0.1.84
```

## 🔍 What This Fixes

The main issue was that after detecting tool calls, we were breaking the stream flow without properly maintaining the SSE connection. The client would either:
- Think the stream ended prematurely
- Timeout waiting for more data
- Not properly receive the follow-up response

Now the stream maintains continuity throughout the entire process: initial response → tool execution → follow-up response → completion.
