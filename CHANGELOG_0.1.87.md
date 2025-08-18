# Changelog for v0.1.87

## 🐛 Fix for Missing Responses After Tool Execution

### The Problem
After tool execution in v0.1.86, responses were not being displayed at all, even though the server was successfully generating and attempting to send them.

### Root Cause
The SSE streaming format was incorrect. We were sending two separate chunks:
1. Content chunk with `finish_reason: None`
2. Empty chunk with `finish_reason: "stop"`

This double-chunk pattern was confusing the OpenAI-compatible SSE client, causing it to not display the content.

### The Solution
Following the OpenAI streaming format more closely, we now send the content and `finish_reason` in a **single chunk**:

**Before (v0.1.86 - broken):**
```javascript
// Two separate chunks
{ delta: { content: "response text" }, finish_reason: null }  // Chunk 1
{ delta: {}, finish_reason: "stop" }                          // Chunk 2
```

**After (v0.1.87 - fixed):**
```javascript
// Single chunk with both content and finish_reason
{ delta: { content: "response text" }, finish_reason: "stop" }
```

### Why This Works
- **Proper SSE Format**: Follows the OpenAI streaming standard exactly
- **Single Chunk**: No confusion from multiple chunks
- **Client Compatibility**: The client properly recognizes and displays the response

## 🚀 Deployment
```bash
pip install --upgrade brain-proxy==0.1.87
```

## ✅ Expected Behavior
With v0.1.87, after tool execution:
- The complete response will be displayed immediately
- No missing or silent responses
- Proper SSE format ensures client compatibility

## 📊 Testing
You should see in logs:
```
[stream-id] Sending complete response with finish_reason=stop in single chunk
[stream-id] Response sent successfully
```

And most importantly - **the response will actually appear in the client!**
