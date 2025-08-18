# Changelog for v0.1.91

## 🚨 CRITICAL FIXES: Async Blocking & Recursive Tools

### Three Major Issues Fixed

#### 1. **Async Generator Blocking**
The memory storage was blocking the async generator from completing properly:

```python
# Before (BLOCKING):
await self.memory_service.store(...)  # Inside generator - blocks completion!

# After (NON-BLOCKING):
asyncio.create_task(self.memory_service.store(...))  # Scheduled, doesn't block
```

This was causing the stream to hang, especially with slower tools like `get_user_weather`.

#### 2. **Recursive Tool Calls Not Working**
When the LLM wanted to call multiple tools in sequence, we were ignoring the additional tool calls:

```python
# Before: Ignored recursive tool calls
if message.tool_calls:
    # Just forced a text response - tools not executed!

# After: Properly handle recursive tools
if message.tool_calls:
    # Execute the additional tools
    more_results = await self._execute_tool_calls(...)
    # Make another follow-up call with all results
    # Now the agent can chain tools together!
```

This fixes the issue where the agent says "consultaré el clima" but doesn't actually call the tool.

#### 3. **More Frequent Keep-Alive Messages**
Added additional keep-alive chunks during tool execution to prevent timeouts:

```python
# Send keep-alive before tool execution
yield keep_alive_chunk
# Execute tool
result = await execute_tool(...)  
# Send keep-alive after tool execution
yield keep_alive_chunk
```

## 🎯 What These Fix

1. **Weather tool responses now display** - No more silent agent after weather
2. **Tools can chain together** - Agent can call multiple tools in sequence
3. **No more async blocking** - Stream completes properly
4. **Better timeout prevention** - More keep-alives during slow operations

## 🚀 Deployment
```bash
pip install --upgrade brain-proxy==0.1.91
```

## ✅ Expected Behavior

With v0.1.91:
- **Weather tool works!** Response displays immediately
- **Recursive tools work!** Agent can chain multiple tools
- **No more hanging!** Async operations don't block
- **Session stays alive!** Better keep-alive coverage

## 📊 Testing

Test scenarios:
1. **Weather**: "What's the weather?" → Should display response
2. **Recursive**: "Check weather then search for umbrellas" → Should execute both
3. **Session continuity**: Multiple tool calls → All should work

## 🔍 Technical Details

The main issue was that slower tools (like weather with 2 HTTP calls) were exposing timing issues in our async flow:
- Memory storage was blocking generator completion
- Recursive tool calls were being ignored
- Not enough keep-alives during long operations

All three issues are now fixed!
