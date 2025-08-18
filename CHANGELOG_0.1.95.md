# Changelog - Version 0.1.95

## Fixed: Tool Call Format Handling in Recursive Execution

### Issue:
When the LLM returned tool calls in follow-up responses for chaining, the code was failing with:
```
'dict' object has no attribute 'function'
```

### Root Cause:
The tool calls from follow-up LLM responses were being returned in a different format (potentially as dictionaries) than expected by the `_execute_tool_calls` method, which expects objects with a `function` attribute containing `name` and `arguments`.

### Solution:
Added proper format handling in the recursive tool execution loop:
1. Check if tool calls are in dict or object format
2. Convert them to a consistent dict format
3. Create mock objects with the expected structure for `_execute_tool_calls`
4. This ensures compatibility regardless of how the LLM returns the tool calls

### Code Changes:
- Modified the tool call processing in `_handle_streaming` method (lines 1570-1599)
- Added `MockToolCall` class to wrap tool call dictionaries with the expected object interface
- Handles both dictionary and object formats gracefully

### Impact:
Recursive/chained tool calls now work properly without format errors, allowing agents to execute multi-step plans that involve multiple sequential tool calls.
