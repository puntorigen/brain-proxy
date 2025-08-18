# Changelog - Version 0.1.94

## Fixed: Recursive/Chained Tool Calls in Streaming Mode

### Issues Addressed:
1. **Fixed argument order bug** in `_execute_tool_calls` call within the streaming handler (line 1569)
   - Was: `self._execute_tool_calls(tenant, more_tool_calls, request.tools)`
   - Now: `self._execute_tool_calls(more_tool_calls, tenant, request.tools)`

2. **Enabled proper tool chaining** by including tools in follow-up LLM calls
   - Previously: Follow-up calls were made without tools, preventing the agent from executing multi-step plans
   - Now: Tools are included in follow-up calls with proper filtering, allowing the agent to chain multiple tool calls

3. **Increased max rounds** for tool chaining from 3 to 5
   - Allows more complex multi-step workflows

### Key Changes:
- Modified `_handle_streaming` method to pass tools in follow-up LLM calls
- Fixed recursive tool execution by ensuring tools are available throughout the chain
- Tools are now filtered and included in each follow-up round, enabling the agent to:
  - Execute multi-step plans autonomously
  - Chain tools based on previous results
  - Complete complex workflows without waiting for user confirmation between steps

### Impact:
Agents can now properly execute complex, multi-step tasks that require chaining multiple tools together. For example, an agent can now:
1. Search for information
2. Process/analyze the results
3. Store relevant findings
4. Take action based on the analysis

All in a single interaction, without requiring user prompts between each step.
