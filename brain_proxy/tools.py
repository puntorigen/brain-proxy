"""Tool registration and handling system for BrainProxy."""

import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, get_type_hints

class ToolRegistry:
    """Registry for storing tool definitions and their implementations."""
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._implementations: Dict[str, Callable] = {}

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], implementation: Callable):
        """Register a tool with its schema and implementation."""
        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self._implementations[name] = implementation

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all registered tool definitions."""
        return list(self._tools.values())

    def get_implementation(self, name: str) -> Optional[Callable]:
        """Get the implementation for a tool by name."""
        return self._implementations.get(name)

def _build_parameters_schema(func: Callable) -> Dict[str, Any]:
    """Build OpenAI function parameters schema from function signature."""
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse docstring for parameter descriptions
    param_desc = {}
    if doc:
        lines = doc.split("\n")
        in_args = False
        current_param = None
        desc_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("Args:"):
                in_args = True
                continue
            elif in_args and line.startswith("Returns:"):
                in_args = False
                if current_param:
                    param_desc[current_param] = " ".join(desc_lines).strip()
            elif in_args:
                if line and not line.startswith(" "):
                    # Save previous param if any
                    if current_param:
                        param_desc[current_param] = " ".join(desc_lines).strip()
                    # New parameter
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        current_param = parts[0].strip()
                        desc_lines = [parts[1].strip()]
                elif line and current_param:
                    desc_lines.append(line)
    
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        if name == 'self':  # Skip self for class methods
            continue
            
        param_type = hints.get(name, Any)
        type_name = getattr(param_type, "__name__", str(param_type))
        
        # Map Python types to JSON Schema types
        type_map = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object"
        }
        
        properties[name] = {
            "type": type_map.get(type_name, "string"),
            "description": param_desc.get(name, "")
        }
        
        if param.default is inspect.Parameter.empty:
            required.append(name)
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }

_registry = ToolRegistry()

def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to register a function as a tool.
    
    Args:
        name: Optional name for the tool. If not provided, uses the function name.
        description: Optional description. If not provided, uses the function's docstring.
    """
    def decorator(func: Callable):
        nonlocal name, description
        tool_name = name or func.__name__
        # Get description from docstring's first line if not provided
        if not description and func.__doc__:
            description = func.__doc__.split('\n')[0].strip()
        tool_description = description or f"Execute {tool_name}"
        
        # Build parameters schema from function signature
        parameters = _build_parameters_schema(func)
        
        # Register the tool
        _registry.register_tool(tool_name, tool_description, parameters, func)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)
        
        return wrapper
    
    # Handle using the decorator without parentheses
    if callable(name):
        func, name = name, None
        return decorator(func)
    
    return decorator

def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry
