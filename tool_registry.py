"""
Tool Registry and Management System for Agent Orchestrator.

This module provides comprehensive tool management, including registration,
delegation, access control, and coordination between agents.
"""

import asyncio
import inspect
import logging
from typing import Dict, List, Any, Callable, Optional, Set, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from functools import wraps

# Type variable for tool functions
T = TypeVar('T')


class ToolAccessLevel(Enum):
    """Tool access levels for agents."""
    PUBLIC = "public"          # Available to all agents
    RESTRICTED = "restricted"  # Requires explicit permission
    ADMIN = "admin"           # Only for master orchestrator
    PRIVATE = "private"       # Agent-specific tools


class ToolCategory(Enum):
    """Tool categories for organization."""
    DATA_PROCESSING = "data_processing"
    FILE_OPERATIONS = "file_operations"
    WEB_SERVICES = "web_services"
    DATABASE = "database"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    SECURITY = "security"
    MONITORING = "monitoring"
    CUSTOM = "custom"


@dataclass
class ToolMetrics:
    """Metrics tracking for tool usage."""
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_used: Optional[datetime] = None
    last_error: Optional[str] = None


@dataclass
class ToolDefinition:
    """Complete tool definition with metadata."""
    name: str
    function: Callable
    description: str
    parameters: Dict[str, Any]
    required_params: List[str] = field(default_factory=list)
    access_level: ToolAccessLevel = ToolAccessLevel.PUBLIC
    category: ToolCategory = ToolCategory.CUSTOM
    version: str = "1.0.0"
    author: Optional[str] = None
    requires_auth: bool = False
    rate_limit: Optional[int] = None  # Calls per minute
    timeout: Optional[float] = None   # Timeout in seconds
    allowed_agents: Optional[Set[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: ToolMetrics = field(default_factory=ToolMetrics)


class ToolRegistry:
    """
    Comprehensive tool registry and management system.
    
    Provides tool registration, access control, usage tracking,
    and coordination capabilities for the agent orchestrator.
    """
    
    def __init__(self, enable_metrics: bool = True):
        """
        Initialize the tool registry.
        
        Args:
            enable_metrics: Whether to track tool usage metrics
        """
        self.tools: Dict[str, ToolDefinition] = {}
        self.enable_metrics = enable_metrics
        
        # Access control
        self.agent_permissions: Dict[str, Set[str]] = {}  # agent_id -> tool_names
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, List[datetime]]] = {}  # tool_name -> agent_id -> timestamps
        
        # Tool categories index
        self.categories: Dict[ToolCategory, Set[str]] = {
            category: set() for category in ToolCategory
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Register built-in tools
        asyncio.create_task(self._register_builtin_tools())
    
    async def _register_builtin_tools(self) -> None:
        """Register built-in utility tools."""
        
        # File operations
        await self.register_tool(
            name="read_file",
            function=self._read_file,
            description="Read contents of a file",
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8"
                }
            },
            required_params=["file_path"],
            category=ToolCategory.FILE_OPERATIONS,
            access_level=ToolAccessLevel.RESTRICTED
        )
        
        await self.register_tool(
            name="write_file",
            function=self._write_file,
            description="Write content to a file",
            parameters={
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8"
                }
            },
            required_params=["file_path", "content"],
            category=ToolCategory.FILE_OPERATIONS,
            access_level=ToolAccessLevel.RESTRICTED
        )
        
        # Data processing
        await self.register_tool(
            name="json_parse",
            function=self._json_parse,
            description="Parse JSON string into dictionary",
            parameters={
                "json_string": {
                    "type": "string",
                    "description": "JSON string to parse"
                }
            },
            required_params=["json_string"],
            category=ToolCategory.DATA_PROCESSING,
            access_level=ToolAccessLevel.PUBLIC
        )
        
        await self.register_tool(
            name="json_stringify",
            function=self._json_stringify,
            description="Convert dictionary to JSON string",
            parameters={
                "data": {
                    "type": "object",
                    "description": "Data to convert to JSON"
                },
                "indent": {
                    "type": "integer",
                    "description": "JSON indentation (default: 2)",
                    "default": 2
                }
            },
            required_params=["data"],
            category=ToolCategory.DATA_PROCESSING,
            access_level=ToolAccessLevel.PUBLIC
        )
        
        # Web services
        await self.register_tool(
            name="http_request",
            function=self._http_request,
            description="Make HTTP request to external service",
            parameters={
                "url": {
                    "type": "string",
                    "description": "URL to request"
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method (GET, POST, PUT, DELETE)",
                    "default": "GET"
                },
                "headers": {
                    "type": "object",
                    "description": "Request headers"
                },
                "data": {
                    "type": "object",
                    "description": "Request data/payload"
                }
            },
            required_params=["url"],
            category=ToolCategory.WEB_SERVICES,
            access_level=ToolAccessLevel.RESTRICTED,
            rate_limit=60,  # 60 calls per minute
            timeout=30.0
        )
    
    async def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any],
        required_params: Optional[List[str]] = None,
        access_level: ToolAccessLevel = ToolAccessLevel.PUBLIC,
        category: ToolCategory = ToolCategory.CUSTOM,
        **kwargs
    ) -> bool:
        """
        Register a new tool in the registry.
        
        Args:
            name: Tool name (must be unique)
            function: Tool function
            description: Tool description
            parameters: Parameter schema
            required_params: List of required parameter names
            access_level: Tool access level
            category: Tool category
            **kwargs: Additional tool metadata
            
        Returns:
            True if registration successful
        """
        if name in self.tools:
            self.logger.warning(f"Tool {name} already exists, updating...")
        
        try:
            # Validate function signature
            sig = inspect.signature(function)
            
            # Create tool definition
            tool_def = ToolDefinition(
                name=name,
                function=function,
                description=description,
                parameters=parameters,
                required_params=required_params or [],
                access_level=access_level,
                category=category,
                **kwargs
            )
            
            # Wrap function with metrics collection if enabled
            if self.enable_metrics:
                tool_def.function = self._wrap_with_metrics(tool_def.function, name)
            
            # Store in registry
            self.tools[name] = tool_def
            self.categories[category].add(name)
            
            self.logger.info(f"Registered tool: {name} ({category.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool {name}: {str(e)}")
            return False
    
    def _wrap_with_metrics(self, func: Callable, tool_name: str) -> Callable:
        """Wrap function with metrics collection."""
        @wraps(func)
        async def metrics_wrapper(*args, **kwargs):
            if tool_name not in self.tools:
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            tool_def = self.tools[tool_name]
            start_time = datetime.now()
            
            try:
                tool_def.metrics.call_count += 1
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                tool_def.metrics.success_count += 1
                return result
                
            except Exception as e:
                tool_def.metrics.error_count += 1
                tool_def.metrics.last_error = str(e)
                raise
                
            finally:
                # Update timing metrics
                execution_time = (datetime.now() - start_time).total_seconds()
                tool_def.metrics.total_execution_time += execution_time
                tool_def.metrics.average_execution_time = (
                    tool_def.metrics.total_execution_time / tool_def.metrics.call_count
                )
                tool_def.metrics.last_used = datetime.now()
        
        return metrics_wrapper
    
    async def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            name: Tool name to remove
            
        Returns:
            True if successful
        """
        if name not in self.tools:
            self.logger.warning(f"Tool {name} not found")
            return False
        
        try:
            tool_def = self.tools[name]
            
            # Remove from category index
            self.categories[tool_def.category].discard(name)
            
            # Remove from agent permissions
            for agent_id in self.agent_permissions:
                self.agent_permissions[agent_id].discard(name)
            
            # Remove from rate limits
            if name in self.rate_limits:
                del self.rate_limits[name]
            
            # Remove tool
            del self.tools[name]
            
            self.logger.info(f"Unregistered tool: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister tool {name}: {str(e)}")
            return False
    
    async def grant_tool_access(self, agent_id: str, tool_names: Union[str, List[str]]) -> bool:
        """
        Grant tool access to an agent.
        
        Args:
            agent_id: Agent ID
            tool_names: Tool name or list of tool names
            
        Returns:
            True if successful
        """
        if isinstance(tool_names, str):
            tool_names = [tool_names]
        
        if agent_id not in self.agent_permissions:
            self.agent_permissions[agent_id] = set()
        
        granted = []
        for tool_name in tool_names:
            if tool_name in self.tools:
                self.agent_permissions[agent_id].add(tool_name)
                granted.append(tool_name)
            else:
                self.logger.warning(f"Tool {tool_name} not found for access grant")
        
        if granted:
            self.logger.info(f"Granted tools {granted} to agent {agent_id}")
            return True
        return False
    
    async def revoke_tool_access(self, agent_id: str, tool_names: Union[str, List[str]]) -> bool:
        """
        Revoke tool access from an agent.
        
        Args:
            agent_id: Agent ID
            tool_names: Tool name or list of tool names
            
        Returns:
            True if successful
        """
        if isinstance(tool_names, str):
            tool_names = [tool_names]
        
        if agent_id not in self.agent_permissions:
            return False
        
        revoked = []
        for tool_name in tool_names:
            if tool_name in self.agent_permissions[agent_id]:
                self.agent_permissions[agent_id].discard(tool_name)
                revoked.append(tool_name)
        
        if revoked:
            self.logger.info(f"Revoked tools {revoked} from agent {agent_id}")
            return True
        return False
    
    def can_agent_use_tool(self, agent_id: str, tool_name: str) -> bool:
        """
        Check if an agent can use a specific tool.
        
        Args:
            agent_id: Agent ID
            tool_name: Tool name
            
        Returns:
            True if agent can use tool
        """
        if tool_name not in self.tools:
            return False
        
        tool_def = self.tools[tool_name]
        
        # Check access level
        if tool_def.access_level == ToolAccessLevel.PUBLIC:
            return True
        elif tool_def.access_level == ToolAccessLevel.ADMIN:
            return False  # Only master orchestrator
        elif tool_def.access_level == ToolAccessLevel.RESTRICTED:
            return agent_id in self.agent_permissions and tool_name in self.agent_permissions[agent_id]
        elif tool_def.access_level == ToolAccessLevel.PRIVATE:
            return (tool_def.allowed_agents and agent_id in tool_def.allowed_agents) or \
                   (agent_id in self.agent_permissions and tool_name in self.agent_permissions[agent_id])
        
        return False
    
    async def execute_tool(
        self,
        tool_name: str,
        agent_id: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Execute a tool function with access control and rate limiting.
        
        Args:
            tool_name: Tool to execute
            agent_id: Agent requesting execution
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        # Check if tool exists
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool_def = self.tools[tool_name]
        
        # Check access permissions
        if not self.can_agent_use_tool(agent_id, tool_name):
            raise PermissionError(f"Agent {agent_id} does not have access to tool {tool_name}")
        
        # Check rate limiting
        if tool_def.rate_limit and not await self._check_rate_limit(tool_name, agent_id, tool_def.rate_limit):
            raise RuntimeError(f"Rate limit exceeded for tool {tool_name}")
        
        # Validate required parameters
        missing_params = []
        for param in tool_def.required_params:
            if param not in arguments:
                missing_params.append(param)
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        try:
            # Execute tool with timeout if specified
            if tool_def.timeout:
                result = await asyncio.wait_for(
                    self._execute_tool_function(tool_def.function, arguments),
                    timeout=tool_def.timeout
                )
            else:
                result = await self._execute_tool_function(tool_def.function, arguments)
            
            return result
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Tool {tool_name} execution timed out")
        except Exception as e:
            self.logger.error(f"Tool {tool_name} execution failed: {str(e)}")
            raise
    
    async def _execute_tool_function(self, function: Callable, arguments: Dict[str, Any]) -> Any:
        """Execute tool function handling async/sync."""
        if asyncio.iscoroutinefunction(function):
            return await function(**arguments)
        else:
            return function(**arguments)
    
    async def _check_rate_limit(self, tool_name: str, agent_id: str, limit: int) -> bool:
        """Check if agent is within rate limit for tool."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Initialize rate limit tracking
        if tool_name not in self.rate_limits:
            self.rate_limits[tool_name] = {}
        if agent_id not in self.rate_limits[tool_name]:
            self.rate_limits[tool_name][agent_id] = []
        
        # Clean old timestamps
        timestamps = self.rate_limits[tool_name][agent_id]
        self.rate_limits[tool_name][agent_id] = [
            ts for ts in timestamps if ts > minute_ago
        ]
        
        # Check if within limit
        if len(self.rate_limits[tool_name][agent_id]) >= limit:
            return False
        
        # Add current timestamp
        self.rate_limits[tool_name][agent_id].append(now)
        return True
    
    def get_available_tools(
        self,
        agent_id: Optional[str] = None,
        category: Optional[ToolCategory] = None,
        access_level: Optional[ToolAccessLevel] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of available tools with optional filtering.
        
        Args:
            agent_id: Filter by agent access permissions
            category: Filter by tool category
            access_level: Filter by access level
            
        Returns:
            List of tool information
        """
        tools = []
        
        for tool_name, tool_def in self.tools.items():
            # Apply filters
            if category and tool_def.category != category:
                continue
            if access_level and tool_def.access_level != access_level:
                continue
            if agent_id and not self.can_agent_use_tool(agent_id, tool_name):
                continue
            
            # Create tool info
            tool_info = {
                "name": tool_name,
                "description": tool_def.description,
                "category": tool_def.category.value,
                "access_level": tool_def.access_level.value,
                "parameters": tool_def.parameters,
                "required_params": tool_def.required_params,
                "version": tool_def.version
            }
            
            # Add metrics if enabled
            if self.enable_metrics:
                tool_info["metrics"] = {
                    "call_count": tool_def.metrics.call_count,
                    "success_rate": (
                        tool_def.metrics.success_count / max(tool_def.metrics.call_count, 1)
                    ),
                    "average_execution_time": tool_def.metrics.average_execution_time,
                    "last_used": tool_def.metrics.last_used.isoformat() if tool_def.metrics.last_used else None
                }
            
            tools.append(tool_info)
        
        return tools
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get OpenAI tool schema for a specific tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            OpenAI tool schema or None if not found
        """
        if tool_name not in self.tools:
            return None
        
        tool_def = self.tools[tool_name]
        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_def.description,
                "parameters": {
                    "type": "object",
                    "properties": tool_def.parameters,
                    "required": tool_def.required_params
                }
            }
        }
    
    def get_agent_tools_schema(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get OpenAI tool schemas for all tools available to an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of OpenAI tool schemas
        """
        schemas = []
        for tool_name in self.tools:
            if self.can_agent_use_tool(agent_id, tool_name):
                schema = self.get_tool_schema(tool_name)
                if schema:
                    schemas.append(schema)
        return schemas
    
    def get_tool_metrics(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a tool."""
        if tool_name not in self.tools or not self.enable_metrics:
            return None
        
        metrics = self.tools[tool_name].metrics
        return {
            "call_count": metrics.call_count,
            "success_count": metrics.success_count,
            "error_count": metrics.error_count,
            "success_rate": metrics.success_count / max(metrics.call_count, 1),
            "error_rate": metrics.error_count / max(metrics.call_count, 1),
            "total_execution_time": metrics.total_execution_time,
            "average_execution_time": metrics.average_execution_time,
            "last_used": metrics.last_used.isoformat() if metrics.last_used else None,
            "last_error": metrics.last_error
        }
    
    # Built-in tool implementations
    async def _read_file(self, file_path: str, encoding: str = "utf-8") -> str:
        """Read file contents."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read file {file_path}: {str(e)}")
    
    async def _write_file(self, file_path: str, content: str, encoding: str = "utf-8") -> str:
        """Write content to file."""
        try:
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            raise RuntimeError(f"Failed to write file {file_path}: {str(e)}")
    
    async def _json_parse(self, json_string: str) -> Dict[str, Any]:
        """Parse JSON string."""
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}")
    
    async def _json_stringify(self, data: Any, indent: int = 2) -> str:
        """Convert data to JSON string."""
        try:
            return json.dumps(data, indent=indent, ensure_ascii=False)
        except TypeError as e:
            raise ValueError(f"Data not JSON serializable: {str(e)}")
    
    async def _http_request(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Make HTTP request (requires aiohttp)."""
        try:
            import aiohttp
        except ImportError:
            raise RuntimeError("aiohttp package required for HTTP requests")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method.upper(),
                    url=url,
                    headers=headers,
                    json=data if method.upper() in ['POST', 'PUT', 'PATCH'] else None
                ) as response:
                    result = {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "url": str(response.url)
                    }
                    
                    # Try to parse as JSON, fall back to text
                    try:
                        result["data"] = await response.json()
                    except:
                        result["data"] = await response.text()
                    
                    return result
                    
        except Exception as e:
            raise RuntimeError(f"HTTP request failed: {str(e)}")


# Decorator for easy tool registration
def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: ToolCategory = ToolCategory.CUSTOM,
    access_level: ToolAccessLevel = ToolAccessLevel.PUBLIC,
    **kwargs
):
    """
    Decorator for registering tool functions.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description
        category: Tool category
        access_level: Access level
        **kwargs: Additional tool metadata
    """
    def decorator(func):
        func._tool_name = name or func.__name__
        func._tool_description = description or func.__doc__ or "No description available"
        func._tool_category = category
        func._tool_access_level = access_level
        func._tool_kwargs = kwargs
        
        # Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = {}
        required_params = []
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # Default type
            
            if param.annotation != inspect.Parameter.empty:
                # Map Python types to JSON schema types
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object"
                }
                param_info["type"] = type_map.get(param.annotation, "string")
            
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
            else:
                param_info["default"] = param.default
            
            parameters[param_name] = param_info
        
        func._tool_parameters = parameters
        func._tool_required_params = required_params
        
        return func
    
    return decorator