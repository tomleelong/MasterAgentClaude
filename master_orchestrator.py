"""
Master Agent Orchestrator for OpenAI Agents SDK

This module provides a comprehensive orchestrator for managing multiple OpenAI agents,
handling tool delegation, coordination, and resource management.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from contextlib import asynccontextmanager

try:
    from openai import AsyncOpenAI
    from openai.types.beta.threads import Message
    from openai.types.beta.assistants import Assistant
    from openai.types.beta.threads.runs import Run
except ImportError:
    raise ImportError(
        "OpenAI SDK not found. Install with: pip install openai"
    )


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TERMINATED = "terminated"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentConfig:
    """Configuration for creating specialized agents."""
    name: str
    description: str
    instructions: str
    tools: List[Dict[str, Any]] = field(default_factory=list)
    model: str = "gpt-4o"
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_completion_tokens: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    temperature: float = 0.7


@dataclass
class Task:
    """Represents a task to be executed by an agent."""
    id: str
    description: str
    agent_id: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: AgentStatus = AgentStatus.IDLE
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInstance:
    """Represents a managed agent instance."""
    id: str
    assistant_id: str
    thread_id: str
    config: AgentConfig
    status: AgentStatus = AgentStatus.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: Optional[datetime] = None
    current_task_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class MasterAgentOrchestrator:
    """
    Master Agent Orchestrator for managing multiple OpenAI agents.
    
    This class provides comprehensive agent management, tool delegation,
    and coordination capabilities for building complex multi-agent systems.
    """
    
    def __init__(
        self,
        api_key: str,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        max_concurrent_agents: int = 10,
        default_model: str = "gpt-4o",
        enable_logging: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize the Master Agent Orchestrator.
        
        Args:
            api_key: OpenAI API key
            organization: OpenAI organization ID
            project: OpenAI project ID
            max_concurrent_agents: Maximum number of concurrent agents
            default_model: Default model for new agents
            enable_logging: Whether to enable logging
            log_level: Logging level
        """
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            project=project
        )
        
        self.max_concurrent_agents = max_concurrent_agents
        self.default_model = default_model
        
        # Agent and task management
        self.agents: Dict[str, AgentInstance] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        
        # Tool registry
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.tool_functions: Dict[str, Callable] = {}
        
        # Coordination and synchronization
        self._execution_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        
        # Logging setup
        if enable_logging:
            self._setup_logging(log_level)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Master Agent Orchestrator initialized")
    
    def _setup_logging(self, log_level: int) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('master_orchestrator.log')
            ]
        )
    
    async def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any],
        required: Optional[List[str]] = None
    ) -> None:
        """
        Register a tool function for use by agents.
        
        Args:
            name: Tool name
            function: Tool function
            description: Tool description
            parameters: Tool parameters schema
            required: Required parameters
        """
        tool_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or []
                }
            }
        }
        
        self.available_tools[name] = tool_schema
        self.tool_functions[name] = function
        
        self.logger.info(f"Registered tool: {name}")
    
    async def create_agent(
        self,
        config: AgentConfig,
        tools: Optional[List[str]] = None
    ) -> str:
        """
        Create a new specialized agent.
        
        Args:
            config: Agent configuration
            tools: List of tool names to assign to agent
            
        Returns:
            Agent ID
        """
        if len(self.agents) >= self.max_concurrent_agents:
            raise RuntimeError(f"Maximum number of agents ({self.max_concurrent_agents}) reached")
        
        try:
            # Prepare tools for agent
            agent_tools = []
            if tools:
                for tool_name in tools:
                    if tool_name in self.available_tools:
                        agent_tools.append(self.available_tools[tool_name])
                    else:
                        self.logger.warning(f"Tool {tool_name} not found in registry")
            
            agent_tools.extend(config.tools)
            
            # Create OpenAI assistant
            assistant = await self.client.beta.assistants.create(
                name=config.name,
                description=config.description,
                instructions=config.instructions,
                tools=agent_tools,
                model=config.model,
                metadata=config.metadata,
                temperature=config.temperature
            )
            
            # Create thread for agent
            thread = await self.client.beta.threads.create()
            
            # Create agent instance
            agent_id = str(uuid.uuid4())
            agent_instance = AgentInstance(
                id=agent_id,
                assistant_id=assistant.id,
                thread_id=thread.id,
                config=config,
                status=AgentStatus.IDLE
            )
            
            self.agents[agent_id] = agent_instance
            
            self.logger.info(f"Created agent: {config.name} (ID: {agent_id})")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Failed to create agent {config.name}: {str(e)}")
            raise
    
    async def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent and clean up resources.
        
        Args:
            agent_id: Agent ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            self.logger.warning(f"Agent {agent_id} not found")
            return False
        
        try:
            agent = self.agents[agent_id]
            
            # Cancel any running task
            if agent.current_task_id:
                await self.cancel_task(agent.current_task_id)
            
            # Delete OpenAI resources
            await self.client.beta.assistants.delete(agent.assistant_id)
            await self.client.beta.threads.delete(agent.thread_id)
            
            # Remove from registry
            del self.agents[agent_id]
            
            self.logger.info(f"Deleted agent: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete agent {agent_id}: {str(e)}")
            return False
    
    async def create_task(
        self,
        description: str,
        agent_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new task for execution.
        
        Args:
            description: Task description
            agent_id: Specific agent ID (optional)
            priority: Task priority
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            description=description,
            agent_id=agent_id,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        
        # Add to queue if no specific agent assigned
        if not agent_id:
            self.task_queue.append(task)
            self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
        
        self.logger.info(f"Created task: {task_id}")
        return task_id
    
    async def execute_task(self, task_id: str) -> Any:
        """
        Execute a specific task.
        
        Args:
            task_id: Task ID to execute
            
        Returns:
            Task result
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        try:
            # Find available agent
            agent_id = task.agent_id or await self._find_available_agent()
            if not agent_id:
                raise RuntimeError("No available agents")
            
            agent = self.agents[agent_id]
            
            # Update task and agent status
            task.status = AgentStatus.RUNNING
            task.started_at = datetime.now()
            task.agent_id = agent_id
            
            agent.status = AgentStatus.RUNNING
            agent.current_task_id = task_id
            agent.last_activity = datetime.now()
            
            self.logger.info(f"Executing task {task_id} on agent {agent_id}")
            
            # Send message to agent thread
            await self.client.beta.threads.messages.create(
                thread_id=agent.thread_id,
                role="user",
                content=task.description
            )
            
            # Create and execute run
            run = await self.client.beta.threads.runs.create(
                thread_id=agent.thread_id,
                assistant_id=agent.assistant_id
            )
            
            # Wait for completion and handle tool calls
            result = await self._handle_run_execution(run, agent, task)
            
            # Update task completion
            task.status = AgentStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Update agent status
            agent.status = AgentStatus.IDLE
            agent.current_task_id = None
            
            self.logger.info(f"Task {task_id} completed successfully")
            return result
            
        except Exception as e:
            task.status = AgentStatus.ERROR
            task.error = str(e)
            
            if task.agent_id and task.agent_id in self.agents:
                self.agents[task.agent_id].status = AgentStatus.ERROR
                self.agents[task.agent_id].current_task_id = None
            
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            raise
    
    async def _handle_run_execution(
        self,
        run: Run,
        agent: AgentInstance,
        task: Task
    ) -> str:
        """Handle run execution with tool calls."""
        while True:
            run = await self.client.beta.threads.runs.retrieve(
                thread_id=agent.thread_id,
                run_id=run.id
            )
            
            if run.status == "completed":
                # Get the latest message
                messages = await self.client.beta.threads.messages.list(
                    thread_id=agent.thread_id,
                    limit=1
                )
                
                if messages.data:
                    content = messages.data[0].content[0]
                    if hasattr(content, 'text'):
                        return content.text.value
                
                return "Task completed"
                
            elif run.status == "requires_action":
                # Handle tool calls
                tool_outputs = []
                
                if run.required_action and run.required_action.submit_tool_outputs:
                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        try:
                            function_name = tool_call.function.name
                            arguments = json.loads(tool_call.function.arguments)
                            
                            if function_name in self.tool_functions:
                                result = await self._execute_tool_function(
                                    function_name,
                                    arguments
                                )
                                
                                tool_outputs.append({
                                    "tool_call_id": tool_call.id,
                                    "output": str(result)
                                })
                            else:
                                tool_outputs.append({
                                    "tool_call_id": tool_call.id,
                                    "output": f"Error: Function {function_name} not found"
                                })
                                
                        except Exception as e:
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": f"Error: {str(e)}"
                            })
                
                # Submit tool outputs
                run = await self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=agent.thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                
            elif run.status in ["failed", "cancelled", "expired"]:
                raise RuntimeError(f"Run failed with status: {run.status}")
            
            # Wait before next check
            await asyncio.sleep(1)
    
    async def _execute_tool_function(
        self,
        function_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute a tool function safely."""
        try:
            function = self.tool_functions[function_name]
            
            # Handle async functions
            if asyncio.iscoroutinefunction(function):
                return await function(**arguments)
            else:
                return function(**arguments)
                
        except Exception as e:
            self.logger.error(f"Tool function {function_name} failed: {str(e)}")
            raise
    
    async def _find_available_agent(self) -> Optional[str]:
        """Find an available agent for task execution."""
        for agent_id, agent in self.agents.items():
            if agent.status == AgentStatus.IDLE:
                return agent_id
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        try:
            if task.agent_id and task.agent_id in self.agents:
                agent = self.agents[task.agent_id]
                agent.status = AgentStatus.IDLE
                agent.current_task_id = None
            
            task.status = AgentStatus.TERMINATED
            task.completed_at = datetime.now()
            
            # Remove from queue if present
            self.task_queue = [t for t in self.task_queue if t.id != task_id]
            
            self.logger.info(f"Cancelled task: {task_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel task {task_id}: {str(e)}")
            return False
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent status and metrics."""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return {
            "id": agent.id,
            "status": agent.status.value,
            "config": {
                "name": agent.config.name,
                "description": agent.config.description,
                "model": agent.config.model
            },
            "created_at": agent.created_at.isoformat(),
            "last_activity": agent.last_activity.isoformat() if agent.last_activity else None,
            "current_task_id": agent.current_task_id,
            "metrics": agent.metrics
        }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and details."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            "id": task.id,
            "description": task.description,
            "status": task.status.value,
            "priority": task.priority.value,
            "agent_id": task.agent_id,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result,
            "error": task.error,
            "metadata": task.metadata
        }
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all managed agents."""
        return [await self.get_agent_status(agent_id) for agent_id in self.agents.keys()]
    
    async def list_tasks(
        self,
        status_filter: Optional[AgentStatus] = None
    ) -> List[Dict[str, Any]]:
        """List all tasks with optional status filter."""
        tasks = []
        for task_id, task in self.tasks.items():
            if status_filter is None or task.status == status_filter:
                task_info = await self.get_task_status(task_id)
                if task_info:
                    tasks.append(task_info)
        return tasks
    
    async def process_task_queue(self) -> None:
        """Process pending tasks in the queue."""
        async with self._execution_lock:
            while self.task_queue and not self._shutdown_event.is_set():
                task = self.task_queue.pop(0)
                
                if task.status == AgentStatus.IDLE:
                    try:
                        await self.execute_task(task.id)
                    except Exception as e:
                        self.logger.error(f"Failed to process task {task.id}: {str(e)}")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
    
    async def start_queue_processor(self) -> None:
        """Start the task queue processor."""
        while not self._shutdown_event.is_set():
            await self.process_task_queue()
            await asyncio.sleep(1)
    
    @asynccontextmanager
    async def managed_execution(self):
        """Context manager for managed execution lifecycle."""
        try:
            # Start queue processor
            processor_task = asyncio.create_task(self.start_queue_processor())
            yield self
        finally:
            # Shutdown
            self._shutdown_event.set()
            processor_task.cancel()
            
            # Wait for current operations to complete
            try:
                await asyncio.wait_for(processor_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Queue processor shutdown timeout")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        self.logger.info("Shutting down Master Agent Orchestrator")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel all running tasks
        running_tasks = [
            task_id for task_id, task in self.tasks.items()
            if task.status == AgentStatus.RUNNING
        ]
        
        for task_id in running_tasks:
            await self.cancel_task(task_id)
        
        # Clean up agents
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            await self.delete_agent(agent_id)
        
        self.logger.info("Shutdown complete")
    
    def __repr__(self) -> str:
        return (
            f"MasterAgentOrchestrator("
            f"agents={len(self.agents)}, "
            f"tasks={len(self.tasks)}, "
            f"tools={len(self.available_tools)})"
        )