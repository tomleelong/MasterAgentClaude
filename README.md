# Master Agent Orchestrator

A comprehensive, production-ready orchestrator for managing multiple OpenAI agents with advanced tool delegation, coordination, and error handling capabilities.

## Features

### 🎯 Core Capabilities
- **Multi-Agent Management**: Create, monitor, and coordinate multiple specialized agents
- **Dynamic Agent Creation**: Generate agents with predefined templates or custom configurations
- **Tool Registry & Delegation**: Comprehensive tool management with access control and rate limiting
- **Task Orchestration**: Queue and execute tasks across multiple agents with priority handling
- **Error Handling & Recovery**: Circuit breakers, automatic retry mechanisms, and comprehensive error tracking

### 🛠 Advanced Features
- **Built-in Agent Templates**: Data analysts, code reviewers, system architects, and more
- **Parallel Task Execution**: Execute multiple tasks concurrently across different agents
- **Resource Monitoring**: Track agent performance, tool usage, and system metrics  
- **Configurable Environments**: Support for development, staging, and production deployments
- **Comprehensive Logging**: Structured logging with multiple output formats

### 🔒 Production Ready
- **Security**: Authentication, CORS support, rate limiting, and input validation
- **Monitoring**: Health checks, metrics collection, and observability integration
- **Scalability**: Connection pooling, resource management, and performance optimization
- **Reliability**: Circuit breakers, graceful degradation, and automatic recovery

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/master-agent-orchestrator
cd master-agent-orchestrator

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Update `.env` with your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Basic Usage

```python
import asyncio
from master_orchestrator import MasterAgentOrchestrator
from agent_factory import AgentFactory
from config import MasterOrchestratorConfig

async def main():
    # Load configuration
    config = MasterOrchestratorConfig.from_env()
    
    # Initialize orchestrator
    orchestrator = MasterAgentOrchestrator(
        api_key=config.openai.api_key,
        max_concurrent_agents=config.orchestrator.max_concurrent_agents
    )
    
    # Create agent factory
    factory = AgentFactory(orchestrator)
    
    # Create a data analysis team
    team = await factory.create_data_analysis_team(
        tools=["calculate_metrics", "generate_report"]
    )
    
    # Create and execute a task
    task_id = await orchestrator.create_task(
        description="Analyze the dataset [1, 2, 3, 4, 5] and generate a report",
        agent_id=team["data_analyst"]
    )
    
    result = await orchestrator.execute_task(task_id)
    print(f"Analysis result: {result}")
    
    # Cleanup
    await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture Overview

### Core Components

#### 1. Master Agent Orchestrator (`master_orchestrator.py`)
The central coordination hub that manages:
- Agent lifecycle (creation, monitoring, deletion)
- Task queue and execution
- Resource allocation and cleanup
- Integration with other components

```python
# Key methods
async def create_agent(config, tools) -> str
async def execute_task(task_id) -> Any
async def list_agents() -> List[Dict]
```

#### 2. Agent Factory (`agent_factory.py`)
Provides templates and factory methods for creating specialized agents:

```python
# Built-in templates
- Data Analyst Agent
- Code Reviewer Agent  
- System Architect Agent
- Testing Specialist Agent
- Documentation Writer Agent
- Project Manager Agent

# Usage
factory = AgentFactory(orchestrator)
team = await factory.create_development_team()
```

#### 3. Tool Registry (`tool_registry.py`)
Manages tool registration, access control, and execution:

```python
# Register custom tools
@tool(name="my_tool", category=ToolCategory.ANALYSIS)
async def my_custom_tool(data: str) -> str:
    return f"Processed: {data}"

# Tool access control
await registry.grant_tool_access(agent_id, ["my_tool"])
```

#### 4. Error Handling (`error_handling.py`)
Comprehensive error management with:
- Error classification and severity assessment
- Circuit breakers for failure isolation
- Automatic retry mechanisms with exponential backoff
- Recovery strategies (retry, fallback, escalate, etc.)

#### 5. Configuration Management (`config.py`)
Centralized configuration with environment variable support:

```python
# Load from environment
config = MasterOrchestratorConfig.from_env()

# Load from file
config = MasterOrchestratorConfig.from_file("config.yaml")

# Validation
issues = config.validate()
```

## Examples and Use Cases

### 1. Data Analysis Workflow

```python
async def data_analysis_example():
    # Create specialized data analysis team
    team = await factory.create_data_analysis_team()
    
    # Analyze multiple datasets in parallel
    datasets = [[1,2,3,4,5], [10,20,30], [100,200,300,400]]
    
    tasks = []
    for i, dataset in enumerate(datasets):
        task_id = await orchestrator.create_task(
            description=f"Analyze dataset {i+1}: {dataset}",
            agent_id=team["data_analyst"]
        )
        tasks.append(task_id)
    
    # Execute all tasks concurrently
    results = await asyncio.gather(
        *[orchestrator.execute_task(task_id) for task_id in tasks]
    )
    
    return results
```

### 2. Code Review Pipeline

```python
async def code_review_example():
    # Create development team
    team = await factory.create_development_team()
    
    # Multi-stage review process
    code_review_task = await orchestrator.create_task(
        description="Review this Python function for bugs and style issues",
        agent_id=team["code_reviewer"]
    )
    
    architecture_review_task = await orchestrator.create_task(
        description="Evaluate system architecture and suggest improvements", 
        agent_id=team["architect"]
    )
    
    # Execute reviews
    code_result = await orchestrator.execute_task(code_review_task)
    arch_result = await orchestrator.execute_task(architecture_review_task)
    
    # Generate final report
    report_task = await orchestrator.create_task(
        description="Compile comprehensive review report",
        agent_id=team["documentation"]
    )
    
    return await orchestrator.execute_task(report_task)
```

### 3. Custom Agent Creation

```python
async def custom_agent_example():
    # Define custom agent configuration
    custom_config = AgentConfig(
        name="Security Analyst",
        description="Specialized agent for security analysis and recommendations",
        instructions='''
        You are a cybersecurity expert. Your role is to:
        1. Analyze code and systems for security vulnerabilities
        2. Recommend security best practices
        3. Generate security reports and documentation
        4. Stay updated on latest security threats and mitigations
        ''',
        model="gpt-4o",
        temperature=0.2
    )
    
    # Create custom agent
    security_agent = await orchestrator.create_agent(
        custom_config, 
        tools=["validate_code", "generate_report", "http_request"]
    )
    
    # Use custom agent
    security_task = await orchestrator.create_task(
        description="Perform security analysis on the authentication system",
        agent_id=security_agent
    )
    
    return await orchestrator.execute_task(security_task)
```

## Configuration

### Environment Variables

Key configuration options:

```bash
# Core settings
OPENAI_API_KEY=your_api_key
MAX_CONCURRENT_AGENTS=10
DEFAULT_MODEL=gpt-4o

# Error handling
ENABLE_CIRCUIT_BREAKERS=true
RETRY_MAX_ATTEMPTS=3
CIRCUIT_FAILURE_THRESHOLD=5

# Security
ENABLE_AUTH=false
RATE_LIMIT_PER_MINUTE=100

# Monitoring  
ENABLE_METRICS=true
PROMETHEUS_ENABLED=true
```

### Configuration Profiles

Predefined profiles for different environments:

```python
# Development profile
config = MasterOrchestratorConfig.get_profile_configs()['development']

# Production profile  
config = MasterOrchestratorConfig.get_profile_configs()['production']
```

## Tool Development

### Creating Custom Tools

```python
from tool_registry import tool, ToolCategory, ToolAccessLevel

@tool(
    name="data_processor",
    description="Process and transform data",
    category=ToolCategory.DATA_PROCESSING,
    access_level=ToolAccessLevel.PUBLIC
)
async def process_data(
    data: List[Dict[str, Any]], 
    operation: str = "transform"
) -> List[Dict[str, Any]]:
    """Process data with specified operation."""
    if operation == "transform":
        return [{"processed": item} for item in data]
    elif operation == "filter":
        return [item for item in data if item.get("active")]
    else:
        raise ValueError(f"Unknown operation: {operation}")

# Register with orchestrator
await orchestrator.register_tool(
    name=process_data._tool_name,
    function=process_data,
    description=process_data._tool_description,
    parameters=process_data._tool_parameters,
    required=process_data._tool_required_params
)
```

### Built-in Tools

The orchestrator includes several built-in tools:

- **File Operations**: `read_file`, `write_file`
- **Data Processing**: `json_parse`, `json_stringify`
- **Web Services**: `http_request`
- **Analysis**: `calculate_metrics`, `validate_code`
- **Reporting**: `generate_report`

## Error Handling and Recovery

### Error Classification

Errors are automatically classified into categories:

- `AGENT_ERROR`: Issues with agent creation or execution
- `TOOL_ERROR`: Tool execution failures
- `NETWORK_ERROR`: Connectivity issues
- `PERMISSION_ERROR`: Access control violations
- `VALIDATION_ERROR`: Input validation failures
- `TIMEOUT_ERROR`: Operation timeouts
- `RESOURCE_ERROR`: Resource exhaustion
- `SYSTEM_ERROR`: System-level failures

### Recovery Strategies

- **RETRY**: Exponential backoff retry for transient failures
- **FALLBACK**: Use alternative tools or agents
- **SKIP**: Skip failed operations and continue
- **ESCALATE**: Alert administrators or higher-level handlers
- **RESET**: Reset system state to recover
- **TERMINATE**: Graceful shutdown for critical errors

### Circuit Breakers

Prevent cascade failures with automatic circuit breakers:

```python
# Circuit breaker automatically opens after 5 failures
# Attempts recovery after 60 seconds
# Requires 3 successes to close circuit
```

## Monitoring and Observability

### Health Checks

```python
# Get system status
agents = await orchestrator.list_agents() 
tasks = await orchestrator.list_tasks()
error_stats = error_handler.get_error_statistics()
```

### Metrics Collection

- Agent performance metrics
- Task execution statistics
- Tool usage analytics
- Error rates and patterns
- Resource utilization

### Logging

Structured logging with configurable levels:

```python
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Best Practices

### Agent Design
- Use specialized agents for specific domains
- Implement clear instructions and constraints
- Monitor agent performance and adjust configurations
- Leverage agent templates for consistency

### Task Management
- Set appropriate task priorities
- Use descriptive task descriptions
- Implement proper error handling
- Monitor task execution times

### Tool Development
- Follow single responsibility principle
- Implement comprehensive error handling
- Add proper input validation
- Include detailed documentation

### Security
- Use environment variables for sensitive data
- Implement proper access controls
- Enable rate limiting in production
- Monitor for unusual activity patterns

### Performance
- Monitor agent resource usage
- Implement caching where appropriate
- Use parallel execution for independent tasks
- Set reasonable timeouts

## API Reference

### Core Classes

#### MasterAgentOrchestrator
Main orchestrator class for agent management.

**Key Methods:**
- `create_agent(config, tools)` - Create new agent
- `delete_agent(agent_id)` - Remove agent  
- `create_task(description, ...)` - Create task
- `execute_task(task_id)` - Execute task
- `list_agents()` - List all agents
- `list_tasks(filter)` - List tasks with optional filter
- `shutdown()` - Graceful shutdown

#### AgentFactory  
Factory for creating specialized agents.

**Key Methods:**
- `create_data_analysis_team(tools)` - Create data analysis team
- `create_development_team(languages, tools)` - Create dev team
- `create_custom_agent(name, description, ...)` - Create custom agent

#### ToolRegistry
Manages tool registration and execution.

**Key Methods:**
- `register_tool(name, function, ...)` - Register tool
- `execute_tool(name, agent_id, args)` - Execute tool
- `get_available_tools(agent_id)` - List available tools

#### ErrorHandler
Comprehensive error handling system.

**Key Methods:**
- `handle_error(error, context, auto_recover)` - Handle error
- `get_error_statistics()` - Get error stats

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: [Read the Docs](https://master-agent-orchestrator.readthedocs.io/)
- Issues: [GitHub Issues](https://github.com/yourusername/master-agent-orchestrator/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/master-agent-orchestrator/discussions)

## Changelog

### v1.0.0 (2025-01-XX)
- Initial release
- Core orchestration functionality
- Agent factory with built-in templates  
- Comprehensive tool registry
- Error handling and recovery
- Production-ready configuration
- Full example suite