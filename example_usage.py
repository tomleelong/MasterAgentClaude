"""
Example Usage and Demonstration of the Master Agent Orchestrator.

This module provides comprehensive examples showing how to use the
orchestrator for various multi-agent scenarios.
"""

import asyncio
import os
import logging
from typing import Dict, List, Any
from datetime import datetime

# Import our orchestrator modules
from master_orchestrator import (
    MasterAgentOrchestrator, 
    AgentConfig, 
    Task, 
    TaskPriority,
    AgentStatus
)
from agent_factory import AgentFactory, AgentTemplates
from tool_registry import ToolRegistry, ToolCategory, ToolAccessLevel, tool
from error_handling import ErrorHandler, ErrorContext


# Custom tool examples
@tool(
    name="calculate_metrics",
    description="Calculate statistical metrics for a dataset",
    category=ToolCategory.ANALYSIS,
    access_level=ToolAccessLevel.PUBLIC
)
async def calculate_metrics(data: List[float], metrics: List[str]) -> Dict[str, float]:
    """Calculate statistical metrics for numerical data."""
    if not data:
        return {}
    
    results = {}
    
    if "mean" in metrics:
        results["mean"] = sum(data) / len(data)
    
    if "median" in metrics:
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            results["median"] = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            results["median"] = sorted_data[n//2]
    
    if "std" in metrics:
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        results["std"] = variance ** 0.5
    
    if "min" in metrics:
        results["min"] = min(data)
    
    if "max" in metrics:
        results["max"] = max(data)
    
    return results


@tool(
    name="generate_report",
    description="Generate a formatted report from analysis results",
    category=ToolCategory.DATA_PROCESSING,
    access_level=ToolAccessLevel.PUBLIC
)
async def generate_report(title: str, data: Dict[str, Any], format_type: str = "markdown") -> str:
    """Generate a formatted report from data."""
    if format_type.lower() == "markdown":
        report = f"# {title}\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for key, value in data.items():
            if isinstance(value, dict):
                report += f"## {key.title()}\n\n"
                for subkey, subvalue in value.items():
                    report += f"- **{subkey}**: {subvalue}\n"
                report += "\n"
            else:
                report += f"- **{key}**: {value}\n"
        
        return report
    
    elif format_type.lower() == "json":
        import json
        return json.dumps({
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "data": data
        }, indent=2)
    
    else:
        # Plain text format
        report = f"{title}\n"
        report += "=" * len(title) + "\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for key, value in data.items():
            report += f"{key}: {value}\n"
        
        return report


@tool(
    name="validate_code",
    description="Validate code syntax and style",
    category=ToolCategory.ANALYSIS,
    access_level=ToolAccessLevel.RESTRICTED
)
async def validate_code(code: str, language: str = "python") -> Dict[str, Any]:
    """Validate code syntax and basic style."""
    results = {
        "syntax_valid": True,
        "style_issues": [],
        "suggestions": []
    }
    
    if language.lower() == "python":
        try:
            import ast
            ast.parse(code)
            results["syntax_valid"] = True
        except SyntaxError as e:
            results["syntax_valid"] = False
            results["style_issues"].append(f"Syntax error: {e}")
        
        # Basic style checks
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                results["style_issues"].append(f"Line {i}: Line too long ({len(line)} chars)")
            
            if line.strip().endswith(' '):
                results["style_issues"].append(f"Line {i}: Trailing whitespace")
    
    # General suggestions
    if not results["style_issues"]:
        results["suggestions"].append("Code looks good!")
    else:
        results["suggestions"].append("Consider fixing style issues for better readability")
    
    return results


class ExampleOrchestrator:
    """Example demonstrating orchestrator usage patterns."""
    
    def __init__(self, api_key: str):
        """Initialize the example orchestrator."""
        self.api_key = api_key
        self.orchestrator = None
        self.tool_registry = None
        self.agent_factory = None
        self.error_handler = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize all components."""
        self.logger.info("Initializing Master Agent Orchestrator...")
        
        # Initialize error handler
        self.error_handler = ErrorHandler(
            enable_circuit_breakers=True,
            enable_auto_recovery=True
        )
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry(enable_metrics=True)
        
        # Initialize orchestrator
        self.orchestrator = MasterAgentOrchestrator(
            api_key=self.api_key,
            max_concurrent_agents=5,
            enable_logging=True
        )
        
        # Initialize agent factory
        self.agent_factory = AgentFactory(self.orchestrator)
        
        # Register custom tools
        await self._register_custom_tools()
        
        self.logger.info("Initialization complete!")
    
    async def _register_custom_tools(self):
        """Register custom tools with the orchestrator."""
        tools = [
            (calculate_metrics, ["data", "metrics"]),
            (generate_report, ["title", "data"]),
            (validate_code, ["code"])
        ]
        
        for tool_func, required_params in tools:
            await self.orchestrator.register_tool(
                name=tool_func._tool_name,
                function=tool_func,
                description=tool_func._tool_description,
                parameters=tool_func._tool_parameters,
                required=required_params
            )
            
            self.logger.info(f"Registered tool: {tool_func._tool_name}")
    
    async def example_1_data_analysis_workflow(self):
        """
        Example 1: Data Analysis Workflow
        
        Demonstrates creating a data analysis team and coordinating
        multiple agents to analyze data and generate reports.
        """
        self.logger.info("=== Example 1: Data Analysis Workflow ===")
        
        try:
            # Create data analysis team
            team = await self.agent_factory.create_data_analysis_team(
                tools=["calculate_metrics", "generate_report", "json_parse", "json_stringify"]
            )
            
            self.logger.info(f"Created data analysis team: {list(team.keys())}")
            
            # Sample data for analysis
            sample_data = [1.2, 2.5, 3.8, 2.1, 4.5, 3.2, 2.9, 5.1, 4.2, 3.6]
            
            # Task 1: Analyze data
            analysis_task = await self.orchestrator.create_task(
                description=f"Analyze the following dataset and calculate mean, median, std, min, max: {sample_data}",
                agent_id=team["data_analyst"],
                priority=TaskPriority.HIGH,
                metadata={"type": "data_analysis", "dataset_size": len(sample_data)}
            )
            
            # Execute analysis task
            analysis_result = await self.orchestrator.execute_task(analysis_task)
            self.logger.info(f"Analysis completed: {analysis_result}")
            
            # Task 2: Generate report
            report_task = await self.orchestrator.create_task(
                description="Generate a comprehensive markdown report of the data analysis results",
                agent_id=team["report_writer"],
                priority=TaskPriority.MEDIUM,
                metadata={"type": "report_generation", "format": "markdown"}
            )
            
            # Execute report task
            report_result = await self.orchestrator.execute_task(report_task)
            self.logger.info(f"Report generated: {report_result}")
            
            return {
                "team": team,
                "analysis_result": analysis_result,
                "report_result": report_result
            }
            
        except Exception as e:
            await self.error_handler.handle_error(
                e, 
                ErrorContext(operation="data_analysis_workflow")
            )
            raise
    
    async def example_2_code_review_pipeline(self):
        """
        Example 2: Code Review Pipeline
        
        Demonstrates creating a development team and implementing
        an automated code review pipeline.
        """
        self.logger.info("=== Example 2: Code Review Pipeline ===")
        
        try:
            # Create development team
            team = await self.agent_factory.create_development_team(
                languages=["Python", "JavaScript"],
                tools=["validate_code", "generate_report", "read_file", "write_file"]
            )
            
            self.logger.info(f"Created development team: {list(team.keys())}")
            
            # Sample code for review
            sample_code = '''
def calculate_fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Usage example
result = calculate_fibonacci(10)
print(f"Fibonacci(10) = {result}")
'''
            
            # Task 1: Code validation
            validation_task = await self.orchestrator.create_task(
                description=f"Validate the following Python code for syntax and style issues:\n\n{sample_code}",
                agent_id=team["code_reviewer"],
                priority=TaskPriority.HIGH,
                metadata={"type": "code_validation", "language": "python"}
            )
            
            validation_result = await self.orchestrator.execute_task(validation_task)
            self.logger.info(f"Code validation completed: {validation_result}")
            
            # Task 2: Architecture review
            architecture_task = await self.orchestrator.create_task(
                description="Review the fibonacci implementation for performance and suggest improvements",
                agent_id=team["architect"],
                priority=TaskPriority.MEDIUM,
                metadata={"type": "architecture_review"}
            )
            
            architecture_result = await self.orchestrator.execute_task(architecture_task)
            self.logger.info(f"Architecture review completed: {architecture_result}")
            
            # Task 3: Generate comprehensive review report
            review_report_task = await self.orchestrator.create_task(
                description="Generate a comprehensive code review report including validation results and architecture recommendations",
                agent_id=team["documentation"],
                priority=TaskPriority.MEDIUM,
                metadata={"type": "review_report"}
            )
            
            review_report_result = await self.orchestrator.execute_task(review_report_task)
            self.logger.info(f"Review report generated: {review_report_result}")
            
            return {
                "team": team,
                "validation_result": validation_result,
                "architecture_result": architecture_result,
                "review_report": review_report_result
            }
            
        except Exception as e:
            await self.error_handler.handle_error(
                e,
                ErrorContext(operation="code_review_pipeline")
            )
            raise
    
    async def example_3_parallel_task_execution(self):
        """
        Example 3: Parallel Task Execution
        
        Demonstrates running multiple tasks in parallel across
        different agents with coordination.
        """
        self.logger.info("=== Example 3: Parallel Task Execution ===")
        
        try:
            # Create multiple specialized agents
            analyst_config = AgentTemplates.data_analyst_agent(
                name="Parallel Analyst 1"
            )
            analyst1_id = await self.orchestrator.create_agent(
                analyst_config,
                tools=["calculate_metrics", "json_stringify"]
            )
            
            analyst_config2 = AgentTemplates.data_analyst_agent(
                name="Parallel Analyst 2"
            )
            analyst2_id = await self.orchestrator.create_agent(
                analyst_config2,
                tools=["calculate_metrics", "json_stringify"]
            )
            
            reviewer_config = AgentTemplates.code_reviewer_agent()
            reviewer_id = await self.orchestrator.create_agent(
                reviewer_config,
                tools=["validate_code"]
            )
            
            # Create multiple tasks for parallel execution
            tasks = []
            
            # Data analysis tasks
            datasets = [
                [1, 2, 3, 4, 5],
                [10, 20, 30, 40, 50],
                [100, 200, 300, 400, 500]
            ]
            
            for i, dataset in enumerate(datasets):
                task_id = await self.orchestrator.create_task(
                    description=f"Analyze dataset {i+1}: {dataset}. Calculate mean and standard deviation.",
                    agent_id=analyst1_id if i % 2 == 0 else analyst2_id,
                    priority=TaskPriority.MEDIUM,
                    metadata={"dataset_id": i+1, "type": "parallel_analysis"}
                )
                tasks.append(task_id)
            
            # Code review tasks
            code_samples = [
                "def hello(): print('Hello World')",
                "for i in range(10): print(i)",
                "import os\nprint(os.getcwd())"
            ]
            
            for i, code in enumerate(code_samples):
                task_id = await self.orchestrator.create_task(
                    description=f"Validate code sample {i+1}: {code}",
                    agent_id=reviewer_id,
                    priority=TaskPriority.LOW,
                    metadata={"code_sample_id": i+1, "type": "parallel_review"}
                )
                tasks.append(task_id)
            
            # Execute all tasks in parallel
            self.logger.info(f"Executing {len(tasks)} tasks in parallel...")
            
            results = await asyncio.gather(
                *[self.orchestrator.execute_task(task_id) for task_id in tasks],
                return_exceptions=True
            )
            
            # Process results
            successful_results = []
            failed_results = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_results.append({"task_id": tasks[i], "error": str(result)})
                else:
                    successful_results.append({"task_id": tasks[i], "result": result})
            
            self.logger.info(f"Parallel execution completed: {len(successful_results)} successful, {len(failed_results)} failed")
            
            return {
                "total_tasks": len(tasks),
                "successful_results": successful_results,
                "failed_results": failed_results,
                "agents_used": [analyst1_id, analyst2_id, reviewer_id]
            }
            
        except Exception as e:
            await self.error_handler.handle_error(
                e,
                ErrorContext(operation="parallel_task_execution")
            )
            raise
    
    async def example_4_error_handling_and_recovery(self):
        """
        Example 4: Error Handling and Recovery
        
        Demonstrates error handling, circuit breakers, and
        automatic recovery mechanisms.
        """
        self.logger.info("=== Example 4: Error Handling and Recovery ===")
        
        try:
            # Create agent with limited tools to trigger errors
            test_config = AgentConfig(
                name="Error Test Agent",
                description="Agent for testing error handling",
                instructions="You are a test agent. Try to execute tasks even with limited tools.",
                model="gpt-4o"
            )
            
            test_agent_id = await self.orchestrator.create_agent(
                test_config,
                tools=[]  # No tools available - will cause tool errors
            )
            
            # Task that will fail due to missing tools
            failing_task_id = await self.orchestrator.create_task(
                description="Calculate the mean of [1, 2, 3, 4, 5] using the calculate_metrics tool",
                agent_id=test_agent_id,
                priority=TaskPriority.LOW,
                metadata={"type": "error_test", "expected_to_fail": True}
            )
            
            try:
                result = await self.orchestrator.execute_task(failing_task_id)
                self.logger.warning(f"Task unexpectedly succeeded: {result}")
            except Exception as e:
                self.logger.info(f"Task failed as expected: {str(e)}")
                
                # Check error statistics
                error_stats = self.error_handler.get_error_statistics()
                self.logger.info(f"Error statistics: {error_stats}")
            
            # Create a task that should succeed
            success_agent_config = AgentTemplates.data_analyst_agent(
                name="Success Test Agent"
            )
            success_agent_id = await self.orchestrator.create_agent(
                success_agent_config,
                tools=["calculate_metrics", "json_stringify"]
            )
            
            success_task_id = await self.orchestrator.create_task(
                description="Calculate basic statistics for the numbers 1 through 10",
                agent_id=success_agent_id,
                priority=TaskPriority.MEDIUM,
                metadata={"type": "recovery_test"}
            )
            
            success_result = await self.orchestrator.execute_task(success_task_id)
            self.logger.info(f"Recovery task succeeded: {success_result}")
            
            return {
                "error_handling_tested": True,
                "success_result": success_result,
                "error_statistics": error_stats
            }
            
        except Exception as e:
            await self.error_handler.handle_error(
                e,
                ErrorContext(operation="error_handling_test")
            )
            raise
    
    async def example_5_resource_monitoring(self):
        """
        Example 5: Resource Monitoring and Management
        
        Demonstrates monitoring agent resources, task queues,
        and system performance.
        """
        self.logger.info("=== Example 5: Resource Monitoring ===")
        
        try:
            # Get system status
            agent_list = await self.orchestrator.list_agents()
            task_list = await self.orchestrator.list_tasks()
            
            self.logger.info(f"Active agents: {len(agent_list)}")
            self.logger.info(f"Total tasks: {len(task_list)}")
            
            # Get tool registry statistics
            available_tools = self.tool_registry.get_available_tools()
            self.logger.info(f"Available tools: {len(available_tools)}")
            
            # Get error handler statistics
            error_stats = self.error_handler.get_error_statistics()
            
            # Create monitoring report
            monitoring_report = {
                "timestamp": datetime.now().isoformat(),
                "system_status": {
                    "total_agents": len(agent_list),
                    "active_agents": len([a for a in agent_list if a["status"] == "idle"]),
                    "total_tasks": len(task_list),
                    "completed_tasks": len([t for t in task_list if t["status"] == "completed"]),
                    "failed_tasks": len([t for t in task_list if t["status"] == "error"]),
                    "available_tools": len(available_tools)
                },
                "agent_details": agent_list,
                "recent_tasks": [t for t in task_list if t["status"] in ["running", "completed", "error"]],
                "tool_summary": [
                    {
                        "name": tool["name"],
                        "category": tool["category"],
                        "access_level": tool["access_level"]
                    } for tool in available_tools
                ],
                "error_statistics": error_stats
            }
            
            self.logger.info("Monitoring report generated successfully")
            
            return monitoring_report
            
        except Exception as e:
            await self.error_handler.handle_error(
                e,
                ErrorContext(operation="resource_monitoring")
            )
            raise
    
    async def cleanup(self):
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.shutdown()
        self.logger.info("Cleanup completed")


async def main():
    """Main example execution function."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize example orchestrator
    example = ExampleOrchestrator(api_key)
    
    try:
        # Initialize components
        await example.initialize()
        
        # Run examples
        print("\n" + "="*60)
        print("MASTER AGENT ORCHESTRATOR - EXAMPLE DEMONSTRATIONS")
        print("="*60)
        
        # Example 1: Data Analysis
        result1 = await example.example_1_data_analysis_workflow()
        print(f"\nExample 1 Results: {result1}")
        
        # Example 2: Code Review
        result2 = await example.example_2_code_review_pipeline()
        print(f"\nExample 2 Results: {result2}")
        
        # Example 3: Parallel Execution
        result3 = await example.example_3_parallel_task_execution()
        print(f"\nExample 3 Results: {result3}")
        
        # Example 4: Error Handling
        result4 = await example.example_4_error_handling_and_recovery()
        print(f"\nExample 4 Results: {result4}")
        
        # Example 5: Resource Monitoring
        result5 = await example.example_5_resource_monitoring()
        print(f"\nExample 5 Results: {result5}")
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await example.cleanup()


if __name__ == "__main__":
    asyncio.run(main())