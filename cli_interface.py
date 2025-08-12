"""
CLI Interface for Master Agent Orchestrator.

This module provides a command-line interface for interacting with the orchestrator
using custom prompts that dynamically create specialized agent teams.
"""

import asyncio
import argparse
import sys
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from master_orchestrator import MasterAgentOrchestrator
from prompt_analyzer import PromptAnalyzer, DynamicAgentCreator
from config import MasterConfig
from tool_registry import ToolRegistry


class CLIInterface:
    """Command-line interface for the Master Agent Orchestrator."""
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.config = MasterConfig.from_env()
        self.orchestrator = None
        self.analyzer = None
        self.creator = None
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('orchestrator_cli.log')
            ]
        )
    
    async def initialize(self):
        """Initialize the orchestrator and related components."""
        print("🚀 Initializing Master Agent Orchestrator...")
        
        try:
            # Initialize orchestrator
            self.orchestrator = MasterAgentOrchestrator(
                api_key=self.config.openai.api_key,
                config=self.config
            )
            
            # Initialize analyzer and creator
            self.analyzer = PromptAnalyzer(self.config)
            self.creator = DynamicAgentCreator(self.orchestrator, self.analyzer)
            
            print("✅ Orchestrator initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize orchestrator: {e}")
            sys.exit(1)
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.orchestrator:
            print("🧹 Cleaning up resources...")
            await self.orchestrator.shutdown()
            print("✅ Cleanup complete!")
    
    async def process_prompt(self, prompt: str, output_format: str = "text") -> Dict[str, Any]:
        """
        Process a user prompt by creating appropriate agents and executing tasks.
        
        Args:
            prompt: User's natural language prompt
            output_format: Output format (text, json, or detailed)
            
        Returns:
            Results dictionary
        """
        results = {
            "prompt": prompt,
            "analysis": None,
            "agents_created": [],
            "task_results": [],
            "success": False,
            "error": None
        }
        
        try:
            print(f"🔍 Analyzing prompt: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'")
            
            # Create agents based on prompt
            agent_ids, analysis = await self.creator.create_agents_for_prompt(prompt)
            results["analysis"] = {
                "categories": [cat.value for cat in analysis.task_categories],
                "agents_needed": analysis.estimated_agents_needed,
                "complexity": analysis.complexity_score,
                "intent": analysis.prompt_intent,
                "deliverables": analysis.key_deliverables
            }
            results["agents_created"] = agent_ids
            
            if not agent_ids:
                raise Exception("No agents were created for this prompt")
            
            print(f"🤖 Created {len(agent_ids)} specialized agents")
            
            # Execute the main task with created agents
            print("⚡ Executing task...")
            
            # Create a comprehensive task for the lead agent
            lead_agent_id = agent_ids[0]
            task_description = f"""
            ORIGINAL USER REQUEST: {prompt}
            
            AVAILABLE TEAM:
            {chr(10).join([f"- Agent {i+1}: {req.name} ({req.specialization})" for i, req in enumerate(analysis.agent_requirements)])}
            
            WORKFLOW STEPS:
            {chr(10).join([f"{i+1}. {step}" for i, step in enumerate(analysis.workflow_steps)])}
            
            Your task is to coordinate with the team to fulfill this request. 
            Break down the work, delegate to appropriate specialists, and synthesize the results.
            
            Expected deliverables: {', '.join(analysis.key_deliverables)}
            """
            
            task_id = await self.orchestrator.create_task(
                description=task_description,
                agent_id=lead_agent_id,
                priority=1
            )
            
            # Execute the task
            result = await self.orchestrator.execute_task(task_id)
            results["task_results"].append({
                "task_id": task_id,
                "agent_id": lead_agent_id,
                "result": result
            })
            
            results["success"] = True
            print("✅ Task completed successfully!")
            
        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Error processing prompt: {e}")
            self.logger.error(f"Prompt processing failed: {e}", exc_info=True)
        
        return results
    
    def format_output(self, results: Dict[str, Any], format_type: str = "text") -> str:
        """Format the results for output."""
        
        if format_type == "json":
            return json.dumps(results, indent=2, default=str)
        
        elif format_type == "detailed":
            output = []
            output.append("="*60)
            output.append("MASTER AGENT ORCHESTRATOR - EXECUTION RESULTS")
            output.append("="*60)
            output.append(f"Prompt: {results['prompt']}")
            output.append("")
            
            if results["analysis"]:
                analysis = results["analysis"]
                output.append("📊 ANALYSIS:")
                output.append(f"  Intent: {analysis['intent']}")
                output.append(f"  Categories: {', '.join(analysis['categories'])}")
                output.append(f"  Complexity: {analysis['complexity']}/10")
                output.append(f"  Agents Created: {analysis['agents_needed']}")
                output.append("")
            
            if results["agents_created"]:
                output.append("🤖 AGENTS CREATED:")
                for i, agent_id in enumerate(results["agents_created"], 1):
                    output.append(f"  {i}. Agent ID: {agent_id}")
                output.append("")
            
            if results["task_results"]:
                output.append("📝 RESULTS:")
                for i, task_result in enumerate(results["task_results"], 1):
                    output.append(f"  Task {i}:")
                    output.append(f"    Agent: {task_result['agent_id']}")
                    output.append(f"    Result: {task_result['result']}")
                output.append("")
            
            if results["error"]:
                output.append(f"❌ ERROR: {results['error']}")
            
            output.append("="*60)
            return "\n".join(output)
        
        else:  # text format
            if results["success"]:
                main_result = results["task_results"][0]["result"] if results["task_results"] else "Task completed"
                return f"✅ SUCCESS:\n{main_result}"
            else:
                return f"❌ FAILED: {results['error']}"
    
    async def interactive_mode(self):
        """Run in interactive mode for continuous prompts."""
        print("🎯 Entering interactive mode. Type 'exit' to quit, 'help' for commands.")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n💭 Enter your prompt: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() in ['help', 'h']:
                    self._show_help()
                    continue
                
                if user_input.lower().startswith('format '):
                    format_type = user_input[7:].strip()
                    if format_type in ['text', 'json', 'detailed']:
                        self.output_format = format_type
                        print(f"📄 Output format set to: {format_type}")
                    else:
                        print("❌ Invalid format. Use: text, json, or detailed")
                    continue
                
                # Process the prompt
                results = await self.process_prompt(user_input)
                output = self.format_output(results, getattr(self, 'output_format', 'text'))
                print("\n" + output)
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
🎯 MASTER AGENT ORCHESTRATOR - HELP

COMMANDS:
  exit, quit, q     - Exit the program
  help, h           - Show this help message
  format <type>     - Set output format (text, json, detailed)

USAGE:
  Simply type your request in natural language. The system will:
  1. Analyze your prompt to understand requirements
  2. Create specialized agents for the task
  3. Execute the task using the agent team
  4. Return the results

EXAMPLES:
  "Analyze this CSV data and create a summary report"
  "Review my Python code for security vulnerabilities"
  "Design a microservices architecture for an e-commerce platform"
  "Create documentation for my REST API"
  "Plan a project timeline for a mobile app development"

The system will automatically determine what type of agents are needed!
        """
        print(help_text)


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Master Agent Orchestrator - Dynamic AI Agent Team Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cli_interface.py
  
  # Single prompt
  python cli_interface.py --prompt "Analyze my sales data and create a report"
  
  # Output to JSON
  python cli_interface.py --prompt "Review my code" --format json
  
  # Detailed output
  python cli_interface.py --prompt "Design a system" --format detailed
        """
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        help="Single prompt to process (non-interactive mode)"
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json', 'detailed'],
        default='text',
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help="Output file path (default: stdout)"
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help="Force interactive mode even with --prompt"
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = CLIInterface()
    
    try:
        await cli.initialize()
        
        if args.prompt and not args.interactive:
            # Single prompt mode
            results = await cli.process_prompt(args.prompt)
            output = cli.format_output(results, args.format)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                print(f"✅ Results saved to: {args.output}")
            else:
                print(output)
        
        else:
            # Interactive mode
            await cli.interactive_mode()
    
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)