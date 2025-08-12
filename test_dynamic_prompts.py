"""
Test script for dynamic prompt processing.

This script tests the dynamic agent creation system with various types of prompts
to ensure the system can handle different request types effectively.
"""

import asyncio
import os
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback: manually load .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

from cli_interface import CLIInterface


# Test prompts covering different categories
TEST_PROMPTS = [
    {
        "category": "Data Analysis",
        "prompt": "I have a CSV file with sales data from the last quarter. Can you analyze it to find trends, identify top-performing products, and create a summary report with visualizations?",
        "expected_agents": ["data_analyst", "report_writer"]
    },
    {
        "category": "Code Review",
        "prompt": "Please review my Python Flask application for security vulnerabilities, performance issues, and code quality. Focus on authentication, database queries, and API endpoints.",
        "expected_agents": ["code_reviewer", "security_specialist"]
    },
    {
        "category": "System Design",
        "prompt": "Design a scalable microservices architecture for a social media platform that can handle 1 million users. Include database design, API structure, and deployment strategy.",
        "expected_agents": ["system_architect", "database_designer"]
    },
    {
        "category": "Documentation",
        "prompt": "Create comprehensive API documentation for my REST API including endpoint descriptions, request/response examples, authentication details, and error handling guides.",
        "expected_agents": ["documentation_writer", "technical_writer"]
    },
    {
        "category": "Project Management",
        "prompt": "Plan a 6-month development timeline for building a mobile e-commerce app. Include phases, milestones, resource allocation, and risk management strategies.",
        "expected_agents": ["project_manager", "planning_specialist"]
    },
    {
        "category": "Testing Strategy",
        "prompt": "Develop a comprehensive testing strategy for a web application including unit tests, integration tests, performance testing, and automated CI/CD pipeline setup.",
        "expected_agents": ["testing_specialist", "automation_engineer"]
    },
    {
        "category": "Multi-Domain",
        "prompt": "I'm launching a SaaS product. Help me with market research, technical architecture, development planning, testing strategy, and go-to-market documentation.",
        "expected_agents": ["researcher", "architect", "project_manager", "testing_specialist", "documentation_writer"]
    },
    {
        "category": "Problem Solving",
        "prompt": "My web application is experiencing slow response times and high server load. Help me diagnose the issues and implement solutions.",
        "expected_agents": ["performance_analyst", "system_optimizer"]
    }
]


async def test_prompt_analysis():
    """Test the prompt analysis and agent creation system."""
    
    print("🧪 TESTING DYNAMIC PROMPT PROCESSING")
    print("=" * 60)
    
    cli = CLIInterface()
    
    try:
        # Initialize the system
        await cli.initialize()
        
        for i, test_case in enumerate(TEST_PROMPTS, 1):
            print(f"\n🔍 TEST {i}: {test_case['category']}")
            print("-" * 40)
            print(f"Prompt: {test_case['prompt'][:100]}...")
            
            try:
                # Process the prompt
                results = await cli.process_prompt(test_case['prompt'], output_format="detailed")
                
                # Display results
                if results["success"]:
                    print("✅ SUCCESS")
                    analysis = results.get("analysis", {})
                    print(f"  📊 Intent: {analysis.get('intent', 'Unknown')}")
                    print(f"  📊 Categories: {', '.join(analysis.get('categories', []))}")
                    print(f"  📊 Complexity: {analysis.get('complexity', 'Unknown')}/10")
                    print(f"  🤖 Agents Created: {len(results.get('agents_created', []))}")
                    
                    if results.get('task_results'):
                        result_preview = results['task_results'][0]['result'][:200]
                        print(f"  📝 Result Preview: {result_preview}...")
                else:
                    print(f"❌ FAILED: {results.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
            
            print()
    
    finally:
        await cli.cleanup()


async def demo_interactive_mode():
    """Demonstrate interactive mode with sample prompts."""
    
    print("🎯 INTERACTIVE MODE DEMO")
    print("=" * 60)
    print("This demo shows how the interactive mode works.")
    print("In real usage, you would type prompts manually.")
    print()
    
    cli = CLIInterface()
    
    try:
        await cli.initialize()
        
        # Simulate interactive prompts
        demo_prompts = [
            "Analyze customer feedback data to identify common complaints",
            "Review my JavaScript code for potential security issues",
            "Design a database schema for a library management system"
        ]
        
        for prompt in demo_prompts:
            print(f"💭 Demo prompt: {prompt}")
            results = await cli.process_prompt(prompt)
            output = cli.format_output(results, "text")
            print(f"🤖 Response: {output}")
            print("-" * 40)
    
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    print("🚀 MASTER AGENT ORCHESTRATOR - DYNAMIC PROMPT TESTING")
    print()
    
    # Choose test mode
    print("Choose test mode:")
    print("1. Comprehensive prompt analysis test")
    print("2. Interactive mode demonstration")
    print("3. Both")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            asyncio.run(test_prompt_analysis())
        elif choice == "2":
            asyncio.run(demo_interactive_mode())
        elif choice == "3":
            asyncio.run(test_prompt_analysis())
            print("\n" + "="*60 + "\n")
            asyncio.run(demo_interactive_mode())
        else:
            print("Invalid choice. Running comprehensive test...")
            asyncio.run(test_prompt_analysis())
    
    except KeyboardInterrupt:
        print("\n👋 Test interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Test failed: {e}")