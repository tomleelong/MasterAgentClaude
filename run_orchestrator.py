#!/usr/bin/env python3
"""
Quick launcher for the Master Agent Orchestrator.

This script provides an easy way to start the orchestrator with custom prompts.
"""

import asyncio
import sys
from pathlib import Path
import os

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

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


async def quick_prompt(prompt_text: str):
    """Process a single prompt quickly."""
    cli = CLIInterface()
    
    try:
        await cli.initialize()
        results = await cli.process_prompt(prompt_text)
        output = cli.format_output(results, "text")
        print(output)
    finally:
        await cli.cleanup()


def main():
    """Main entry point."""
    
    # Check if prompt is provided as argument
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(f"🎯 Processing prompt: {prompt}")
        asyncio.run(quick_prompt(prompt))
    else:
        # Start interactive mode
        from cli_interface import main as cli_main
        asyncio.run(cli_main())


if __name__ == "__main__":
    main()