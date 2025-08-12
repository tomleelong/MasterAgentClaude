"""
Setup script for Master Agent Orchestrator package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text().split('\n')
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="master-agent-orchestrator",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive orchestrator for managing multiple OpenAI agents with tool delegation and coordination",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/master-agent-orchestrator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0", 
            "pytest-mock>=3.12.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "isort>=5.13.0",
            "flake8>=6.1.0",
            "mypy>=1.8.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
            "structlog>=23.2.0",
        ],
        "database": [
            "sqlalchemy>=2.0.0",
            "asyncpg>=0.29.0",
        ],
        "redis": [
            "redis>=5.0.0",
            "aioredis>=2.0.1",
        ],
        "security": [
            "cryptography>=41.0.0",
            "python-jose>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "master-orchestrator=example_usage:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai, agents, orchestrator, openai, multi-agent, automation, coordination",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/master-agent-orchestrator/issues",
        "Source": "https://github.com/yourusername/master-agent-orchestrator",
        "Documentation": "https://master-agent-orchestrator.readthedocs.io/",
    },
)