"""
Agent Factory for creating specialized agents with predefined configurations.

This module provides factory methods and templates for creating common types
of specialized agents with optimized configurations.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from master_orchestrator import AgentConfig, TaskPriority


class AgentTemplates:
    """Predefined agent templates for common use cases."""
    
    @staticmethod
    def data_analyst_agent(
        name: str = "Data Analyst",
        custom_instructions: Optional[str] = None
    ) -> AgentConfig:
        """
        Create a data analyst agent configuration.
        
        Args:
            name: Agent name
            custom_instructions: Additional custom instructions
            
        Returns:
            AgentConfig for data analyst
        """
        base_instructions = """
        You are a specialized data analyst agent. Your role is to:
        1. Analyze datasets and identify patterns, trends, and insights
        2. Perform statistical analysis and data validation
        3. Create data visualizations and reports
        4. Clean and preprocess data for analysis
        5. Recommend data-driven solutions and strategies
        
        Always provide clear explanations of your analysis methodology
        and ensure your findings are actionable and well-documented.
        """
        
        if custom_instructions:
            base_instructions += f"\n\nAdditional Instructions:\n{custom_instructions}"
        
        return AgentConfig(
            name=name,
            description="Specialized agent for data analysis, statistics, and insights generation",
            instructions=base_instructions,
            tools=[],  # Tools will be added by orchestrator
            model="gpt-4o",
            temperature=0.3,  # Lower temperature for more consistent analysis
            metadata={"type": "data_analyst", "specialization": "analytics"}
        )
    
    @staticmethod
    def code_reviewer_agent(
        name: str = "Code Reviewer",
        programming_languages: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None
    ) -> AgentConfig:
        """
        Create a code reviewer agent configuration.
        
        Args:
            name: Agent name
            programming_languages: List of languages to focus on
            custom_instructions: Additional custom instructions
            
        Returns:
            AgentConfig for code reviewer
        """
        languages = programming_languages or ["Python", "JavaScript", "TypeScript", "Java"]
        language_list = ", ".join(languages)
        
        base_instructions = f"""
        You are a specialized code review agent. Your role is to:
        1. Review code for bugs, security vulnerabilities, and performance issues
        2. Ensure code follows best practices and coding standards
        3. Check for proper documentation and type hints
        4. Suggest improvements for code readability and maintainability
        5. Verify test coverage and quality
        
        Programming languages you specialize in: {language_list}
        
        Provide constructive feedback with specific examples and recommendations.
        Always explain the reasoning behind your suggestions.
        """
        
        if custom_instructions:
            base_instructions += f"\n\nAdditional Instructions:\n{custom_instructions}"
        
        return AgentConfig(
            name=name,
            description="Specialized agent for code review, quality assurance, and best practices",
            instructions=base_instructions,
            tools=[],
            model="gpt-4o",
            temperature=0.2,  # Low temperature for consistent reviews
            metadata={
                "type": "code_reviewer", 
                "languages": languages,
                "specialization": "code_quality"
            }
        )
    
    @staticmethod
    def system_architect_agent(
        name: str = "System Architect",
        focus_areas: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None
    ) -> AgentConfig:
        """
        Create a system architect agent configuration.
        
        Args:
            name: Agent name
            focus_areas: List of architectural focus areas
            custom_instructions: Additional custom instructions
            
        Returns:
            AgentConfig for system architect
        """
        areas = focus_areas or [
            "microservices", "scalability", "performance", 
            "security", "cloud architecture", "database design"
        ]
        areas_list = ", ".join(areas)
        
        base_instructions = f"""
        You are a specialized system architect agent. Your role is to:
        1. Design scalable and maintainable system architectures
        2. Evaluate technology choices and architectural patterns
        3. Create technical specifications and documentation
        4. Identify potential bottlenecks and failure points
        5. Recommend best practices for system design and implementation
        
        Your areas of expertise include: {areas_list}
        
        Focus on creating robust, scalable solutions that meet both
        current requirements and future growth needs.
        """
        
        if custom_instructions:
            base_instructions += f"\n\nAdditional Instructions:\n{custom_instructions}"
        
        return AgentConfig(
            name=name,
            description="Specialized agent for system architecture, design patterns, and technical planning",
            instructions=base_instructions,
            tools=[],
            model="gpt-4o",
            temperature=0.4,
            metadata={
                "type": "system_architect", 
                "focus_areas": areas,
                "specialization": "architecture"
            }
        )
    
    @staticmethod
    def testing_agent(
        name: str = "Testing Specialist",
        testing_types: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None
    ) -> AgentConfig:
        """
        Create a testing specialist agent configuration.
        
        Args:
            name: Agent name
            testing_types: List of testing types to focus on
            custom_instructions: Additional custom instructions
            
        Returns:
            AgentConfig for testing specialist
        """
        test_types = testing_types or [
            "unit testing", "integration testing", "end-to-end testing",
            "performance testing", "security testing", "API testing"
        ]
        types_list = ", ".join(test_types)
        
        base_instructions = f"""
        You are a specialized testing agent. Your role is to:
        1. Design comprehensive test strategies and test cases
        2. Create automated test suites and testing frameworks
        3. Identify edge cases and potential failure scenarios
        4. Perform various types of testing including: {types_list}
        5. Analyze test results and provide actionable feedback
        
        Focus on ensuring high code quality through thorough testing
        and continuous quality assurance practices.
        """
        
        if custom_instructions:
            base_instructions += f"\n\nAdditional Instructions:\n{custom_instructions}"
        
        return AgentConfig(
            name=name,
            description="Specialized agent for test design, automation, and quality assurance",
            instructions=base_instructions,
            tools=[],
            model="gpt-4o",
            temperature=0.3,
            metadata={
                "type": "testing_specialist", 
                "testing_types": test_types,
                "specialization": "quality_assurance"
            }
        )
    
    @staticmethod
    def documentation_agent(
        name: str = "Documentation Writer",
        doc_types: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None
    ) -> AgentConfig:
        """
        Create a documentation writer agent configuration.
        
        Args:
            name: Agent name
            doc_types: List of documentation types to focus on
            custom_instructions: Additional custom instructions
            
        Returns:
            AgentConfig for documentation writer
        """
        documentation_types = doc_types or [
            "API documentation", "user guides", "technical specifications",
            "code comments", "README files", "architecture documentation"
        ]
        doc_list = ", ".join(documentation_types)
        
        base_instructions = f"""
        You are a specialized documentation agent. Your role is to:
        1. Create clear, comprehensive, and user-friendly documentation
        2. Write technical specifications and API documentation
        3. Generate code comments and inline documentation
        4. Create user guides and tutorials
        5. Maintain consistency in documentation style and format
        
        Documentation types you specialize in: {doc_list}
        
        Focus on making complex technical concepts accessible to your
        target audience while maintaining accuracy and completeness.
        """
        
        if custom_instructions:
            base_instructions += f"\n\nAdditional Instructions:\n{custom_instructions}"
        
        return AgentConfig(
            name=name,
            description="Specialized agent for technical writing and documentation creation",
            instructions=base_instructions,
            tools=[],
            model="gpt-4o",
            temperature=0.5,  # Slightly higher for creative writing
            metadata={
                "type": "documentation_writer", 
                "doc_types": documentation_types,
                "specialization": "technical_writing"
            }
        )
    
    @staticmethod
    def project_manager_agent(
        name: str = "Project Manager",
        methodologies: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None
    ) -> AgentConfig:
        """
        Create a project manager agent configuration.
        
        Args:
            name: Agent name
            methodologies: List of project management methodologies
            custom_instructions: Additional custom instructions
            
        Returns:
            AgentConfig for project manager
        """
        pm_methodologies = methodologies or ["Agile", "Scrum", "Kanban", "Waterfall"]
        methods_list = ", ".join(pm_methodologies)
        
        base_instructions = f"""
        You are a specialized project management agent. Your role is to:
        1. Plan and coordinate project activities and timelines
        2. Manage task priorities and resource allocation
        3. Track project progress and identify potential risks
        4. Facilitate communication between team members
        5. Ensure project deliverables meet quality standards and deadlines
        
        Project management methodologies you're familiar with: {methods_list}
        
        Focus on keeping projects on track while maintaining team
        productivity and stakeholder satisfaction.
        """
        
        if custom_instructions:
            base_instructions += f"\n\nAdditional Instructions:\n{custom_instructions}"
        
        return AgentConfig(
            name=name,
            description="Specialized agent for project coordination, planning, and management",
            instructions=base_instructions,
            tools=[],
            model="gpt-4o",
            temperature=0.6,
            metadata={
                "type": "project_manager", 
                "methodologies": pm_methodologies,
                "specialization": "project_coordination"
            }
        )


class AgentFactory:
    """Factory class for creating and managing specialized agents."""
    
    def __init__(self, orchestrator):
        """
        Initialize the agent factory.
        
        Args:
            orchestrator: MasterAgentOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.templates = AgentTemplates()
    
    async def create_data_analysis_team(
        self,
        tools: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Create a complete data analysis team.
        
        Args:
            tools: List of tool names to assign to agents
            
        Returns:
            Dictionary mapping role names to agent IDs
        """
        team_tools = tools or []
        team = {}
        
        # Data analyst
        analyst_config = self.templates.data_analyst_agent()
        team["data_analyst"] = await self.orchestrator.create_agent(
            analyst_config, team_tools
        )
        
        # Documentation writer for reports
        doc_config = self.templates.documentation_agent(
            name="Report Writer",
            doc_types=["analysis reports", "data summaries", "visualizations"]
        )
        team["report_writer"] = await self.orchestrator.create_agent(
            doc_config, team_tools
        )
        
        return team
    
    async def create_development_team(
        self,
        languages: Optional[List[str]] = None,
        tools: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Create a complete software development team.
        
        Args:
            languages: Programming languages to focus on
            tools: List of tool names to assign to agents
            
        Returns:
            Dictionary mapping role names to agent IDs
        """
        team_tools = tools or []
        team = {}
        
        # System architect
        architect_config = self.templates.system_architect_agent()
        team["architect"] = await self.orchestrator.create_agent(
            architect_config, team_tools
        )
        
        # Code reviewer
        reviewer_config = self.templates.code_reviewer_agent(
            programming_languages=languages
        )
        team["code_reviewer"] = await self.orchestrator.create_agent(
            reviewer_config, team_tools
        )
        
        # Testing specialist
        tester_config = self.templates.testing_agent()
        team["tester"] = await self.orchestrator.create_agent(
            tester_config, team_tools
        )
        
        # Documentation writer
        doc_config = self.templates.documentation_agent()
        team["documentation"] = await self.orchestrator.create_agent(
            doc_config, team_tools
        )
        
        # Project manager
        pm_config = self.templates.project_manager_agent()
        team["project_manager"] = await self.orchestrator.create_agent(
            pm_config, team_tools
        )
        
        return team
    
    async def create_custom_agent(
        self,
        name: str,
        description: str,
        instructions: str,
        specialization: str,
        tools: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Create a custom agent with specific configuration.
        
        Args:
            name: Agent name
            description: Agent description
            instructions: Detailed instructions
            specialization: Specialization type
            tools: List of tool names
            **kwargs: Additional configuration options
            
        Returns:
            Agent ID
        """
        config = AgentConfig(
            name=name,
            description=description,
            instructions=instructions,
            metadata={"type": "custom", "specialization": specialization},
            **kwargs
        )
        
        return await self.orchestrator.create_agent(config, tools)
    
    def get_available_templates(self) -> List[str]:
        """Get list of available agent templates."""
        return [
            "data_analyst_agent",
            "code_reviewer_agent", 
            "system_architect_agent",
            "testing_agent",
            "documentation_agent",
            "project_manager_agent"
        ]
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template information or None if not found
        """
        template_map = {
            "data_analyst_agent": {
                "description": "Analyzes data, identifies patterns, creates visualizations",
                "specialization": "Data analysis and insights",
                "recommended_tools": ["data_processing", "visualization", "statistics"]
            },
            "code_reviewer_agent": {
                "description": "Reviews code quality, security, and best practices",
                "specialization": "Code quality assurance",
                "recommended_tools": ["code_analysis", "security_scan", "linting"]
            },
            "system_architect_agent": {
                "description": "Designs system architecture and technical specifications",
                "specialization": "System design and architecture",
                "recommended_tools": ["architecture_tools", "documentation", "modeling"]
            },
            "testing_agent": {
                "description": "Creates test strategies, automated tests, and QA processes",
                "specialization": "Quality assurance and testing",
                "recommended_tools": ["testing_frameworks", "automation", "performance_testing"]
            },
            "documentation_agent": {
                "description": "Creates technical documentation and user guides",
                "specialization": "Technical writing and documentation",
                "recommended_tools": ["documentation_tools", "markdown", "api_docs"]
            },
            "project_manager_agent": {
                "description": "Manages projects, coordinates tasks, and tracks progress",
                "specialization": "Project coordination and management",
                "recommended_tools": ["project_tracking", "scheduling", "reporting"]
            }
        }
        
        return template_map.get(template_name)