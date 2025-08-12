"""
Dynamic Prompt Analyzer for Master Agent Orchestrator.

This module analyzes user prompts to determine required agents, tools, and workflows.
It uses AI to understand the requirements and dynamically creates appropriate agents.
"""

import asyncio
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from openai import AsyncOpenAI
from config import MasterConfig
from agent_factory import AgentConfig


class TaskCategory(Enum):
    """Categories of tasks that can be identified from prompts."""
    DATA_ANALYSIS = "data_analysis"
    CODE_REVIEW = "code_review"
    SYSTEM_DESIGN = "system_design"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    PROJECT_MANAGEMENT = "project_management"
    RESEARCH = "research"
    CONTENT_CREATION = "content_creation"
    PROBLEM_SOLVING = "problem_solving"
    AUTOMATION = "automation"


class AgentRole(Enum):
    """Types of agent roles that can be created."""
    ANALYST = "analyst"
    REVIEWER = "reviewer"
    ARCHITECT = "architect"
    TESTER = "tester"
    WRITER = "writer"
    MANAGER = "manager"
    RESEARCHER = "researcher"
    SPECIALIST = "specialist"


@dataclass
class AgentRequirement:
    """Requirements for creating an agent based on prompt analysis."""
    role: AgentRole
    name: str
    description: str
    specialization: str
    required_skills: List[str]
    tools_needed: List[str]
    priority: int = 1
    temperature: float = 0.7
    model: str = "gpt-4o"


@dataclass
class PromptAnalysis:
    """Result of analyzing a user prompt."""
    task_categories: List[TaskCategory]
    agent_requirements: List[AgentRequirement]
    workflow_steps: List[str]
    required_tools: List[str]
    complexity_score: int
    estimated_agents_needed: int
    prompt_intent: str
    key_deliverables: List[str]


class PromptAnalyzer:
    """
    Analyzes user prompts to determine required agents and workflows.
    Uses AI to understand complex requirements and create appropriate agent teams.
    """
    
    def __init__(self, config: MasterConfig):
        """Initialize the prompt analyzer."""
        self.config = config
        self.client = AsyncOpenAI(api_key=config.openai.api_key)
        self.logger = logging.getLogger(__name__)
        
        # Predefined patterns for quick matching
        self.category_patterns = {
            TaskCategory.DATA_ANALYSIS: [
                r'\b(analyz|data|statistic|metric|report|dashboard|visualiz)\w*',
                r'\b(csv|excel|dataset|database|sql)\b',
                r'\b(trend|pattern|insight|correlation)\w*'
            ],
            TaskCategory.CODE_REVIEW: [
                r'\b(code|review|refactor|bug|debug|optimize)\w*',
                r'\b(python|javascript|java|cpp|golang|rust)\b',
                r'\b(function|class|method|algorithm)\w*'
            ],
            TaskCategory.SYSTEM_DESIGN: [
                r'\b(architect|design|system|infrastructure|scalab)\w*',
                r'\b(api|microservice|database|cloud|deployment)\w*',
                r'\b(performance|security|reliability)\w*'
            ],
            TaskCategory.TESTING: [
                r'\b(test|qa|quality|validation|verification)\w*',
                r'\b(unit|integration|e2e|performance|security) test\w*',
                r'\b(automation|ci\/cd|pipeline)\w*'
            ],
            TaskCategory.DOCUMENTATION: [
                r'\b(document|write|explain|guide|tutorial)\w*',
                r'\b(api doc|readme|manual|specification)\w*',
                r'\b(help|instruction|how-to)\w*'
            ],
            TaskCategory.PROJECT_MANAGEMENT: [
                r'\b(manage|plan|coordinate|schedule|timeline)\w*',
                r'\b(project|task|milestone|deadline|resource)\w*',
                r'\b(agile|scrum|kanban|sprint)\w*'
            ]
        }
    
    async def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """
        Analyze a user prompt to determine required agents and workflow.
        
        Args:
            prompt: User's natural language prompt
            
        Returns:
            PromptAnalysis with detailed requirements
        """
        self.logger.info(f"Analyzing prompt: {prompt[:100]}...")
        
        # Step 1: Quick pattern matching for basic categorization
        initial_categories = self._quick_categorize(prompt)
        
        # Step 2: AI-powered deep analysis
        ai_analysis = await self._ai_analyze_prompt(prompt, initial_categories)
        
        # Step 3: Generate agent requirements
        agent_requirements = self._generate_agent_requirements(ai_analysis, prompt)
        
        # Step 4: Create final analysis
        analysis = PromptAnalysis(
            task_categories=ai_analysis.get('categories', initial_categories),
            agent_requirements=agent_requirements,
            workflow_steps=ai_analysis.get('workflow_steps', []),
            required_tools=ai_analysis.get('required_tools', []),
            complexity_score=ai_analysis.get('complexity_score', 5),
            estimated_agents_needed=len(agent_requirements),
            prompt_intent=ai_analysis.get('intent', 'General task completion'),
            key_deliverables=ai_analysis.get('deliverables', [])
        )
        
        self.logger.info(f"Analysis complete: {len(agent_requirements)} agents needed")
        return analysis
    
    def _quick_categorize(self, prompt: str) -> List[TaskCategory]:
        """Quick pattern-based categorization of the prompt."""
        categories = []
        prompt_lower = prompt.lower()
        
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    categories.append(category)
                    break
        
        return categories if categories else [TaskCategory.PROBLEM_SOLVING]
    
    async def _ai_analyze_prompt(self, prompt: str, initial_categories: List[TaskCategory]) -> Dict[str, Any]:
        """Use AI to perform deep analysis of the prompt."""
        
        analysis_prompt = f"""
        Analyze this user prompt and provide a detailed breakdown:
        
        USER PROMPT: "{prompt}"
        
        INITIAL CATEGORIES: {[cat.value for cat in initial_categories]}
        
        Please provide a JSON response with:
        1. "intent": Main goal/intent of the request
        2. "categories": List of relevant task categories from: {[cat.value for cat in TaskCategory]}
        3. "workflow_steps": Ordered list of steps needed to complete the task
        4. "required_tools": Tools/capabilities needed (e.g., "file_operations", "web_search", "data_analysis")
        5. "complexity_score": 1-10 scale of task complexity
        6. "deliverables": What the user expects as output
        7. "agent_roles": What types of agents would be needed
        8. "specializations": Specific skills or knowledge domains required
        
        Focus on being practical and specific. Consider what a real team would need to accomplish this task.
        
        Respond with valid JSON only.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response (handle cases where AI adds extra text)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            return json.loads(content)
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            # Fallback to basic analysis
            return {
                "intent": "Complete user request",
                "categories": [cat.value for cat in initial_categories],
                "workflow_steps": ["Understand requirements", "Execute task", "Provide results"],
                "required_tools": ["general"],
                "complexity_score": 5,
                "deliverables": ["Task completion"],
                "agent_roles": ["specialist"],
                "specializations": ["general"]
            }
    
    def _generate_agent_requirements(self, ai_analysis: Dict[str, Any], original_prompt: str) -> List[AgentRequirement]:
        """Generate specific agent requirements based on analysis."""
        requirements = []
        
        categories = [TaskCategory(cat) for cat in ai_analysis.get('categories', [])]
        agent_roles = ai_analysis.get('agent_roles', ['specialist'])
        specializations = ai_analysis.get('specializations', ['general'])
        complexity = ai_analysis.get('complexity_score', 5)
        
        # Map categories to agent types
        agent_mapping = {
            TaskCategory.DATA_ANALYSIS: (AgentRole.ANALYST, "data_analysis", ["statistical_analysis", "data_visualization", "reporting"]),
            TaskCategory.CODE_REVIEW: (AgentRole.REVIEWER, "code_quality", ["code_review", "security_analysis", "performance_optimization"]),
            TaskCategory.SYSTEM_DESIGN: (AgentRole.ARCHITECT, "system_architecture", ["system_design", "scalability", "infrastructure"]),
            TaskCategory.TESTING: (AgentRole.TESTER, "quality_assurance", ["test_automation", "qa_processes", "validation"]),
            TaskCategory.DOCUMENTATION: (AgentRole.WRITER, "technical_writing", ["documentation", "content_creation", "communication"]),
            TaskCategory.PROJECT_MANAGEMENT: (AgentRole.MANAGER, "project_coordination", ["planning", "resource_management", "coordination"]),
            TaskCategory.RESEARCH: (AgentRole.RESEARCHER, "research_analysis", ["information_gathering", "analysis", "synthesis"]),
            TaskCategory.CONTENT_CREATION: (AgentRole.WRITER, "content_development", ["writing", "creativity", "communication"]),
            TaskCategory.AUTOMATION: (AgentRole.SPECIALIST, "automation", ["scripting", "workflow_automation", "efficiency"])
        }
        
        # Create agents based on identified categories
        priority = 1
        for category in categories:
            if category in agent_mapping:
                role, spec, skills = agent_mapping[category]
                
                # Adjust temperature based on task type
                temp = 0.3 if category in [TaskCategory.CODE_REVIEW, TaskCategory.TESTING] else 0.7
                if category == TaskCategory.CONTENT_CREATION:
                    temp = 0.8
                
                requirement = AgentRequirement(
                    role=role,
                    name=f"{category.value.replace('_', ' ').title()} Agent",
                    description=f"Specialized agent for {category.value.replace('_', ' ')} tasks",
                    specialization=spec,
                    required_skills=skills,
                    tools_needed=ai_analysis.get('required_tools', ['general']),
                    priority=priority,
                    temperature=temp,
                    model="gpt-4o" if complexity > 7 else "gpt-4o"
                )
                requirements.append(requirement)
                priority += 1
        
        # If no specific categories, create a general specialist
        if not requirements:
            requirements.append(AgentRequirement(
                role=AgentRole.SPECIALIST,
                name="General Task Specialist",
                description="General-purpose agent for task completion",
                specialization="general_problem_solving",
                required_skills=specializations,
                tools_needed=ai_analysis.get('required_tools', ['general']),
                priority=1,
                temperature=0.7
            ))
        
        return requirements
    
    def create_agent_config(self, requirement: AgentRequirement, custom_prompt: str) -> AgentConfig:
        """Convert an AgentRequirement to an AgentConfig."""
        
        # Create specialized instructions based on the requirement and original prompt
        instructions = f"""
        You are a {requirement.name} with expertise in {requirement.specialization}.
        
        Your primary skills include: {', '.join(requirement.required_skills)}
        
        ORIGINAL USER REQUEST: "{custom_prompt}"
        
        Your role is to help accomplish this request by leveraging your specialized knowledge in:
        {requirement.specialization.replace('_', ' ')}
        
        Guidelines:
        1. Focus on your area of expertise while collaborating with other agents
        2. Provide detailed, actionable insights and recommendations
        3. Ask clarifying questions when you need more information
        4. Communicate clearly and professionally
        5. Consider both immediate needs and long-term implications
        
        Always strive for excellence and accuracy in your specialized domain.
        """
        
        return AgentConfig(
            name=requirement.name,
            description=requirement.description,
            instructions=instructions.strip(),
            tools=[],  # Tools will be assigned based on availability
            model=requirement.model,
            temperature=requirement.temperature,
            metadata={
                "type": requirement.role.value,
                "specialization": requirement.specialization,
                "skills": ", ".join(requirement.required_skills),
                "generated_from_prompt": "true"
            }
        )


class DynamicAgentCreator:
    """Creates agents dynamically based on prompt analysis."""
    
    def __init__(self, orchestrator, analyzer: PromptAnalyzer):
        """Initialize the dynamic agent creator."""
        self.orchestrator = orchestrator
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
    
    async def create_agents_for_prompt(self, prompt: str) -> Tuple[List[str], PromptAnalysis]:
        """
        Analyze a prompt and create appropriate agents.
        
        Args:
            prompt: User's natural language prompt
            
        Returns:
            Tuple of (agent_ids, analysis)
        """
        self.logger.info("Creating agents for custom prompt...")
        
        # Analyze the prompt
        analysis = await self.analyzer.analyze_prompt(prompt)
        
        # Create agents based on requirements
        agent_ids = []
        for requirement in analysis.agent_requirements:
            try:
                # Convert requirement to agent config
                agent_config = self.analyzer.create_agent_config(requirement, prompt)
                
                # Create the agent
                agent_id = await self.orchestrator.create_agent(
                    agent_config, 
                    tools=requirement.tools_needed
                )
                agent_ids.append(agent_id)
                
                self.logger.info(f"Created agent: {agent_config.name} (ID: {agent_id})")
                
            except Exception as e:
                self.logger.error(f"Failed to create agent {requirement.name}: {e}")
        
        self.logger.info(f"Successfully created {len(agent_ids)} agents for prompt")
        return agent_ids, analysis