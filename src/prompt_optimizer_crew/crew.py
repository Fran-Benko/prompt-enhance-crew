"""
Main Crew orchestration for the Prompt Optimizer system.
"""

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from crewai import Agent, Crew, Process, Task
from crewai.agents.parser import OutputParserException
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from .tools.domain_classifier_tool import DomainClassifierTool
from .tools.internet_search_tool import InternetSearchTool
from .utils.llm_loader import load_llm

# Load environment variables
load_dotenv()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="crewai_tools")

logger = logging.getLogger(__name__)


class OptimizationResult(BaseModel):
    """Result of prompt optimization."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    optimized_prompt: str
    domain: str
    agents_used: List[str]
    execution_time: float
    feedback_summary: Optional[Dict[str, str]] = None
    errors: Optional[List[str]] = None


class PromptOptimizerCrew:
    """
    Main crew orchestrator for prompt optimization.
    
    This class manages the multi-agent system that analyzes and optimizes prompts
    through specialized evaluator agents and a coordinator agent.
    """
    
    def __init__(
        self,
        use_quantization: Optional[str] = None,
        enable_internet_search: Optional[bool] = None,
        max_iterations: int = 3
    ):
        """
        Initialize the Prompt Optimizer Crew.
        
        Args:
            use_quantization: Quantization mode ('4bit', '8bit', or None)
            enable_internet_search: Whether to enable internet search for agents
            max_iterations: Maximum iterations for agent tasks
        """
        self.config_dir = Path(__file__).parent / "config"
        self.use_quantization = use_quantization or os.getenv("USE_QUANTIZATION", "4bit")
        self.enable_internet_search = (
            enable_internet_search
            if enable_internet_search is not None
            else os.getenv("ENABLE_INTERNET_SEARCH", "true").lower() == "true"
        )
        self.max_iterations = max_iterations
        
        # Load configurations
        self.agents_config = self._load_yaml(self.config_dir / "agents.yaml")
        self.tasks_config = self._load_yaml(self.config_dir / "tasks.yaml")
        self.domain_mapping = self._load_yaml(self.config_dir / "domain_agent_mapping.yaml")
        
        # Initialize tools
        self.domain_classifier = DomainClassifierTool()
        self.internet_search = InternetSearchTool() if self.enable_internet_search else None
        
        logger.info("Prompt Optimizer Crew initialized successfully")
    
    def _load_yaml(self, path: Path) -> dict:
        """Load YAML configuration file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return {}
    
    def _create_agent(self, agent_name: str, llm_name: Optional[str] = None) -> Agent:
        """
        Create an agent from configuration.
        
        Args:
            agent_name: Name of the agent from agents.yaml
            llm_name: Optional specific LLM to use for this agent
            
        Returns:
            Configured CrewAI Agent
        """
        agent_config = self.agents_config.get(agent_name, {})
        
        if not agent_config:
            raise ValueError(f"Agent configuration not found: {agent_name}")
        
        # Load LLM for this agent
        llm = load_llm(model_name=llm_name or agent_config.get("llm"))
        
        # Prepare tools
        tools = []
        if self.internet_search and agent_config.get("allow_internet_search", False):
            tools.append(self.internet_search)
        
        # Create agent
        agent = Agent(
            role=agent_config.get("role", agent_name),
            goal=agent_config.get("goal", ""),
            backstory=agent_config.get("backstory", ""),
            llm=llm,
            tools=tools,
            verbose=True,
            allow_delegation=agent_config.get("allow_delegation", False),
            max_iter=self.max_iterations
        )
        
        logger.debug(f"Created agent: {agent_name}")
        return agent
    
    def _create_task(self, task_name: str, agent: Agent, context: Dict) -> Task:
        """
        Create a task from configuration.
        
        Args:
            task_name: Name of the task from tasks.yaml
            agent: Agent to assign the task to
            context: Context variables for task description
            
        Returns:
            Configured CrewAI Task
        """
        task_config = self.tasks_config.get(task_name, {})
        
        if not task_config:
            raise ValueError(f"Task configuration not found: {task_name}")
        
        # Format description with context
        description = task_config.get("description", "").format(**context)
        expected_output = task_config.get("expected_output", "").format(**context)
        
        task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent
        )
        
        logger.debug(f"Created task: {task_name}")
        return task
    
    def _select_relevant_agents(self, domain: str) -> List[str]:
        """
        Select relevant agents based on domain.
        
        Args:
            domain: Detected domain of the prompt
            
        Returns:
            List of agent names relevant to the domain
        """
        domain_config = self.domain_mapping.get("domains", {}).get(domain, {})
        agents = domain_config.get("agents", [])
        
        # Always include coordinator
        if "coordinator" not in agents:
            agents.append("coordinator")
        
        logger.info(f"Selected agents for domain '{domain}': {agents}")
        return agents
    
    def optimize_prompt(
        self,
        prompt: str,
        domain: Optional[str] = None,
        selected_agents: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Optimize a prompt using the multi-agent system.
        
        Args:
            prompt: The prompt to optimize
            domain: Optional domain specification (auto-detected if not provided)
            selected_agents: Optional list of specific agents to use
            
        Returns:
            OptimizationResult with the optimized prompt and metadata
        """
        start_time = time.time()
        
        logger.info("Starting prompt optimization...")
        
        # Step 1: Classify domain if not provided
        if domain is None:
            domain = self.domain_classifier.classify(prompt)
            logger.info(f"Detected domain: {domain}")
        
        # Step 2: Select relevant agents
        if selected_agents is None:
            selected_agents = self._select_relevant_agents(domain)
        else:
            # Ensure coordinator is included
            if "coordinator" not in selected_agents:
                selected_agents.append("coordinator")
        
        # Step 3: Create evaluator agents and tasks
        evaluator_agents = []
        evaluator_tasks = []
        feedback_context = {"prompt": prompt, "domain": domain}
        
        for agent_name in selected_agents:
            if agent_name == "coordinator":
                continue  # Handle coordinator separately
            
            try:
                agent = self._create_agent(agent_name)
                task = self._create_task(
                    f"{agent_name}_evaluation",
                    agent,
                    feedback_context
                )
                evaluator_agents.append(agent)
                evaluator_tasks.append(task)
            except Exception as e:
                logger.warning(f"Failed to create agent/task for {agent_name}: {e}")
        
        # Step 4: Create coordinator agent and synthesis task
        coordinator_agent = self._create_agent("coordinator")
        
        # Prepare context for coordinator with feedback from evaluators
        synthesis_context = {
            "prompt": prompt,
            "domain": domain,
            "feedback": "Feedback will be collected from evaluator agents"
        }
        
        synthesis_task = self._create_task(
            "synthesis",
            coordinator_agent,
            synthesis_context
        )
        
        # Step 5: Execute the crew with error handling
        logger.info("Executing crew with agents: " + ", ".join(selected_agents))
        
        crew = Crew(
            agents=evaluator_agents + [coordinator_agent],
            tasks=evaluator_tasks + [synthesis_task],
            process=Process.sequential,
            verbose=True
        )
        
        errors = []
        optimized_prompt = prompt  # Fallback to original prompt
        
        try:
            result = crew.kickoff()
            optimized_prompt = str(result)
            
            execution_time = time.time() - start_time
            
            logger.info(f"Optimization completed in {execution_time:.2f}s")
            
            return OptimizationResult(
                optimized_prompt=optimized_prompt,
                domain=domain,
                agents_used=selected_agents,
                execution_time=execution_time,
                errors=errors if errors else None
            )
            
        except OutputParserException as e:
            error_msg = f"Agent output parsing error: {str(e)}"
            logger.warning(error_msg)
            errors.append(error_msg)
            
            # Try to continue with partial results
            execution_time = time.time() - start_time
            
            return OptimizationResult(
                optimized_prompt=optimized_prompt,
                domain=domain,
                agents_used=selected_agents,
                execution_time=execution_time,
                errors=errors
            )
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            raise
            
        except Exception as e:
            error_msg = f"Error during crew execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            
            execution_time = time.time() - start_time
            
            # Return partial result instead of raising
            return OptimizationResult(
                optimized_prompt=optimized_prompt,
                domain=domain,
                agents_used=selected_agents,
                execution_time=execution_time,
                errors=errors
            )
