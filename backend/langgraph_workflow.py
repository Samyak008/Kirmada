from __future__ import annotations  # ensure annotations aren't evaluated at import time

import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Union
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import yaml
import json
from datetime import datetime
import logging
from models import (
    AgentState, AgentType, TaskStatus, WorkflowPhase, 
    AgentMessage, SupervisorDecision, QualityCheck, Task
)
from tools import get_tools_for_agent, validate_tool_input, ToolResult


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """LangGraph state for the agentic workflow"""
    agent_state: AgentState
    messages: List[BaseMessage]
    current_agent: Optional[AgentType]
    next_agent: Optional[AgentType]
    task_results: Dict[str, Any]
    errors: List[Dict[str, Any]]
    workflow_completed: bool


class UserInput(BaseModel):
    """Schema for initial input shown in LangGraph Studio."""
    project_id: str = Field(default="proj_demo")
    project_name: str = Field(default="Demo Project")
    content_request: str = Field(default="Create a short video about AI trends.")
    content_type: str = Field(default="youtube_video")
    target_audience: str = Field(default="tech enthusiasts")
    deadline: Optional[str] = Field(default=None, description="ISO datetime string")
    current_phase: str = Field(default="search")
    current_step: str = Field(default="initialize")
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class AgentRouter:
    """Handles routing between different agents based on current state"""
    
    def __init__(self, config_path: str = "agent_prompts.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def should_continue(self, state: WorkflowState) -> str:
        """Determine if workflow should continue and which agent to call next"""
        logger.info(f"AgentRouter checking if workflow should continue. Current phase: {state['agent_state'].current_phase}")
        
        agent_state = state["agent_state"]
        
        # Check if workflow is completed
        if agent_state.workflow_completed:
            logger.info("Workflow is completed, ending")
            return "end"
        
        # Check for errors that need handling
        if state["errors"]:
            logger.warning(f"Errors found in state: {len(state['errors'])} errors")
            return "error_handler"
        
        # Determine next agent based on current phase and completed tasks
        next_agent = self._determine_next_agent(agent_state)
        
        if next_agent is None:
            logger.info("No next agent determined, ending workflow")
            return "end"
        
        agent_name = f"agent_{next_agent.value}"
        logger.info(f"Router directing to: {agent_name}")
        return agent_name
    
    def _determine_next_agent(self, agent_state: AgentState) -> Optional[AgentType]:
        """Determine which agent should be called next based on current state"""
        logger.info(f"Determining next agent. Current phase: {agent_state.current_phase}")
        
        current_phase = agent_state.current_phase
        
        # Phase-based routing (matching original workflow)
        phase_routing = {
            "search": [AgentType.RESEARCH],
            "crawl": [AgentType.RESEARCH],
            "store_article": [AgentType.STORAGE],
            "generate_script": [AgentType.CONTENT],
            "store_script": [AgentType.STORAGE],
            "shot_analysis": [AgentType.CONTENT],
            "parallel_generation": [AgentType.ASSET_GENERATION],
            "visual_table_generation": [AgentType.CONTENT],
            "asset_gathering": [AgentType.STORAGE],
            "notion_integration": [AgentType.PROJECT_MANAGEMENT],
            "finalize": [AgentType.SUPERVISOR]
        }
        
        available_agents = phase_routing.get(current_phase, [])
        
        # Check which agents have pending tasks
        for agent in available_agents:
            agent_tasks = [task for task in agent_state.tasks 
                          if task.agent_type == agent and task.status == TaskStatus.PENDING]
            if agent_tasks:
                logger.info(f"Found {len(agent_tasks)} pending tasks for {agent.value} agent in phase {current_phase}")
                return agent
        
        # If no pending tasks, check if phase is complete and move to next phase
        if self._is_phase_complete(agent_state, current_phase):
            logger.info(f"Current phase '{current_phase}' is complete")
            next_phase = self._get_next_phase(current_phase)
            if next_phase:
                logger.info(f"Advancing from phase '{current_phase}' to '{next_phase}'")
                agent_state.current_phase = next_phase
                return self._determine_next_agent(agent_state)
        
        logger.info(f"No next agent found for phase '{current_phase}'")
        return None
    
    def _is_phase_complete(self, agent_state: AgentState, phase: str) -> bool:
        """Check if current phase is complete"""
        logger.debug(f"Checking if phase '{phase}' is complete")
        
        phase_agents = {
            "search": [AgentType.RESEARCH],
            "crawl": [AgentType.RESEARCH],
            "store_article": [AgentType.STORAGE],
            "generate_script": [AgentType.CONTENT],
            "store_script": [AgentType.STORAGE],
            "shot_analysis": [AgentType.CONTENT],
            "parallel_generation": [AgentType.ASSET_GENERATION],
            "visual_table_generation": [AgentType.CONTENT],
            "asset_gathering": [AgentType.STORAGE],
            "notion_integration": [AgentType.PROJECT_MANAGEMENT],
            "finalize": [AgentType.SUPERVISOR]
        }
        
        required_agents = phase_agents.get(phase, [])
        for agent in required_agents:
            agent_tasks = [task for task in agent_state.tasks 
                          if task.agent_type == agent and task.status != TaskStatus.COMPLETED]
            if agent_tasks:
                logger.debug(f"Phase '{phase}' not complete: {len(agent_tasks)} tasks remaining for {agent.value}")
                return False
        
        logger.info(f"Phase '{phase}' is complete")
        return True
    
    def _get_next_phase(self, current_phase: str) -> Optional[str]:
        """Get the next phase in the workflow"""
        phase_sequence = [
            "search", "crawl", "store_article", "generate_script", 
            "store_script", "shot_analysis", "parallel_generation",
            "visual_table_generation", "asset_gathering", 
            "notion_integration", "finalize"
        ]
        
        try:
            current_index = phase_sequence.index(current_phase)
            if current_index < len(phase_sequence) - 1:
                return phase_sequence[current_index + 1]
        except ValueError:
            pass
        
        return None


# Simple no-op router node so conditional routing has a source node
def router_node(state: WorkflowState) -> WorkflowState:
    logger.debug("Router node processing state")
    return state


def _default_agent_state_from_input(inp: Union[UserInput, Dict[str, Any]]) -> AgentState:
    """Build a minimal AgentState if one wasn't provided."""
    logger.info("Building default agent state from input")
    # Check if inp is a UserInput Pydantic model
    if isinstance(inp, UserInput):
        inbound = inp.dict()
    elif isinstance(inp, dict):
        inbound = inp
    else:
        # If it's neither a UserInput nor a dict, treat as empty
        inbound = {}
    return AgentState(
        project_id=inbound.get("project_id", "proj_demo"),
        project_name=inbound.get("project_name", "Demo Project"),
        content_request=inbound.get("content_request", "Create a short video about AI trends."),
        content_type=inbound.get("content_type", "youtube_video"),
        target_audience=inbound.get("target_audience", "tech enthusiasts"),
        deadline=inbound.get("deadline", "2099-12-31T23:59:59"),
        current_phase=inbound.get("current_phase", "search"),
        current_step=inbound.get("current_step", "initialize"),
        tasks=[],
        supervisor_decisions=[],
        research_data=None,
        script=None,
        assets=[],
        project_folders=[],
        notion_pages=[],
        messages=[],
        errors=[],
        workflow_completed=False,
    )


def init_node(state: Union[UserInput, Dict[str, Any]]) -> WorkflowState:
    """Normalize user input into full WorkflowState."""
    logger.info("Initializing workflow")
    logger.debug(f"Raw state input: {type(state)}, value: {state}")
    
    inbound = state or {}
    
    # Check if agent_state is already provided in the inbound dict
    if isinstance(inbound, dict) and 'agent_state' in inbound:
        agent_state = inbound['agent_state']
        logger.debug(f"Using existing agent_state from inbound: {type(agent_state)}")
    else:
        # Create agent state from the inbound data or UserInput object
        agent_state = _default_agent_state_from_input(inbound)
        logger.debug(f"Created new agent_state: {type(agent_state)}")
    
    messages = inbound.get("messages", []) if isinstance(inbound, dict) else []
    
    # Log the type and attributes of agent_state before accessing them
    logger.debug(f"agent_state type: {type(agent_state)}")
    if hasattr(agent_state, 'project_name'):
        logger.info(f"Initialized workflow with project: {agent_state.project_name}, content: {agent_state.content_request[:50]}..., current phase: {agent_state.current_phase}")
    else:
        logger.error(f"agent_state does not have expected attributes. Type: {type(agent_state)}, Keys: {getattr(agent_state, '__dict__', 'No __dict__') if hasattr(agent_state, '__dict__') else 'No __dict__'}")
        # Create a default AgentState as fallback
        agent_state = _default_agent_state_from_input({})
        logger.info(f"Initialized workflow with project: {agent_state.project_name}, content: {agent_state.content_request[:50]}..., current phase: {agent_state.current_phase}")
    
    return WorkflowState(
        agent_state=agent_state,
        messages=messages,
        current_agent=None,
        next_agent=None,
        task_results={},
        errors=[],
        workflow_completed=False,
    )


def create_workflow_graph(config_path: str = "agent_prompts.yaml") -> StateGraph:
    """Create the LangGraph workflow"""
    logger.info(f"Creating workflow graph with config: {config_path}")
    
    # Import agents from their separate modules
    from supervisor_agent import SupervisorAgent
    from research_agent import ResearchAgent
    from content_agent import ContentAgent
    from asset_generation_agent import AssetGenerationAgent
    from storage_agent import StorageAgent
    from project_management_agent import ProjectManagementAgent
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize agents
    supervisor = SupervisorAgent(config_path)
    research_agent = ResearchAgent(config_path)
    content_agent = ContentAgent(config_path)
    asset_agent = AssetGenerationAgent(config_path)
    storage_agent = StorageAgent(config_path)
    project_agent = ProjectManagementAgent(config_path)
    
    # Initialize router
    router = AgentRouter(config_path)
    
    # Create the graph
    workflow = StateGraph(WorkflowState)
    workflow.add_node("init", init_node)
    # Add nodes
    workflow.add_node("supervisor", supervisor.process)
    workflow.add_node("agent_research", research_agent.process)
    workflow.add_node("agent_content", content_agent.process)
    workflow.add_node("agent_asset_generation", asset_agent.process)
    workflow.add_node("agent_storage", storage_agent.process)
    workflow.add_node("agent_project_management", project_agent.process)
    workflow.add_node("error_handler", lambda state: state)  # Placeholder for error handling
    workflow.add_node("router", router_node)  # NEW: define router node

    # Add edges
    workflow.add_edge("supervisor", "router")
    workflow.add_edge("agent_research", "router")
    workflow.add_edge("agent_content", "router")
    workflow.add_edge("agent_asset_generation", "router")
    workflow.add_edge("agent_storage", "router")
    workflow.add_edge("agent_project_management", "router")
    workflow.add_edge("error_handler", "router")
    workflow.add_edge("init", "supervisor")

    # Add conditional routing
    workflow.add_conditional_edges(
        "router",
        router.should_continue,
        {
            "agent_research": "agent_research",
            "agent_content": "agent_content",
            "agent_asset_generation": "agent_asset_generation",
            "agent_storage": "agent_storage",
            "agent_project_management": "agent_project_management",
            "agent_supervisor": "supervisor",  # NEW: matches f"agent_{AgentType.SUPERVISOR.value}"
            "error_handler": "error_handler",
            "end": END
        }
    )

    # Set entry point
    workflow.set_entry_point("init")
    logger.info("Workflow graph created successfully")
    return workflow


# Example usage
if __name__ == "__main__":
    # Create and compile the workflow
    graph = create_workflow_graph()
    app = graph.compile()
    
    # Example initial state
    initial_state = WorkflowState(
        agent_state=AgentState(
            project_id="proj_001",
            project_name="Tech Content Production",
            content_request="Create a video about AI trends in 2024",
            content_type="youtube_video",
            target_audience="tech enthusiasts"
        ),
        messages=[],
        current_agent=None,
        next_agent=None,
        task_results={},
        errors=[],
        workflow_completed=False
    )
    
    # Run the workflow
    result = app.invoke(initial_state)
    print("Workflow completed!")