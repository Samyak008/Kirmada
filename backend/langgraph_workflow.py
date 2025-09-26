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
from enum import Enum
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


class InputType(str, Enum):
    """Type of input provided by user"""
    PROMPT = "prompt"
    STATEMENT = "statement"
    DOCUMENT = "document"
    WEBSITE = "website"
    VIDEO = "video"


class UserInput(BaseModel):
    """Schema for initial input shown in LangGraph Studio."""
    input_type: InputType = Field(default=InputType.STATEMENT, description="Type of input provided")
    content: str = Field(default="Create a short video about AI trends.", description="The main content input from the user")
    project_name: str = Field(default="Demo Project")
    content_type: str = Field(default="youtube_video", description="Type of content to create")
    target_audience: str = Field(default="tech enthusiasts", description="Target audience for the content")
    deadline: Optional[str] = Field(default=None, description="ISO datetime string")
    document_content: Optional[str] = Field(default=None, description="Content of uploaded document")
    website_url: Optional[str] = Field(default=None, description="URL of website to process")
    video_url: Optional[str] = Field(default=None, description="URL of video to process")
    project_id: Optional[str] = Field(default=None, description="Project ID if starting from existing project")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the workflow")


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
    
    # Initialize default values
    project_id = "proj_demo"
    project_name = "Demo Project"
    content_request = "Create a short video about AI trends."
    content_type = "youtube_video"
    target_audience = "tech enthusiasts"
    deadline = "2099-12-31T23:59:59"
    
    # Check if inp is a UserInput Pydantic model
    if isinstance(inp, UserInput):
        inbound = inp.dict()
        
        # Extract values from UserInput
        project_name = inbound.get("project_name", project_name)
        content_request = inbound.get("content", content_request)  # Changed from content_request to content
        content_type = inbound.get("content_type", content_type)
        target_audience = inbound.get("target_audience", target_audience)
        deadline_str = inbound.get("deadline", deadline)
        
        # Process different input types
        if hasattr(inp, 'input_type'):
            input_type = inp.input_type
            if input_type == InputType.DOCUMENT and inbound.get("document_content"):
                content_request = inbound["document_content"]
            elif input_type == InputType.WEBSITE and inbound.get("website_url"):
                content_request = f"Process content from website: {inbound['website_url']}"
            elif input_type == InputType.VIDEO and inbound.get("video_url"):
                content_request = f"Process content from video: {inbound['video_url']}"
            else:
                # For statement or prompt, use the content field
                content_request = inbound.get("content", content_request)
        
        # Generate project ID based on timestamp if not provided
        if not inbound.get("project_id"):
            project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            project_id = inbound["project_id"]
        
        # Parse deadline if provided
        try:
            if deadline_str:
                import datetime as dt
                deadline = dt.datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))
            else:
                deadline = dt.datetime(2099, 12, 31, 23, 59, 59)
        except:
            import datetime as dt
            deadline = dt.datetime(2099, 12, 31, 23, 59, 59)
        
    elif isinstance(inp, dict):
        inbound = inp
        project_id = inbound.get("project_id", f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        project_name = inbound.get("project_name", project_name)
        content_request = inbound.get("content", content_request)  # Changed from content_request to content
        content_type = inbound.get("content_type", content_type)
        target_audience = inbound.get("target_audience", target_audience)
        
        # Handle different input types from dict
        input_type = inbound.get("input_type", "statement")
        if input_type == "document" and inbound.get("document_content"):
            content_request = inbound["document_content"]
        elif input_type == "website" and inbound.get("website_url"):
            content_request = f"Process content from website: {inbound['website_url']}"
        elif input_type == "video" and inbound.get("video_url"):
            content_request = f"Process content from video: {inbound['video_url']}"
        else:
            content_request = inbound.get("content", content_request)
        
        deadline_str = inbound.get("deadline", deadline)
        try:
            import datetime as dt
            if deadline_str:
                deadline = dt.datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))
            else:
                deadline = dt.datetime(2099, 12, 31, 23, 59, 59)
        except:
            deadline = dt.datetime(2099, 12, 31, 23, 59, 59)
    else:
        # If it's neither a UserInput nor a dict, treat as empty
        inbound = {}
    
    return AgentState(
        project_id=project_id,
        project_name=project_name,
        content_request=content_request,
        content_type=content_type,
        target_audience=target_audience,
        deadline=deadline,
        current_phase="search",  # Start with search phase to analyze statement
        current_step="initialize",
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
    
    # Initialize messages list
    messages = []
    if isinstance(inbound, dict) and 'messages' in inbound:
        messages = inbound['messages']
    elif hasattr(inbound, 'messages'):
        messages = inbound.messages
    
    # Add initial message from user input
    if isinstance(inbound, UserInput):
        user_message = HumanMessage(content=f"User input: {inbound.content}")
        messages.append(user_message)
    elif isinstance(inbound, dict):
        content = inbound.get('content', 'No input provided')
        user_message = HumanMessage(content=f"User input: {content}")
        messages.append(user_message)
    
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
    from agents.supervisor_agent import SupervisorAgent
    from agents.research_agent import ResearchAgent
    from agents.content_agent import ContentAgent
    from agents.asset_generation_agent import AssetGenerationAgent
    from agents.storage_agent import StorageAgent
    from agents.project_management_agent import ProjectManagementAgent
    
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
    
    # Example using different input types:
    
    # 1. Statement input (simplified approach)
    user_input_statement = UserInput(
        input_type=InputType.STATEMENT,
        content="I want to create a YouTube video about the latest AI trends in 2025",
        project_name="AI Trends Video",
        content_type="youtube_video",
        target_audience="tech enthusiasts"
    )
    
    # 2. Document input
    document_content = """
    # Research Summary: AI in Healthcare
    
    Recent advances in artificial intelligence have revolutionized healthcare...
    Key findings include improved diagnostic accuracy and patient outcomes...
    """
    user_input_document = UserInput(
        input_type=InputType.DOCUMENT,
        content="Please create content based on the following research document",
        project_name="Healthcare AI Content",
        document_content=document_content,
        content_type="blog_article",
        target_audience="medical professionals"
    )
    
    # 3. Website input
    user_input_website = UserInput(
        input_type=InputType.WEBSITE,
        content="Create a presentation about this research paper",
        project_name="Website Content",
        website_url="https://example.com/ai-research-2025",
        content_type="presentation",
        target_audience="researchers"
    )
    
    # 4. Video input
    user_input_video = UserInput(
        input_type=InputType.VIDEO,
        content="Create a summary and social media content from this video",
        project_name="Video Content Analysis",
        video_url="https://example.com/interview-2025",
        content_type="social_media_post",
        target_audience="general audience"
    )
    
    # Run the workflow with statement input as an example
    result = app.invoke(user_input_statement)
    print("Workflow completed!")
    print(f"Result: {result}")