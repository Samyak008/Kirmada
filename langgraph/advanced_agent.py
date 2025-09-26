"""
Advanced Agent that incorporates complex workflow logic from backend 
while maintaining compatibility with existing langgraph functionality
"""
import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Union
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import yaml
import json
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from enum import Enum

# Load environment variables
load_dotenv("../.env")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define necessary backend classes locally since models.py doesn't exist in langgraph
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ContentType(str, Enum):
    YOUTUBE_VIDEO = "youtube_video"
    SOCIAL_MEDIA_POST = "social_media_post"
    BLOG_ARTICLE = "blog_article"
    PODCAST = "podcast"
    PRESENTATION = "presentation"
    VIDEO_CONTENT = "video_content"

class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    RESEARCH = "research"
    CONTENT = "content"
    ASSET_GENERATION = "asset_generation"
    STORAGE = "storage"
    PROJECT_MANAGEMENT = "project_management"

class WorkflowPhase(BaseModel):
    phase_name: str
    description: str
    required_agents: List[AgentType]
    expected_duration_minutes: int
    success_criteria: List[str]
    dependencies: List[str] = []

class Task(BaseModel):
    task_id: str
    agent_type: AgentType
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: int = Field(default=1, ge=1, le=5)
    dependencies: List[str] = []  # task IDs this task depends on
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class AgentState(BaseModel):
    # Core project information
    project_id: str
    project_name: str
    content_request: str
    content_type: ContentType
    target_audience: str
    deadline: Optional[datetime] = None
    
    # Current workflow state
    current_phase: str = "planning"
    current_agent: Optional[AgentType] = None
    workflow_completed: bool = False
    
    # Research data
    research_data: Optional[Dict[str, Any]] = None
    
    # Content creation
    script: Optional[str] = None
    
    # Shot analysis (from original workflow)
    shot_breakdown: List[Dict[str, Any]] = []
    shot_timing: List[Dict[str, Any]] = []
    shot_types: List[str] = []
    
    # Asset generation tracking
    prompts_generated: List[Dict[str, Any]] = []
    images_generated: List[str] = []
    image_prompt_mapping: Dict[str, Dict[str, Any]] = {}
    voice_files: List[str] = []
    broll_assets: Dict[str, Any] = {}
    
    # Visual table generation
    visual_table: Optional[Dict[str, Any]] = None
    
    # Assets
    assets: List[Dict[str, Any]] = []  # Using dict instead of Asset for simplicity
    
    # Storage management
    project_folders: List[Dict[str, Any]] = []  # Using dict instead of ProjectFolder for simplicity
    database_records: List[Dict[str, Any]] = []  # Using dict instead of DatabaseRecord for simplicity
    project_folder_path: str = ""
    asset_organization_result: str = ""
    
    # Project management
    notion_pages: List[Dict[str, Any]] = []  # Using dict instead of NotionPage for simplicity
    notion_project_id: str = ""
    notion_status: str = ""
    tasks: List[Task] = []
    project_status: str = "active"
    
    # Workflow tracking
    completed_tasks: List[str] = []  # task IDs
    failed_tasks: List[str] = []
    next_actions: List[str] = []
    current_step: str = "search"  # From original workflow
    
    # Quality control
    quality_checks: List[Dict[str, Any]] = []  # Using dict instead of QualityCheck for simplicity
    revisions_needed: List[str] = []
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    # Agent communication
    agent_messages: List[Dict[str, Any]] = []
    supervisor_decisions: List[Dict[str, Any]] = []  # Using dict instead of SupervisorDecision for simplicity
    
    # Error handling
    errors: List[Dict[str, Any]] = []
    retry_count: int = 0
    max_retries: int = 3

# Define state structure compatible with existing langgraph flow
class AdvancedAgentState(TypedDict):
    """Advanced state that combines existing functionality with backend concepts"""
    messages: List[BaseMessage]
    user_request: str
    content_type: str
    content_generated: Optional[str]
    assets_generated: List[str]
    workflow_phase: str  # Current phase of the workflow
    completed_phases: List[str]
    error: Optional[str]
    final_result: Optional[str]
    user_preferences: Dict[str, Any]
    project_metadata: Dict[str, Any]
    research_data: Optional[Dict[str, Any]]
    script: Optional[str]
    image_prompts: List[str]
    voiceover: Optional[str]
    # Backend-style fields
    agent_state: Optional[AgentState]
    current_agent: Optional[AgentType]
    next_agent: Optional[AgentType]
    task_results: Dict[str, Any]
    errors: List[Dict[str, Any]]
    workflow_completed: bool

# Define tool functions that integrate with backend components
@tool
def search_articles_tool(query: str) -> str:
    """
    Search for trending articles using backend search infrastructure
    """
    # This would connect to the actual backend search infrastructure
    # For now, simulate with mock data
    mock_results = [
        {
            "url": f"https://example.com/article-{i}",
            "title": f"Article about {query} #{i}",
            "content": f"Content discussing {query} in detail, with data points and analysis",
            "relevance_score": 0.9 - i * 0.1
        }
        for i in range(1, 4)
    ]
    return json.dumps(mock_results)

@tool
def extract_article_content_tool(url: str) -> str:
    """
    Extract content from an article using backend crawling infrastructure
    """
    # This would connect to the actual backend crawling infrastructure
    # For now, simulate with mock data
    mock_content = {
        "url": url,
        "title": f"Title for {url}",
        "content": f"Full content extracted from {url} with detailed analysis",
        "images": [f"https://example.com/image-{i}.jpg" for i in range(1, 3)],
        "metadata": {"author": "Author", "date": "2024-01-01", "domain": "example.com"}
    }
    return json.dumps(mock_content)

@tool
def store_article_content_tool(article_data: str) -> str:
    """
    Store article content in database using backend storage infrastructure
    """
    # This would connect to the actual backend storage infrastructure
    # For now, simulate with mock response
    try:
        data = json.loads(article_data)
        mock_id = f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return f"Successfully stored article with ID: {mock_id}"
    except Exception as e:
        return f"Error storing article: {str(e)}"
# Enhanced system prompt that incorporates backend workflow concepts
SYSTEM_PROMPT = """
You are an advanced AI assistant that combines the simplicity of direct content creation with the robustness of multi-agent workflows.

Your capabilities include:

🔍 **RESEARCH AGENT FUNCTION:**
- Searches for relevant content using advanced search algorithms
- Extracts and analyzes content from provided URLs
- Validates information accuracy and quality

📝 **CONTENT GENERATION:**
- Creates scripts optimized for various platforms
- Generates visual timing plans and image prompts
- Produces voiceovers and other media elements

🗄️ **STORAGE & MANAGEMENT:**
- Stores content in structured databases
- Maintains metadata for future retrieval
- Manages project assets and organization

🎯 **WORKFLOW MANAGEMENT:**

**PHASE 1 - INITIALIZATION:**
- Understand the user's content request completely
- Determine the appropriate content type and structure
- Set up workflow tracking and project metadata

**PHASE 2 - RESEARCH & DISCOVERY:**
- Search for relevant content using available tools
- Extract content from selected sources
- Validate and structure information for processing

**PHASE 3 - CONTENT CREATION:**
- Generate scripts and visual elements
- Create timing plans and prompts
- Produce voiceovers and other assets

**PHASE 4 - ASSEMBLY & VALIDATION:**
- Combine all elements into final content
- Validate content quality and compliance
- Store final content and metadata

Your approach should be:
- Decompose complex requests into manageable tasks
- Maintain context throughout the workflow
- Handle errors gracefully with fallback options
- Optimize for quality and user satisfaction
"""

class AdvancedContentAgent:
    """
    Advanced content agent that combines the simplicity of react agent
    with the robustness of multi-agent workflows
    """
    def __init__(self, config_path: str = "agent_prompts.yaml"):
        # Initialize tools from existing system
        from agents.search_agent import search_tools
        from agents.crawl_agent import crawl_tools
        from agents.supabase_agent import supabase_tools_sync_wrapped
        from agents.prompt_generation_agent import prompt_generation_tools
        from agents.image_generation_agent import image_generation_tools
        from agents.voice_generation_agent import voice_tools
        from agents.voice_cloning_setup import voice_cloning_tools
        
        # Combine tools with new backend-integrated tools
        self.tools = (
            search_tools + crawl_tools + supabase_tools_sync_wrapped + 
            prompt_generation_tools + image_generation_tools + 
            voice_tools + voice_cloning_tools + [
                search_articles_tool,
                extract_article_content_tool,
                store_article_content_tool
            ]
        )
        
        # Initialize LLM (using same as existing system)
        from langchain_community.chat_models import ChatLiteLLM
        
        self.llm = ChatLiteLLM(
            model="openai/gpt-5-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=4000,
            temperature=0.1
        )
        
        # Create the react agent with enhanced tools
        from langgraph.prebuilt import create_react_agent
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            state_schema=AdvancedAgentState,
            prompt=SYSTEM_PROMPT
        )
    
    def process_request(self, state: AdvancedAgentState) -> AdvancedAgentState:
        """
        Process user request with advanced workflow capabilities
        """
        logger.info(f"Processing request in phase: {state.get('workflow_phase', 'initial')}")
        
        try:
            # Invoke the agent with the current state
            result = self.agent.invoke(state)
            
            # Update state with the result
            state.update(result)
            
            # Check if workflow is completed
            messages = state.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage):
                last_message = messages[-1]
                if "workflow completed" in last_message.content.lower() or "content created successfully" in last_message.content.lower():
                    state["workflow_phase"] = "completed"
                    state["project_metadata"]["status"] = "completed"
                    state["project_metadata"]["completed_at"] = datetime.now().isoformat()
                    state["workflow_completed"] = True
            
            return state
        except Exception as e:
            logger.error(f"Error in process_request: {e}")
            state["error"] = str(e)
            state["errors"] = state.get("errors", []) + [{"error": str(e), "timestamp": datetime.now().isoformat()}]
            return state

# Initialize the advanced agent
advanced_agent = AdvancedContentAgent()

# Define state management functions that mirror backend logic
def init_node(state: AdvancedAgentState) -> AdvancedAgentState:
    """Initialize state with backend-style project setup"""
    logger.info("Initializing advanced agent workflow")
    
    # Set up initial state
    state["workflow_phase"] = "initial"
    state["completed_phases"] = []
    state["assets_generated"] = []
    state["content_generated"] = None
    state["user_preferences"] = {}
    state["project_metadata"] = {
        "created_at": datetime.now().isoformat(),
        "project_id": f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "status": "in_progress",
        "request": state.get("user_request", "No request provided")
    }
    state["research_data"] = None
    state["script"] = None
    state["image_prompts"] = []
    state["voiceover"] = None
    state["task_results"] = {}
    state["errors"] = []
    state["workflow_completed"] = False
    
    # Initialize backend-style agent state if not present
    if not state.get("agent_state"):
        state["agent_state"] = AgentState(
            project_id=state["project_metadata"]["project_id"],
            project_name="Advanced Content Project",
            content_request=state["project_metadata"]["request"],
            content_type=state.get("content_type", "youtube"),
            target_audience="general",
            deadline=None,
            current_phase="initial",
            current_agent=None,
            workflow_completed=False,
            research_data=None,
            script=None,
            assets=[],
            project_folders=[],
            notion_pages=[],
            messages=[],
            errors=[],
        )
    
    return state

def research_node(state: AdvancedAgentState) -> AdvancedAgentState:
    """Research phase using backend-style task management"""
    logger.info("Starting research phase")
    
    # Update workflow phase
    state["workflow_phase"] = "research"
    state["agent_state"].current_phase = "research"
    
    # Use the advanced agent to handle research tasks
    user_request = state.get("user_request", "")
    user_message = HumanMessage(content=f"Research trending content about: {user_request}")
    
    # Create agent state for the research task
    agent_state = state.copy()
    agent_state["messages"] = [user_message]
    
    # Process the request with the advanced agent
    result = advanced_agent.process_request(agent_state)
    
    # Update main state with results
    state.update(result)
    
    # Track completed phase
    if "research" not in state.get("completed_phases", []):
        state["completed_phases"] = state.get("completed_phases", []) + ["research"]
        state["agent_state"].completed_tasks = state["agent_state"].completed_tasks + ["research_task"]
    
    return state

def content_creation_node(state: AdvancedAgentState) -> AdvancedAgentState:
    """Content creation phase using backend-style task management"""
    logger.info("Starting content creation phase")
    
    # Update workflow phase
    state["workflow_phase"] = "content_creation"
    state["agent_state"].current_phase = "content_creation"
    
    script = state.get("script")
    user_request = state.get("user_request", "")
    
    # If we don't have a script yet, generate one
    if not script:
        content_type = state.get("content_type", "youtube")
        
        # Create agent state for the content creation task
        user_message = HumanMessage(content=f"Generate a {content_type} script for: {user_request}")
        agent_state = state.copy()
        agent_state["messages"] = [user_message]
        
        # Process with advanced agent
        result = advanced_agent.process_request(agent_state)
        state.update(result)
    
    # Track completed phase
    if "content_creation" not in state.get("completed_phases", []):
        state["completed_phases"] = state.get("completed_phases", []) + ["content_creation"]
        state["agent_state"].completed_tasks = state["agent_state"].completed_tasks + ["content_creation_task"]
    
    return state

def visual_creation_node(state: AdvancedAgentState) -> AdvancedAgentState:
    """Visual creation phase using backend-style task management"""
    logger.info("Starting visual creation phase")
    
    # Update workflow phase
    state["workflow_phase"] = "visual_creation"
    state["agent_state"].current_phase = "visual_creation"
    
    script = state.get("script")
    
    if script:
        # Create agent state for the visual creation task
        user_message = HumanMessage(content=f"Generate image prompts based on this script: {script[:500]}...")
        agent_state = state.copy()
        agent_state["messages"] = [user_message]
        
        # Process with advanced agent
        result = advanced_agent.process_request(agent_state)
        state.update(result)
    
    # Track completed phase
    if "visual_creation" not in state.get("completed_phases", []):
        state["completed_phases"] = state.get("completed_phases", []) + ["visual_creation"]
        state["agent_state"].completed_tasks = state["agent_state"].completed_tasks + ["visual_creation_task"]
    
    return state

def validation_node(state: AdvancedAgentState) -> AdvancedAgentState:
    """Validation phase using backend-style task management"""
    logger.info("Starting validation phase")
    
    # Update workflow phase
    state["workflow_phase"] = "validation"
    state["agent_state"].current_phase = "validation"
    
    final_content = state.get("final_result", "")
    
    if final_content:
        # Create agent state for the validation task
        user_message = HumanMessage(content=f"Validate this content: {final_content[:500]}...")
        agent_state = state.copy()
        agent_state["messages"] = [user_message]
        
        # Process with advanced agent
        result = advanced_agent.process_request(agent_state)
        state.update(result)
    
    # Mark as completed
    state["workflow_phase"] = "completed"
    state["project_metadata"]["status"] = "completed"
    state["project_metadata"]["completed_at"] = datetime.now().isoformat()
    state["workflow_completed"] = True
    state["agent_state"].workflow_completed = True
    state["agent_state"].current_phase = "completed"
    
    # Track completed phase
    if "validation" not in state.get("completed_phases", []):
        state["completed_phases"] = state.get("completed_phases", []) + ["validation"]
        state["agent_state"].completed_tasks = state["agent_state"].completed_tasks + ["validation_task"]
    
    return state

# Supervisor node to manage workflow progression
def supervisor_node(state: AdvancedAgentState) -> AdvancedAgentState:
    """Supervisor node that manages workflow progression with backend-style logic"""
    logger.info(f"Supervisor checking workflow state. Current phase: {state.get('workflow_phase')}")
    
    # Determine the next phase based on current state
    current_phase = state.get("workflow_phase", "initial")
    completed_phases = state.get("completed_phases", [])
    
    # Define the sequence of phases
    phase_sequence = [
        "initial", 
        "research", 
        "content_creation", 
        "visual_creation", 
        "validation", 
        "completed"
    ]
    
    try:
        current_index = phase_sequence.index(current_phase)
        if current_index < len(phase_sequence) - 1:
            next_phase = phase_sequence[current_index + 1]
            state["workflow_phase"] = next_phase
            state["agent_state"].current_phase = next_phase
            logger.info(f"Supervisor advancing workflow from {current_phase} to {next_phase}")
        else:
            logger.info("Workflow completed")
    except ValueError:
        # If current phase is not in the sequence, start from research
        state["workflow_phase"] = "research"
        logger.info(f"Unknown phase '{current_phase}', advancing to research phase")
    
    return state

def create_advanced_workflow():
    """Create the advanced workflow graph that combines react agent simplicity with multi-agent robustness"""
    logger.info("Creating advanced workflow")
    
    # Create state graph
    workflow = StateGraph(AdvancedAgentState)
    
    # Add nodes
    workflow.add_node("init", init_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("content_creation", content_creation_node)
    workflow.add_node("visual_creation", visual_creation_node)
    workflow.add_node("validation", validation_node)
    
    # Set entry point
    workflow.set_entry_point("init")
    
    # Add edges
    workflow.add_edge("init", "supervisor")
    workflow.add_edge("research", "supervisor")
    workflow.add_edge("content_creation", "supervisor")
    workflow.add_edge("visual_creation", "supervisor")
    workflow.add_edge("validation", END)
    
    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("workflow_phase", "initial"),
        {
            "research": "research",
            "content_creation": "content_creation",
            "visual_creation": "visual_creation",
            "validation": "validation",
            "completed": END  # If phase is already completed, end the workflow
        }
    )
    
    return workflow

# Example usage
if __name__ == "__main__":
    # Create the advanced workflow
    workflow_graph = create_advanced_workflow()
    app = workflow_graph.compile()
    
    # Example: Simple prompt input
    initial_state = {
        "messages": [HumanMessage(content="Create a YouTube video about the latest AI trends in 2025")],
        "user_request": "Create a YouTube video about the latest AI trends in 2025",
        "content_type": "youtube",
        "content_generated": None,
        "assets_generated": [],
        "workflow_phase": "initial",
        "completed_phases": [],
        "error": None,
        "final_result": None,
        "user_preferences": {},
        "project_metadata": {},
        "research_data": None,
        "script": None,
        "image_prompts": [],
        "voiceover": None,
        "task_results": {},
        "errors": [],
        "workflow_completed": False
    }
    
    print("Running advanced consumer-focused workflow...")
    result = app.invoke(initial_state)
    
    print("\nAdvanced workflow completed!")
    print(f"Final result: {result.get('final_result', 'No final result generated')}")
    print(f"Completed phases: {result.get('completed_phases', [])}")
    print(f"Assets generated: {len(result.get('assets_generated', []))}")
    print(f"Project metadata: {result.get('project_metadata', {})}")
    print(f"Workflow completed: {result.get('workflow_completed', False)}")
"""
"""