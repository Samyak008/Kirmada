from __future__ import annotations
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import yaml
import json
from datetime import datetime
import logging
from enum import Enum
from typing import TypedDict

# Define necessary enums and models
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    RESEARCH = "research"
    CONTENT = "content"
    ASSET_GENERATION = "asset_generation"
    STORAGE = "storage"
    PROJECT_MANAGEMENT = "project_management"

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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LangGraphState(TypedDict):
    """LangGraph state for the agentic workflow"""
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
    agent_state: Optional[Dict[str, Any]]
    current_agent: Optional[AgentType]
    next_agent: Optional[AgentType]
    task_results: Dict[str, Any]
    errors: List[Dict[str, Any]]
    workflow_completed: bool


class SpecializedAgent:
    """Base class for specialized agents (Research, Content, Asset Generation, etc.)"""
    
    def __init__(self, agent_type: AgentType, config: Dict[str, Any] = None):
        self.agent_type = agent_type
        if config:
            self.config = config
        else:
            # Use the system prompt from agent1.py for each agent
            agent_prompts = {
                AgentType.SUPERVISOR: "You are a supervisor agent specialized in orchestrating the content production workflow. Your responsibilities include understanding user content requests, creating production plans, assigning tasks to specialized agents, monitoring progress and handling issues, making strategic decisions about content direction, and ensuring all deliverables meet quality standards before finalization.",
                AgentType.RESEARCH: "You are a research agent specialized in finding high-quality information and content. Your responsibilities include searching for trending tech articles, extracting content from URLs, analyzing and summarizing information, and verifying accuracy.",
                AgentType.CONTENT: "You are a content agent specialized in creating engaging scripts and planning visual elements. Your responsibilities include generating compelling scripts, breaking down scripts into shots, analyzing pacing, creating hooks and calls to action, and planning visual elements.",
                AgentType.ASSET_GENERATION: "You are an asset generation agent specialized in creating visual and audio elements. Your responsibilities include generating AI image prompts, creating images, converting scripts to voiceover recordings, finding b-roll footage, and organizing assets.",
                AgentType.STORAGE: "You are a storage agent specialized in data management and organization. Your responsibilities include storing content in Supabase, organizing files in Google Drive, managing metadata and tagging, ensuring data persistence, and creating organized folder structures.",
                AgentType.PROJECT_MANAGEMENT: "You are a project management agent specialized in tracking and reporting. Your responsibilities include setting up Notion workspace for team collaboration, creating project tracking, updating project status, generating reports on project progress, and facilitating workflow between team members."
            }
            self.config = {"agents": {agent_type.value: {"system_prompt": agent_prompts[agent_type]}}}
        
        self.system_prompt = self.config["agents"][agent_type.value]["system_prompt"]
        self.tools = []
    
    def process(self, state: LangGraphState) -> LangGraphState:
        """Process agent-specific tasks"""
        logger.info(f"{self.agent_type.value} agent processing. Current phase: {state['workflow_phase']}")
        
        # Get pending tasks for this agent
        agent_tasks = state.get("task_results", {}).get(self.agent_type.value, [])
        pending_tasks = [task for task in agent_tasks if task.get('status', 'pending') == 'pending']
        
        logger.info(f"{self.agent_type.value} agent found {len(pending_tasks)} pending tasks")
        
        if not pending_tasks:
            logger.info(f"{self.agent_type.value} agent has no pending tasks, returning state")
            # For backward compatibility, we might still need to do some processing
            # even without explicit tasks if we're in a phase that requires action
            if self._should_process_without_task(state):
                result = self._execute_without_task(state)
                self._update_agent_state(state, result)
            return state
        
        # Process the highest priority task
        task = max(pending_tasks, key=lambda t: t.get('priority', 1))
        if 'status' in task:
            task['status'] = 'in_progress'
        task['started_at'] = datetime.now().isoformat()
        
        logger.info(f"{self.agent_type.value} agent starting task: {task.get('description', 'Unknown task')}")
        
        try:
            # Execute agent-specific logic
            result = self._execute_task(task, state)
            logger.info(f"{self.agent_type.value} agent completed task execution. Success: {result.get('success', True)}")
            
            # Update task status
            task['status'] = 'completed'
            task['completed_at'] = datetime.now().isoformat()
            task['result'] = result
            
            # Update state with results
            self._update_agent_state(state, result)
            
            logger.info(f"{self.agent_type.value} agent completed task successfully. New phase: {state.get('workflow_phase')}")
            
        except Exception as e:
            logger.error(f"{self.agent_type.value} agent failed task: {str(e)}")
            # Handle task failure
            task['status'] = 'failed'
            task['error_message'] = str(e)
            
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append({
                "agent": self.agent_type.value,
                "task": task.get('task_id', 'unknown'),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def _should_process_without_task(self, state: LangGraphState) -> bool:
        """Check if agent should process even without explicit tasks"""
        # If we're in the agent's phase but no tasks are defined, 
        # we may still need to process based on the state
        phase_to_agent = {
            "search": AgentType.RESEARCH,
            "crawl": AgentType.RESEARCH,
            "generate_script": AgentType.CONTENT,
            "shot_analysis": AgentType.CONTENT,
            "parallel_generation": AgentType.ASSET_GENERATION,
            "visual_table_generation": AgentType.CONTENT,
            "asset_gathering": AgentType.STORAGE,
            "store_article": AgentType.STORAGE,
            "store_script": AgentType.STORAGE,
            "notion_integration": AgentType.PROJECT_MANAGEMENT,
            "finalize": AgentType.SUPERVISOR
        }
        
        current_phase = state.get('workflow_phase', 'initial')
        agent_for_phase = phase_to_agent.get(current_phase)
        
        return agent_for_phase == self.agent_type
    
    def _execute_without_task(self, state: LangGraphState) -> Dict[str, Any]:
        """Execute default processing when no explicit task is defined"""
        # By default, just return an empty success
        return {"success": True, "message": f"{self.agent_type.value} processed default action"}

    def _execute_task(self, task: Dict[str, Any], state: LangGraphState) -> Dict[str, Any]:
        """Execute agent-specific task logic - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_task")
    
    def _update_agent_state(self, state: LangGraphState, result: Dict[str, Any]) -> None:
        """Update agent state with task results - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _update_agent_state")