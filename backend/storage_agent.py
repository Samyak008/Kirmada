from __future__ import annotations
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import yaml
import json
from datetime import datetime
import logging
from models import (
    AgentState, AgentType, TaskStatus, WorkflowPhase, 
    AgentMessage, SupervisorDecision, QualityCheck, Task
)
from tools import get_tools_for_agent, validate_tool_input, ToolResult
from specialized_agent import SpecializedAgent
from typing import TypedDict


class WorkflowState(TypedDict):
    """LangGraph state for the agentic workflow"""
    agent_state: AgentState
    messages: List[BaseMessage]
    current_agent: Optional[AgentType]
    next_agent: Optional[AgentType]
    task_results: Dict[str, Any]
    errors: List[Dict[str, Any]]
    workflow_completed: bool


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StorageAgent(SpecializedAgent):
    """Storage agent implementation"""
    
    def __init__(self, config_path: str = "agent_prompts.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        super().__init__(AgentType.STORAGE, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute storage tasks"""
        logger.info(f"Storage agent executing task: {task.description}. Current phase: {agent_state.current_phase}")
        result = {
            "data_stored": True,
            "folders_created": 3,
            "files_uploaded": 12,
            "metadata_tagged": True
        }
        logger.info(f"Storage agent task completed successfully. Result: {result}")
        return result
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with storage results"""
        logger.info(f"Storage agent updating agent state. Current phase: {agent_state.current_phase}")
        
        from models import ProjectFolder, DatabaseRecord
        
        # Create mock storage records
        folder = ProjectFolder(
            folder_id="project_folder_1",
            folder_name=agent_state.project_name,
            google_drive_path="/Content Production/Projects/",
            files=["file1", "file2"]
        )
        
        agent_state.project_folders.append(folder)
        
        # Update current phase to advance the workflow
        if agent_state.current_phase == "store_article":
            agent_state.current_phase = "generate_script"
            logger.info("Advanced phase from store_article to generate_script")
        elif agent_state.current_phase == "store_script":
            agent_state.current_phase = "shot_analysis"
            logger.info("Advanced phase from store_script to shot_analysis")