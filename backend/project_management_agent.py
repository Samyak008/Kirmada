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


class ProjectManagementAgent(SpecializedAgent):
    """Project management agent implementation"""
    
    def __init__(self, config_path: str = "agent_prompts.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        super().__init__(AgentType.PROJECT_MANAGEMENT, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute project management tasks"""
        logger.info(f"Project management agent executing task: {task.description}. Current phase: {agent_state.current_phase}")
        result = {
            "notion_workspace_created": True,
            "project_tracking_setup": True,
            "milestones_defined": 6,
            "team_coordination_complete": True
        }
        logger.info(f"Project management agent task completed successfully. Result: {result}")
        return result
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with project management results"""
        logger.info(f"Project management agent updating agent state. Current phase: {agent_state.current_phase}")
        
        from models import NotionPage
        
        # Create mock Notion page
        page = NotionPage(
            page_id="notion_page_1",
            title=agent_state.project_name,
            content={"status": "active", "progress": 25}
        )
        
        agent_state.notion_pages.append(page)
        
        # Update current phase to advance the workflow
        if agent_state.current_phase == "notion_integration":
            agent_state.notion_project_id = result.get("notion_project_id", "")
            agent_state.notion_status = result.get("notion_status", "active")
            agent_state.current_phase = "finalize"
            logger.info("Advanced phase from notion_integration to finalize")