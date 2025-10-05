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
from langchain_openai import ChatOpenAI
import os

from .specialized_agent import SpecializedAgent, AgentType, Task, TaskStatus, LangGraphState

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectManagementAgent(SpecializedAgent):
    """Project management agent implementation that simulates Notion integration and project tracking"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.PROJECT_MANAGEMENT, config)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=4000,
            temperature=0.1
        )
        
        # Project Management doesn't use external tools in this implementation
        self.tools = []
    
    def _execute_task(self, task: Dict[str, Any], state: LangGraphState) -> Dict[str, Any]:
        """Execute project management tasks"""
        logger.info(f"Project management agent executing task: {task.get('description', 'Unknown task')}. Current phase: {state['workflow_phase']}")
        
        try:
            task_desc_raw = task.get("description", "") or task.get("task_type", "")
            task_desc = str(task_desc_raw).lower()
            
            if "notion_integration" in task_desc or state['workflow_phase'] == "notion_integration":
                logger.info("Starting Notion integration phase")
                
                # Get project information
                project_name = state.get('project_metadata', {}).get('project_name', 'Content Project')
                content_request = state.get('user_request', 'Content Request')
                
                # Simulate creating a Notion workspace
                # In a real implementation, this would call Notion API
                notion_project_id = f"notion_proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                notion_workspace_url = f"https://notion.so/workspace/{notion_project_id}"
                
                # Create mock Notion pages and databases
                notion_pages = [
                    {
                        "page_id": f"page_{notion_project_id}_overview",
                        "title": f"{project_name} - Project Overview",
                        "url": f"{notion_workspace_url}/overview"
                    },
                    {
                        "page_id": f"page_{notion_project_id}_research",
                        "title": f"{project_name} - Research Data",
                        "url": f"{notion_workspace_url}/research"
                    },
                    {
                        "page_id": f"page_{notion_project_id}_assets",
                        "title": f"{project_name} - Assets",
                        "url": f"{notion_workspace_url}/assets"
                    }
                ]
                
                # Update state with Notion information
                state['notion_project_id'] = notion_project_id
                state['notion_workspace_url'] = notion_workspace_url
                state['notion_pages'] = notion_pages
                
                result = {
                    "success": True,
                    "notion_workspace_created": True,
                    "project_tracking_setup": True,
                    "milestones_defined": 6,
                    "team_coordination_complete": True,
                    "notion_project_id": notion_project_id,
                    "notion_workspace_url": notion_workspace_url,
                    "notion_pages_created": len(notion_pages),
                    "notion_status": "active"
                }
                
                logger.info("Notion integration completed successfully")
                return result
            
            elif "finalize" in task_desc or state['workflow_phase'] == "finalize":
                logger.info("Starting finalization phase")
                
                # Perform final project checks
                assets_count = len(state.get('assets_generated', []))
                content_generated = bool(state.get('content_generated'))
                script_created = bool(state.get('script'))
                
                # Update project status
                state['workflow_completed'] = True
                state['final_result'] = {
                    "project_completed": True,
                    "assets_generated": assets_count,
                    "content_generated": content_generated,
                    "script_created": script_created,
                    "completion_timestamp": datetime.now().isoformat()
                }
                
                result = {
                    "success": True,
                    "project_finalized": True,
                    "assets_count": assets_count,
                    "content_generated": content_generated,
                    "script_created": script_created,
                    "completion_timestamp": datetime.now().isoformat()
                }
                
                logger.info("Finalization completed successfully")
                return result
            
            else:
                logger.error(f"Unknown project management task: {task.get('description', 'Unknown task')}")
                return {
                    "success": False,
                    "error": f"Unknown project management task: {task.get('description', 'Unknown task')}"
                }
                
        except Exception as e:
            logger.error(f"Error during project management execution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "project_management_completed": False
            }
    
    def _update_agent_state(self, state: LangGraphState, result: Dict[str, Any]) -> None:
        """Update agent state with project management results"""
        logger.info(f"Project management agent updating agent state. Current phase: {state['workflow_phase']}")
        
        # Update current phase to advance the workflow
        if state['workflow_phase'] == "notion_integration":
            state['notion_project_id'] = result.get("notion_project_id", "")
            state['notion_status'] = result.get("notion_status", "active")
            state['workflow_phase'] = "finalize"
            logger.info("Advanced phase from notion_integration to finalize")
        elif state['workflow_phase'] == "finalize":
            state['workflow_completed'] = True
            state['workflow_phase'] = "completed"
            logger.info("Advanced phase from finalize to completed")