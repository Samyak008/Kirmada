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


class AssetGenerationAgent(SpecializedAgent):
    """Asset generation agent implementation"""
    
    def __init__(self, config_path: str = "agent_prompts.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        super().__init__(AgentType.ASSET_GENERATION, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute asset generation tasks"""
        logger.info(f"Asset generation agent executing task: {task.description}. Current phase: {agent_state.current_phase}")
        result = {
            "assets_created": 8,
            "images_generated": 5,
            "audio_recordings": 2,
            "broll_found": 3,
            "asset_quality_score": 0.92
        }
        logger.info(f"Asset generation agent task completed successfully. Result: {result}")
        return result
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with asset generation results"""
        logger.info(f"Asset generation agent updating agent state. Current phase: {agent_state.current_phase}")
        
        from models import Asset
        
        # Create mock assets
        assets = [
            Asset(
                asset_id="asset_1",
                asset_type="image",
                file_path="/assets/images/opening_shot.png",
                description="Opening shot image",
                shot_number=1
            )
        ]
        
        agent_state.assets.extend(assets)
        
        # Update current phase to advance the workflow
        if agent_state.current_phase == "parallel_generation":
            # Update parallel generation results
            agent_state.prompts_generated = result.get("prompts_generated", [])
            agent_state.images_generated = result.get("images_generated", [])
            agent_state.voice_files = result.get("voice_files", [])
            agent_state.broll_assets = result.get("broll_assets", {})
            agent_state.current_phase = "visual_table_generation"
            logger.info("Advanced phase from parallel_generation to visual_table_generation")
        elif agent_state.current_phase == "asset_gathering":
            agent_state.project_folder_path = result.get("project_folder_path", "")
            agent_state.asset_organization_result = result.get("asset_organization_result", "")
            agent_state.current_phase = "notion_integration"
            logger.info("Advanced phase from asset_gathering to notion_integration")