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
from .specialized_agent import SpecializedAgent
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


class ContentAgent(SpecializedAgent):
    """Content agent implementation"""
    
    def __init__(self, config_path: str = "agent_prompts.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        super().__init__(AgentType.CONTENT, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute content creation tasks"""
        logger.info(f"Content agent executing task: {task.description}. Current phase: {agent_state.current_phase}")
        result = {
            "script_created": True,
            "shots_planned": 10,
            "visual_elements_planned": 15,
            "content_quality_score": 0.88
        }
        logger.info(f"Content agent task completed successfully. Result: {result}")
        return result
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with content creation results"""
        logger.info(f"Content agent updating agent state. Current phase: {agent_state.current_phase}")
        
        from models import Script, Shot
        
        # Create mock script
        shots = [
            Shot(
                shot_number=1,
                description="Opening hook shot",
                duration_seconds=5.0,
                visual_elements=["Title card", "Background animation"],
                shot_type="establishing",
                timing_start=0.0,
                timing_end=5.0
            )
        ]
        
        script = Script(
            title=f"Content about {agent_state.content_request}",
            content_type=agent_state.content_type,
            target_audience=agent_state.target_audience,
            duration_minutes=5.0,
            hook="Engaging opening hook",
            main_content="Main content here",
            call_to_action="Subscribe for more content",
            shots=shots
        )
        
        agent_state.script = script
        
        # Update current phase to advance the workflow
        if agent_state.current_phase == "generate_script":
            agent_state.current_phase = "store_script"
            logger.info("Advanced phase from generate_script to store_script")
        elif agent_state.current_phase == "shot_analysis":
            # Update shot analysis data
            agent_state.shot_breakdown = result.get("shot_breakdown", [])
            agent_state.shot_timing = result.get("shot_timing", [])
            agent_state.shot_types = result.get("shot_types", [])
            agent_state.current_phase = "parallel_generation"
            logger.info("Advanced phase from shot_analysis to parallel_generation")
        elif agent_state.current_phase == "visual_table_generation":
            agent_state.visual_table = result.get("visual_table", {})
            agent_state.current_phase = "asset_gathering"
            logger.info("Advanced phase from visual_table_generation to asset_gathering")