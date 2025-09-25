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


class SpecializedAgent:
    """Base class for specialized agents (Research, Content, Asset Generation, etc.)"""
    
    def __init__(self, agent_type: AgentType, config: Dict[str, Any]):
        self.agent_type = agent_type
        self.config = config
        self.system_prompt = config["agents"][agent_type.value]["system_prompt"]
        self.tools = get_tools_for_agent(agent_type.value)
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Process agent-specific tasks"""
        logger.info(f"{self.agent_type.value} agent processing. Current phase: {state['agent_state'].current_phase}")
        
        # Don't add system prompt as a message to avoid cluttering the logs
        # The system prompt is used internally but not added to state messages
        agent_state = state["agent_state"]

        
        # Get pending tasks for this agent
        pending_tasks = [task for task in agent_state.tasks 
                        if task.agent_type == self.agent_type and task.status == TaskStatus.PENDING]
        
        logger.info(f"{self.agent_type.value} agent found {len(pending_tasks)} pending tasks")
        
        if not pending_tasks:
            logger.info(f"{self.agent_type.value} agent has no pending tasks, returning state")
            return state
        
        # Process the highest priority task
        task = max(pending_tasks, key=lambda t: t.priority)
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        logger.info(f"{self.agent_type.value} agent starting task: {task.description}")
        
        try:
            # Execute agent-specific logic
            result = self._execute_task(task, agent_state)
            logger.info(f"{self.agent_type.value} agent completed task execution. Success: {result.get('success', True) if isinstance(result, dict) else True}")
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            if isinstance(result, dict):
                task.result = result
            
            # Update agent state with results
            self._update_agent_state(agent_state, result if isinstance(result, dict) else {})
            
            # Add completion message
            completion_message = AIMessage(
                content=f"{self.agent_type.value} agent completed task: {task.description}"
            )
            state["messages"].append(completion_message)
            
            logger.info(f"{self.agent_type.value} agent completed task successfully. New phase: {agent_state.current_phase}")
            
        except Exception as e:
            logger.error(f"{self.agent_type.value} agent failed task: {str(e)}")
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            
            error_message = AIMessage(
                content=f"{self.agent_type.value} agent failed task: {str(e)}"
            )
            state["messages"].append(error_message)
            state["errors"].append({
                "agent": self.agent_type.value,
                "task": task.task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute agent-specific task logic - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_task")
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with task results - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _update_agent_state")