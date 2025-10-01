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
import os
TRACE_VERBOSE = os.getenv("AGENT_TRACE_VERBOSE", "false").lower() in {"1", "true", "yes"}


class SupervisorAgent:
    """Supervisor agent that makes decisions and coordinates the workflow"""
    
    def __init__(self, config_path: str = "agent_prompts.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.system_prompt = self.config["agents"]["supervisor"]["system_prompt"]
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Process supervisor decisions and task assignments"""
        logger.info(f"Supervisor processing state. Current phase: {state['agent_state'].current_phase}")
        
        # Use the system prompt for decision making, but don't add to messages unless needed for debugging
        agent_state = state["agent_state"]
        
        # Check if this is the first time the supervisor is being called
        # If current phase is initial state, analyze the user input to determine the right phase
        if agent_state.current_phase in ["planning", "initialize"]:
            logger.info("Supervisor initializing workflow based on user input")
            
            # Analyze the content request to determine the appropriate starting phase
            content_request = agent_state.content_request.lower()
            
            if "website:" in content_request or agent_state.content_type == "presentation":
                agent_state.current_phase = "crawl"
            elif "video:" in content_request:
                agent_state.current_phase = "analyze_video"
            else:
                # Default to search for most content requests
                agent_state.current_phase = "search"
        
        # Check if there are already pending tasks for the current phase
        current_phase_tasks = [task for task in agent_state.tasks 
                              if task.status == TaskStatus.PENDING]
        
        if current_phase_tasks:
            logger.info(f"Found {len(current_phase_tasks)} pending tasks, skipping new task creation")
            # Optionally add a message to reduce trace noise
            if TRACE_VERBOSE:
                supervisor_message = AIMessage(
                    content=f"Supervisor continuing with {len(current_phase_tasks)} existing pending tasks"
                )
                state["messages"].append(supervisor_message)
            return state
        
        # Analyze current state and make decisions
        decisions = self._make_decisions(agent_state)
        
        logger.info(f"Supervisor made {len(decisions)} decisions")
        
        # Update agent state with decisions
        created_tasks = 0
        for decision in decisions:
            agent_state.supervisor_decisions.append(decision)
            
            # Create task if needed
            if decision.chosen_agent != AgentType.SUPERVISOR:
                task = self._create_task(decision, agent_state)
                agent_state.tasks.append(task)
                created_tasks += 1
                logger.info(f"Created task for {decision.chosen_agent.value}: {task.description}")
        
        # Optionally add summary message (to reduce Studio traces)
        if TRACE_VERBOSE:
            supervisor_message = AIMessage(
                content=f"Supervisor decisions made: {created_tasks} tasks assigned for phase {agent_state.current_phase}. Content request analyzed: {agent_state.content_request[:100]}..."
            )
            state["messages"].append(supervisor_message)
        
        logger.info(f"Supervisor completed. Current phase: {agent_state.current_phase}, Tasks created: {created_tasks}")
        return state
    
    def _make_decisions(self, agent_state: AgentState) -> List[SupervisorDecision]:
        """Make strategic decisions about next steps"""
        logger.info(f"Making decisions for phase: {agent_state.current_phase}")
        
        decisions = []
        
        # When in initial phase, analyze the user's content request to determine best path
        if agent_state.current_phase == "planning" or agent_state.current_phase == "search":
            # Analyze the content request to determine appropriate next steps
            content_request = agent_state.content_request.lower()
            
            # Determine if this is document, website, or general content request
            if "website:" in content_request:
                # This is a website processing request
                logger.info("Detected website processing request")
                agent_state.current_phase = "crawl"
                decisions.append(SupervisorDecision(
                    decision_id=f"crawl_{datetime.now().timestamp()}",
                    context="Process website content: " + content_request,
                    chosen_agent=AgentType.RESEARCH,
                    reasoning="Research agent needed to crawl and extract website content",
                    expected_outcome="Website content extracted and structured",
                    priority=1
                ))
            elif "video:" in content_request:
                # This is a video processing request
                logger.info("Detected video processing request")
                agent_state.current_phase = "analyze_video"
                decisions.append(SupervisorDecision(
                    decision_id=f"analyze_video_{datetime.now().timestamp()}",
                    context="Analyze video content: " + content_request,
                    chosen_agent=AgentType.RESEARCH,
                    reasoning="Research agent needed to analyze video content",
                    expected_outcome="Video content analyzed and key points extracted",
                    priority=1
                ))
            else:
                # This is a general content request - start with search
                logger.info("Creating decision for search phase")
                agent_state.current_phase = "search"
                decisions.append(SupervisorDecision(
                    decision_id=f"search_{datetime.now().timestamp()}",
                    context=f"Search for content related to: {content_request}",
                    chosen_agent=AgentType.RESEARCH,
                    reasoning="Research agent needed to search for relevant content based on user request",
                    expected_outcome="Relevant search results and URLs identified for the topic",
                    priority=1
                ))
        
        elif agent_state.current_phase == "analyze_video":
            logger.info("Creating decision for video analysis phase")
            decisions.append(SupervisorDecision(
                decision_id=f"video_analysis_{datetime.now().timestamp()}",
                context="Analyze video content for summary and key points",
                chosen_agent=AgentType.RESEARCH,
                reasoning="Research agent needed to extract key points from video content",
                expected_outcome="Video content analyzed with key points and summary extracted",
                priority=1
            ))
        
        elif agent_state.current_phase == "crawl":
            logger.info("Creating decision for crawl phase")
            decisions.append(SupervisorDecision(
                decision_id=f"crawl_{datetime.now().timestamp()}",
                context="Crawl phase needs to extract article content",
                chosen_agent=AgentType.RESEARCH,
                reasoning="Research agent needed to extract and structure content",
                expected_outcome="Article content extracted and structured",
                priority=1
            ))
        
        elif agent_state.current_phase == "store_article":
            logger.info("Creating decision for store_article phase")
            decisions.append(SupervisorDecision(
                decision_id=f"store_article_{datetime.now().timestamp()}",
                context="Store article phase needs database storage",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to store article data",
                expected_outcome="Article stored in database with ID",
                priority=1
            ))
        
        elif agent_state.current_phase == "generate_script":
            logger.info("Creating decision for generate_script phase")
            decisions.append(SupervisorDecision(
                decision_id=f"generate_script_{datetime.now().timestamp()}",
                context="Script generation phase needs content creation",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to create engaging script",
                expected_outcome="Script content and hook generated",
                priority=1
            ))
        
        elif agent_state.current_phase == "store_script":
            logger.info("Creating decision for store_script phase")
            decisions.append(SupervisorDecision(
                decision_id=f"store_script_{datetime.now().timestamp()}",
                context="Store script phase needs database storage",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to store script data",
                expected_outcome="Script stored in database with ID",
                priority=1
            ))
        
        elif agent_state.current_phase == "shot_analysis":
            logger.info("Creating decision for shot_analysis phase")
            decisions.append(SupervisorDecision(
                decision_id=f"shot_analysis_{datetime.now().timestamp()}",
                context="Shot analysis phase needs shot breakdown",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to analyze shots and timing",
                expected_outcome="Shot breakdown, timing, and types analyzed",
                priority=1
            ))
        
        elif agent_state.current_phase == "parallel_generation":
            logger.info("Creating decision for parallel_generation phase")
            decisions.append(SupervisorDecision(
                decision_id=f"parallel_generation_{datetime.now().timestamp()}",
                context="Parallel generation phase needs asset creation",
                chosen_agent=AgentType.ASSET_GENERATION,
                reasoning="Asset generation agent needed for parallel asset creation",
                expected_outcome="Prompts, images, voice, and b-roll generated",
                priority=1
            ))
        
        elif agent_state.current_phase == "visual_table_generation":
            logger.info("Creating decision for visual_table_generation phase")
            decisions.append(SupervisorDecision(
                decision_id=f"visual_table_{datetime.now().timestamp()}",
                context="Visual table generation phase needs organization",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to generate visual table",
                expected_outcome="Visual table created for organization",
                priority=1
            ))
        
        elif agent_state.current_phase == "asset_gathering":
            logger.info("Creating decision for asset_gathering phase")
            decisions.append(SupervisorDecision(
                decision_id=f"asset_gathering_{datetime.now().timestamp()}",
                context="Asset gathering phase needs organization",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to gather and organize assets",
                expected_outcome="Assets gathered and organized",
                priority=1
            ))
        
        elif agent_state.current_phase == "notion_integration":
            logger.info("Creating decision for notion_integration phase")
            decisions.append(SupervisorDecision(
                decision_id=f"notion_integration_{datetime.now().timestamp()}",
                context="Notion integration phase needs project setup",
                chosen_agent=AgentType.PROJECT_MANAGEMENT,
                reasoning="Project management agent needed for Notion setup",
                expected_outcome="Notion project created and integrated",
                priority=1
            ))
        
        elif agent_state.current_phase == "finalize":
            logger.info("Creating decision for finalize phase")
            decisions.append(SupervisorDecision(
                decision_id=f"finalize_{datetime.now().timestamp()}",
                context="Finalization phase needs project completion",
                chosen_agent=AgentType.SUPERVISOR,
                reasoning="Supervisor needed for final review and completion",
                expected_outcome="Project finalized and ready for handoff",
                priority=1
            ))
        
        return decisions
    
    def _create_task(self, decision: SupervisorDecision, agent_state: AgentState) -> Task:
        """Create a task based on supervisor decision"""
        task = Task(
            task_id=f"task_{decision.decision_id}",
            agent_type=decision.chosen_agent,
            description=decision.expected_outcome,
            status=TaskStatus.PENDING,
            priority=decision.priority,
            assigned_at=datetime.now()
        )
        logger.info(f"Created new task: {task.task_id} for {task.agent_type.value}")
        return task