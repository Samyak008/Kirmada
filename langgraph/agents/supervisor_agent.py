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

from .specialized_agent import SpecializedAgent, AgentType, Task, TaskStatus, LangGraphState

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupervisorDecision(BaseModel):
    decision_id: str
    context: str
    chosen_agent: AgentType
    reasoning: str
    expected_outcome: str
    priority: int = Field(ge=1, le=5)
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentRouter:
    """Handles routing between different agents based on current state"""
    
    def __init__(self, config_path: str = None):
        # Use default config if none provided
        if config_path:
            import os
            if os.path.exists(config_path):
                import yaml
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = {}
        else:
            self.config = {}
    
    def should_continue(self, state: LangGraphState) -> str:
        """Determine if workflow should continue and which agent to call next"""
        logger.info(f"AgentRouter checking if workflow should continue. Current phase: {state['workflow_phase']}")
        
        # Check if workflow is completed
        if state.get("workflow_completed", False):
            logger.info("Workflow is completed, ending")
            return "end"
        
        # Check for errors that need handling
        if state.get("errors"):
            logger.warning(f"Errors found in state: {len(state['errors'])} errors")
            return "error_handler"
        
        # Determine next agent based on current phase
        next_agent = self._determine_next_agent(state)
        
        if next_agent is None:
            logger.info("No next agent determined, ending workflow")
            return "end"
        
        agent_name = f"agent_{next_agent.value}"
        logger.info(f"Router directing to: {agent_name}")
        return agent_name
    
    def _determine_next_agent(self, state: LangGraphState) -> Optional[AgentType]:
        """Determine which agent should be called next based on current state"""
        logger.info(f"Determining next agent. Current phase: {state['workflow_phase']}")
        
        current_phase = state['workflow_phase']
        
        # Phase-based routing (matching backend workflow)
        phase_routing = {
            "initial": [AgentType.SUPERVISOR],
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
        
        # For now, just return the first available agent for the phase
        if available_agents:
            return available_agents[0]
        
        logger.info(f"No next agent found for phase '{current_phase}'")
        return None


class SupervisorAgent(SpecializedAgent):
    """Supervisor agent that makes decisions and coordinates the workflow"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.SUPERVISOR, config)
    
    def process(self, state: LangGraphState) -> LangGraphState:
        """Process supervisor decisions and task assignments"""
        logger.info(f"Supervisor processing state. Current phase: {state['workflow_phase']}")
        
        # Check if this is the first time the supervisor is being called
        # If current phase is initial state, analyze the user input to determine the right phase
        if state['workflow_phase'] in ["initial", "planning", "initialize"]:
            logger.info("Supervisor initializing workflow based on user input")
            
            # Analyze the content request to determine the appropriate starting phase
            content_request = state['user_request'].lower()
            
            if "website:" in content_request or state.get('content_type', '') == "presentation":
                state['workflow_phase'] = "crawl"
            elif "video:" in content_request:
                state['workflow_phase'] = "analyze_video"
            else:
                # Default to search for most content requests
                state['workflow_phase'] = "search"
        
        # Check if there are already pending tasks for the current phase
        current_phase_tasks = []
        for agent_tasks in state.get("task_results", {}).values():
            for task in agent_tasks:
                if task.get('status', 'pending') == 'pending':
                    current_phase_tasks.append(task)
        
        if current_phase_tasks:
            logger.info(f"Found {len(current_phase_tasks)} pending tasks, skipping new task creation")
            return state
        
        # Analyze current state and make decisions
        decisions = self._make_decisions(state)
        
        logger.info(f"Supervisor made {len(decisions)} decisions")
        
        # Update state with decisions
        created_tasks = 0
        for decision in decisions:
            if 'supervisor_decisions' not in state:
                state['supervisor_decisions'] = []
            state['supervisor_decisions'].append(decision.dict())
            
            # Create task if needed
            if decision.chosen_agent != AgentType.SUPERVISOR:
                task = self._create_task(decision, state)
                agent_type_str = decision.chosen_agent.value
                if agent_type_str not in state['task_results']:
                    state['task_results'][agent_type_str] = []
                state['task_results'][agent_type_str].append(task)
                created_tasks += 1
                logger.info(f"Created task for {decision.chosen_agent.value}: {task['description']}")
        
        logger.info(f"Supervisor completed. Current phase: {state['workflow_phase']}, Tasks created: {created_tasks}")
        return state
    
    def _make_decisions(self, state: LangGraphState) -> List[SupervisorDecision]:
        """Make strategic decisions about next steps"""
        logger.info(f"Making decisions for phase: {state['workflow_phase']}")
        
        decisions = []
        
        # When in initial phase, analyze the user's content request to determine best path
        if state['workflow_phase'] in ["initial", "planning", "search"]:
            # Analyze the content request to determine appropriate next steps
            content_request = state['user_request'].lower()
            
            # Determine if this is document, website, or general content request
            if "website:" in content_request:
                # This is a website processing request
                logger.info("Detected website processing request")
                state['workflow_phase'] = "crawl"
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
                state['workflow_phase'] = "analyze_video"
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
                state['workflow_phase'] = "search"
                decisions.append(SupervisorDecision(
                    decision_id=f"search_{datetime.now().timestamp()}",
                    context=f"Search for content related to: {content_request}",
                    chosen_agent=AgentType.RESEARCH,
                    reasoning="Research agent needed to search for relevant content based on user request",
                    expected_outcome="Relevant search results and URLs identified for the topic",
                    priority=1
                ))
        
        elif state['workflow_phase'] == "analyze_video":
            logger.info("Creating decision for video analysis phase")
            decisions.append(SupervisorDecision(
                decision_id=f"video_analysis_{datetime.now().timestamp()}",
                context="Analyze video content for summary and key points",
                chosen_agent=AgentType.RESEARCH,
                reasoning="Research agent needed to extract key points from video content",
                expected_outcome="Video content analyzed with key points and summary extracted",
                priority=1
            ))
        
        elif state['workflow_phase'] == "crawl":
            logger.info("Creating decision for crawl phase")
            decisions.append(SupervisorDecision(
                decision_id=f"crawl_{datetime.now().timestamp()}",
                context="Crawl phase needs to extract article content",
                chosen_agent=AgentType.RESEARCH,
                reasoning="Research agent needed to extract and structure content",
                expected_outcome="Article content extracted and structured",
                priority=1
            ))
        
        elif state['workflow_phase'] == "store_article":
            logger.info("Creating decision for store_article phase")
            decisions.append(SupervisorDecision(
                decision_id=f"store_article_{datetime.now().timestamp()}",
                context="Store article phase needs database storage",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to store article data",
                expected_outcome="Article stored in database with ID",
                priority=1
            ))
        
        elif state['workflow_phase'] == "generate_script":
            logger.info("Creating decision for generate_script phase")
            decisions.append(SupervisorDecision(
                decision_id=f"generate_script_{datetime.now().timestamp()}",
                context="Script generation phase needs content creation",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to create engaging script",
                expected_outcome="Script content and hook generated",
                priority=1
            ))
        
        elif state['workflow_phase'] == "store_script":
            logger.info("Creating decision for store_script phase")
            decisions.append(SupervisorDecision(
                decision_id=f"store_script_{datetime.now().timestamp()}",
                context="Store script phase needs database storage",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to store script data",
                expected_outcome="Script stored in database with ID",
                priority=1
            ))
        
        elif state['workflow_phase'] == "shot_analysis":
            logger.info("Creating decision for shot_analysis phase")
            decisions.append(SupervisorDecision(
                decision_id=f"shot_analysis_{datetime.now().timestamp()}",
                context="Shot analysis phase needs shot breakdown",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to analyze shots and timing",
                expected_outcome="Shot breakdown, timing, and types analyzed",
                priority=1
            ))
        
        elif state['workflow_phase'] == "parallel_generation":
            logger.info("Creating decision for parallel_generation phase")
            decisions.append(SupervisorDecision(
                decision_id=f"parallel_generation_{datetime.now().timestamp()}",
                context="Parallel generation phase needs asset creation",
                chosen_agent=AgentType.ASSET_GENERATION,
                reasoning="Asset generation agent needed for parallel asset creation",
                expected_outcome="Prompts, images, voice, and b-roll generated",
                priority=1
            ))
        
        elif state['workflow_phase'] == "visual_table_generation":
            logger.info("Creating decision for visual_table_generation phase")
            decisions.append(SupervisorDecision(
                decision_id=f"visual_table_{datetime.now().timestamp()}",
                context="Visual table generation phase needs organization",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to generate visual table",
                expected_outcome="Visual table created for organization",
                priority=1
            ))
        
        elif state['workflow_phase'] == "asset_gathering":
            logger.info("Creating decision for asset_gathering phase")
            decisions.append(SupervisorDecision(
                decision_id=f"asset_gathering_{datetime.now().timestamp()}",
                context="Asset gathering phase needs organization",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to gather and organize assets",
                expected_outcome="Assets gathered and organized",
                priority=1
            ))
        
        elif state['workflow_phase'] == "notion_integration":
            logger.info("Creating decision for notion_integration phase")
            decisions.append(SupervisorDecision(
                decision_id=f"notion_integration_{datetime.now().timestamp()}",
                context="Notion integration phase needs project setup",
                chosen_agent=AgentType.PROJECT_MANAGEMENT,
                reasoning="Project management agent needed for Notion setup",
                expected_outcome="Notion project created and integrated",
                priority=1
            ))
        
        elif state['workflow_phase'] == "finalize":
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
    
    def _create_task(self, decision: SupervisorDecision, state: LangGraphState) -> Dict[str, Any]:
        """Create a task based on supervisor decision"""
        task = {
            "task_id": f"task_{decision.decision_id}",
            "agent_type": decision.chosen_agent.value,
            "description": decision.expected_outcome,
            "status": "pending",
            "priority": decision.priority,
            "assigned_at": datetime.now().isoformat()
        }
        logger.info(f"Created new task: {task['task_id']} for {task['agent_type']}")
        return task