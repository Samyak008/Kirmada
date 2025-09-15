from __future__ import annotations  # ensure annotations aren't evaluated at import time

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
import yaml
import json
from datetime import datetime
from models import (
    AgentState, AgentType, TaskStatus, WorkflowPhase, 
    AgentMessage, SupervisorDecision, QualityCheck, Task  # add Task here
)
from tools import get_tools_for_agent, validate_tool_input, ToolResult


class WorkflowState(TypedDict):
    """LangGraph state for the agentic workflow"""
    agent_state: AgentState
    messages: List[BaseMessage]
    current_agent: Optional[AgentType]
    next_agent: Optional[AgentType]
    task_results: Dict[str, Any]
    errors: List[Dict[str, Any]]
    workflow_completed: bool


class AgentRouter:
    """Handles routing between different agents based on current state"""
    
    def __init__(self, config_path: str = "agent_prompts.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def should_continue(self, state: WorkflowState) -> str:
        """Determine if workflow should continue and which agent to call next"""
        agent_state = state["agent_state"]
        
        # Check if workflow is completed
        if agent_state.workflow_completed:
            return "end"
        
        # Check for errors that need handling
        if state["errors"]:
            return "error_handler"
        
        # Determine next agent based on current phase and completed tasks
        next_agent = self._determine_next_agent(agent_state)
        
        if next_agent is None:
            return "end"
        
        return f"agent_{next_agent.value}"
    
    def _determine_next_agent(self, agent_state: AgentState) -> Optional[AgentType]:
        """Determine which agent should be called next based on current state"""
        current_phase = agent_state.current_phase
        
        # Phase-based routing (matching original workflow)
        phase_routing = {
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
        
        # Check which agents have pending tasks
        for agent in available_agents:
            agent_tasks = [task for task in agent_state.tasks 
                          if task.agent_type == agent and task.status == TaskStatus.PENDING]
            if agent_tasks:
                return agent
        
        # If no pending tasks, check if phase is complete and move to next phase
        if self._is_phase_complete(agent_state, current_phase):
            next_phase = self._get_next_phase(current_phase)
            if next_phase:
                agent_state.current_phase = next_phase
                return self._determine_next_agent(agent_state)
        
        return None
    
    def _is_phase_complete(self, agent_state: AgentState, phase: str) -> bool:
        """Check if current phase is complete"""
        phase_agents = {
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
        
        required_agents = phase_agents.get(phase, [])
        for agent in required_agents:
            agent_tasks = [task for task in agent_state.tasks 
                          if task.agent_type == agent and task.status != TaskStatus.COMPLETED]
            if agent_tasks:
                return False
        
        return True
    
    def _get_next_phase(self, current_phase: str) -> Optional[str]:
        """Get the next phase in the workflow"""
        phase_sequence = [
            "search", "crawl", "store_article", "generate_script", 
            "store_script", "shot_analysis", "parallel_generation",
            "visual_table_generation", "asset_gathering", 
            "notion_integration", "finalize"
        ]
        
        try:
            current_index = phase_sequence.index(current_phase)
            if current_index < len(phase_sequence) - 1:
                return phase_sequence[current_index + 1]
        except ValueError:
            pass
        
        return None


class SupervisorAgent:
    """Supervisor agent that makes decisions and coordinates the workflow"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_prompt = config["agents"]["supervisor"]["system_prompt"]
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Process supervisor decisions and task assignments"""
        agent_state = state["agent_state"]
        
        # Analyze current state and make decisions
        decisions = self._make_decisions(agent_state)
        
        # Update agent state with decisions
        for decision in decisions:
            agent_state.supervisor_decisions.append(decision)
            
            # Create task if needed
            if decision.chosen_agent != AgentType.SUPERVISOR:
                task = self._create_task(decision, agent_state)
                agent_state.tasks.append(task)
        
        # Update messages
        supervisor_message = AIMessage(
            content=f"Supervisor decisions made: {len(decisions)} tasks assigned"
        )
        state["messages"].append(supervisor_message)
        
        return state
    
    def _make_decisions(self, agent_state: AgentState) -> List[SupervisorDecision]:
        """Make strategic decisions about next steps"""
        decisions = []
        
        # Analyze current phase and determine what needs to be done
        if agent_state.current_phase == "search":
            decisions.append(SupervisorDecision(
                decision_id=f"search_{datetime.now().timestamp()}",
                context="Search phase needs to find trending articles",
                chosen_agent=AgentType.RESEARCH,
                reasoning="Research agent needed to search for relevant content",
                expected_outcome="Search results and URLs identified",
                priority=1
            ))
        
        elif agent_state.current_phase == "crawl":
            decisions.append(SupervisorDecision(
                decision_id=f"crawl_{datetime.now().timestamp()}",
                context="Crawl phase needs to extract article content",
                chosen_agent=AgentType.RESEARCH,
                reasoning="Research agent needed to extract and structure content",
                expected_outcome="Article content extracted and structured",
                priority=1
            ))
        
        elif agent_state.current_phase == "store_article":
            decisions.append(SupervisorDecision(
                decision_id=f"store_article_{datetime.now().timestamp()}",
                context="Store article phase needs database storage",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to store article data",
                expected_outcome="Article stored in database with ID",
                priority=1
            ))
        
        elif agent_state.current_phase == "generate_script":
            decisions.append(SupervisorDecision(
                decision_id=f"generate_script_{datetime.now().timestamp()}",
                context="Script generation phase needs content creation",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to create engaging script",
                expected_outcome="Script content and hook generated",
                priority=1
            ))
        
        elif agent_state.current_phase == "store_script":
            decisions.append(SupervisorDecision(
                decision_id=f"store_script_{datetime.now().timestamp()}",
                context="Store script phase needs database storage",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to store script data",
                expected_outcome="Script stored in database with ID",
                priority=1
            ))
        
        elif agent_state.current_phase == "shot_analysis":
            decisions.append(SupervisorDecision(
                decision_id=f"shot_analysis_{datetime.now().timestamp()}",
                context="Shot analysis phase needs shot breakdown",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to analyze shots and timing",
                expected_outcome="Shot breakdown, timing, and types analyzed",
                priority=1
            ))
        
        elif agent_state.current_phase == "parallel_generation":
            decisions.append(SupervisorDecision(
                decision_id=f"parallel_generation_{datetime.now().timestamp()}",
                context="Parallel generation phase needs asset creation",
                chosen_agent=AgentType.ASSET_GENERATION,
                reasoning="Asset generation agent needed for parallel asset creation",
                expected_outcome="Prompts, images, voice, and b-roll generated",
                priority=1
            ))
        
        elif agent_state.current_phase == "visual_table_generation":
            decisions.append(SupervisorDecision(
                decision_id=f"visual_table_{datetime.now().timestamp()}",
                context="Visual table generation phase needs organization",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to generate visual table",
                expected_outcome="Visual table created for organization",
                priority=1
            ))
        
        elif agent_state.current_phase == "asset_gathering":
            decisions.append(SupervisorDecision(
                decision_id=f"asset_gathering_{datetime.now().timestamp()}",
                context="Asset gathering phase needs organization",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to gather and organize assets",
                expected_outcome="Assets gathered and organized",
                priority=1
            ))
        
        elif agent_state.current_phase == "notion_integration":
            decisions.append(SupervisorDecision(
                decision_id=f"notion_integration_{datetime.now().timestamp()}",
                context="Notion integration phase needs project setup",
                chosen_agent=AgentType.PROJECT_MANAGEMENT,
                reasoning="Project management agent needed for Notion setup",
                expected_outcome="Notion project created and integrated",
                priority=1
            ))
        
        elif agent_state.current_phase == "finalize":
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
        return Task(
            task_id=f"task_{decision.decision_id}",
            agent_type=decision.chosen_agent,
            description=decision.expected_outcome,
            status=TaskStatus.PENDING,
            priority=decision.priority,
            assigned_at=datetime.now()
        )


class SpecializedAgent:
    """Base class for specialized agents (Research, Content, Asset Generation, etc.)"""
    
    def __init__(self, agent_type: AgentType, config: Dict[str, Any]):
        self.agent_type = agent_type
        self.config = config
        self.system_prompt = config["agents"][agent_type.value]["system_prompt"]
        self.tools = get_tools_for_agent(agent_type.value)
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Process agent-specific tasks"""
        agent_state = state["agent_state"]
        
        # Get pending tasks for this agent
        pending_tasks = [task for task in agent_state.tasks 
                        if task.agent_type == self.agent_type and task.status == TaskStatus.PENDING]
        
        if not pending_tasks:
            return state
        
        # Process the highest priority task
        task = max(pending_tasks, key=lambda t: t.priority)
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Execute agent-specific logic
            result = self._execute_task(task, agent_state)
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Update agent state with results
            self._update_agent_state(agent_state, result)
            
            # Add completion message
            completion_message = AIMessage(
                content=f"{self.agent_type.value} agent completed task: {task.description}"
            )
            state["messages"].append(completion_message)
            
        except Exception as e:
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


class ResearchAgent(SpecializedAgent):
    """Research agent implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.RESEARCH, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute research tasks"""
        # This is where you would implement actual research logic
        # For now, return mock data structure
        return {
            "research_completed": True,
            "sources_found": 5,
            "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
            "research_quality_score": 0.85
        }
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with research results"""
        # Create mock research data
        from models import ResearchData, Source
        
        sources = [
            Source(
                url="https://example.com/article1",
                title="Sample Article 1",
                content="Sample content from article 1",
                relevance_score=0.9,
                tags=["tech", "ai"]
            )
        ]
        
        research_data = ResearchData(
            topic=agent_state.content_request,
            sources=sources,
            key_findings=result.get("key_findings", []),
            research_notes="Research completed successfully"
        )
        
        agent_state.research_data = research_data
        
        # Update current step based on phase
        if agent_state.current_phase == "search":
            agent_state.current_step = "crawl"
        elif agent_state.current_phase == "crawl":
            agent_state.current_step = "store_article"


class ContentAgent(SpecializedAgent):
    """Content agent implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.CONTENT, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute content creation tasks"""
        return {
            "script_created": True,
            "shots_planned": 10,
            "visual_elements_planned": 15,
            "content_quality_score": 0.88
        }
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with content creation results"""
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
        
        # Update current step based on phase
        if agent_state.current_phase == "generate_script":
            agent_state.current_step = "store_script"
        elif agent_state.current_phase == "shot_analysis":
            # Update shot analysis data
            agent_state.shot_breakdown = result.get("shot_breakdown", [])
            agent_state.shot_timing = result.get("shot_timing", [])
            agent_state.shot_types = result.get("shot_types", [])
            agent_state.current_step = "parallel_generation"
        elif agent_state.current_phase == "visual_table_generation":
            agent_state.visual_table = result.get("visual_table", {})
            agent_state.current_step = "asset_gathering"


class AssetGenerationAgent(SpecializedAgent):
    """Asset generation agent implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.ASSET_GENERATION, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute asset generation tasks"""
        return {
            "assets_created": 8,
            "images_generated": 5,
            "audio_recordings": 2,
            "broll_found": 3,
            "asset_quality_score": 0.92
        }
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with asset generation results"""
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
        
        # Update current step based on phase
        if agent_state.current_phase == "parallel_generation":
            # Update parallel generation results
            agent_state.prompts_generated = result.get("prompts_generated", [])
            agent_state.images_generated = result.get("images_generated", [])
            agent_state.voice_files = result.get("voice_files", [])
            agent_state.broll_assets = result.get("broll_assets", {})
            agent_state.current_step = "visual_table_generation"
        elif agent_state.current_phase == "asset_gathering":
            agent_state.project_folder_path = result.get("project_folder_path", "")
            agent_state.asset_organization_result = result.get("asset_organization_result", "")
            agent_state.current_step = "notion_integration"


class StorageAgent(SpecializedAgent):
    """Storage agent implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.STORAGE, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute storage tasks"""
        return {
            "data_stored": True,
            "folders_created": 3,
            "files_uploaded": 12,
            "metadata_tagged": True
        }
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with storage results"""
        from models import ProjectFolder, DatabaseRecord
        
        # Create mock storage records
        folder = ProjectFolder(
            folder_id="project_folder_1",
            folder_name=agent_state.project_name,
            google_drive_path="/Content Production/Projects/",
            files=["file1", "file2"]
        )
        
        agent_state.project_folders.append(folder)
        
        # Update current step based on phase
        if agent_state.current_phase == "store_article":
            agent_state.current_step = "generate_script"
        elif agent_state.current_phase == "store_script":
            agent_state.current_step = "shot_analysis"


class ProjectManagementAgent(SpecializedAgent):
    """Project management agent implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.PROJECT_MANAGEMENT, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute project management tasks"""
        return {
            "notion_workspace_created": True,
            "project_tracking_setup": True,
            "milestones_defined": 6,
            "team_coordination_complete": True
        }
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with project management results"""
        from models import NotionPage
        
        # Create mock Notion page
        page = NotionPage(
            page_id="notion_page_1",
            title=agent_state.project_name,
            content={"status": "active", "progress": 25}
        )
        
        agent_state.notion_pages.append(page)
        
        # Update current step based on phase
        if agent_state.current_phase == "notion_integration":
            agent_state.notion_project_id = result.get("notion_project_id", "")
            agent_state.notion_status = result.get("notion_status", "active")
            agent_state.current_step = "finalize"


# Simple no-op router node so conditional routing has a source node
def router_node(state: WorkflowState) -> WorkflowState:
    return state

def create_workflow_graph(config_path: str = "agent_prompts.yaml") -> StateGraph:
    """Create the LangGraph workflow"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize agents
    supervisor = SupervisorAgent(config)
    research_agent = ResearchAgent(config)
    content_agent = ContentAgent(config)
    asset_agent = AssetGenerationAgent(config)
    storage_agent = StorageAgent(config)
    project_agent = ProjectManagementAgent(config)
    
    # Initialize router
    router = AgentRouter(config_path)
    
    # Create the graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("supervisor", supervisor.process)
    workflow.add_node("agent_research", research_agent.process)
    workflow.add_node("agent_content", content_agent.process)
    workflow.add_node("agent_asset_generation", asset_agent.process)
    workflow.add_node("agent_storage", storage_agent.process)
    workflow.add_node("agent_project_management", project_agent.process)
    workflow.add_node("error_handler", lambda state: state)  # Placeholder for error handling
    workflow.add_node("router", router_node)  # NEW: define router node

    # Add edges
    workflow.add_edge("supervisor", "router")
    workflow.add_edge("agent_research", "router")
    workflow.add_edge("agent_content", "router")
    workflow.add_edge("agent_asset_generation", "router")
    workflow.add_edge("agent_storage", "router")
    workflow.add_edge("agent_project_management", "router")
    workflow.add_edge("error_handler", "router")

    # Add conditional routing
    workflow.add_conditional_edges(
        "router",
        router.should_continue,
        {
            "agent_research": "agent_research",
            "agent_content": "agent_content",
            "agent_asset_generation": "agent_asset_generation",
            "agent_storage": "agent_storage",
            "agent_project_management": "agent_project_management",
            "agent_supervisor": "supervisor",  # NEW: matches f"agent_{AgentType.SUPERVISOR.value}"
            "error_handler": "error_handler",
            "end": END
        }
    )

    # Set entry point
    workflow.set_entry_point("supervisor")
    return workflow


# Example usage
if __name__ == "__main__":
    # Create and compile the workflow
    graph = create_workflow_graph()
    app = graph.compile()
    
    # Example initial state
    initial_state = WorkflowState(
        agent_state=AgentState(
            project_id="proj_001",
            project_name="Tech Content Production",
            content_request="Create a video about AI trends in 2024",
            content_type="youtube_video",
            target_audience="tech enthusiasts"
        ),
        messages=[],
        current_agent=None,
        next_agent=None,
        task_results={},
        errors=[],
        workflow_completed=False
    )
    
    # Run the workflow
    result = app.invoke(initial_state)
    print("Workflow completed!")
