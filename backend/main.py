"""
Main entry point for the Agentic Content Production System
"""

import asyncio
import yaml
from typing import Dict, Any
from langgraph_workflow import create_workflow_graph
from models import AgentState, ContentType, WorkflowState
from datetime import datetime


class AgenticContentProductionSystem:
    """Main system class for orchestrating the agentic content production workflow"""
    
    def __init__(self, config_path: str = "agent_prompts.yaml"):
        """Initialize the system with configuration"""
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create the workflow graph
        self.workflow_graph = create_workflow_graph(config_path)
        self.app = self.workflow_graph.compile()
    
    def create_project(self, 
                      project_name: str,
                      content_request: str,
                      content_type: ContentType,
                      target_audience: str,
                      deadline: str = None) -> str:
        """Create a new content production project"""
        
        # Create initial agent state
        agent_state = AgentState(
            project_id=f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            project_name=project_name,
            content_request=content_request,
            content_type=content_type,
            target_audience=target_audience,
            deadline=datetime.fromisoformat(deadline) if deadline else None,
            current_phase="search",
            current_step="search"
        )
        
        # Create initial workflow state
        initial_state = WorkflowState(
            agent_state=agent_state,
            messages=[],
            current_agent=None,
            next_agent=None,
            task_results={},
            errors=[],
            workflow_completed=False
        )
        
        return agent_state.project_id
    
    def run_workflow(self, project_id: str) -> Dict[str, Any]:
        """Run the complete workflow for a project"""
        
        # Create initial state (in a real implementation, you'd load this from storage)
        initial_state = WorkflowState(
            agent_state=AgentState(
                project_id=project_id,
                project_name="Sample Project",
                content_request="Create content about AI trends",
                content_type=ContentType.YOUTUBE_VIDEO,
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
        try:
            result = self.app.invoke(initial_state)
            return {
                "success": True,
                "project_id": project_id,
                "final_state": result,
                "workflow_completed": result["workflow_completed"]
            }
        except Exception as e:
            return {
                "success": False,
                "project_id": project_id,
                "error": str(e)
            }
    
    def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get the current status of a project"""
        # In a real implementation, you'd load this from storage
        return {
            "project_id": project_id,
            "status": "active",
            "current_phase": "planning",
            "tasks_completed": 0,
            "total_tasks": 10
        }
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get information about available agents and their capabilities"""
        return self.config["agents"]


def main():
    """Example usage of the system"""
    
    # Initialize the system
    system = AgenticContentProductionSystem()
    
    # Create a new project
    project_id = system.create_project(
        project_name="AI Trends 2024 Video",
        content_request="Create an engaging YouTube video about the top AI trends in 2024",
        content_type=ContentType.YOUTUBE_VIDEO,
        target_audience="tech enthusiasts and AI professionals",
        deadline="2024-12-31T23:59:59"
    )
    
    print(f"Created project: {project_id}")
    
    # Run the workflow
    print("Starting workflow...")
    result = system.run_workflow(project_id)
    
    if result["success"]:
        print("Workflow completed successfully!")
        print(f"Project ID: {result['project_id']}")
        print(f"Workflow completed: {result['workflow_completed']}")
    else:
        print(f"Workflow failed: {result['error']}")
    
    # Get project status
    status = system.get_project_status(project_id)
    print(f"Project status: {status}")
    
    # Display agent capabilities
    capabilities = system.get_agent_capabilities()
    print("\nAvailable agents:")
    for agent_type, agent_info in capabilities.items():
        print(f"- {agent_info['name']}: {agent_info['description']}")


if __name__ == "__main__":
    main()
