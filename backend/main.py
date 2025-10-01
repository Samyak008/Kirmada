"""
Main entry point for the Agentic Content Production System
"""

import asyncio
import yaml
from typing import Dict, Any
from langgraph_workflow import create_workflow_graph, UserInput, InputType
from models import ContentType
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
    
    def create_project_from_prompt(self,
                                   prompt: str,
                                   heading: str = "Demo Project") -> Dict[str, Any]:
        """Create and run a project based on a single prompt from the user"""
        
        # Create user input from the statement
        user_input = UserInput(
            input_type=InputType.PROMPT,
            prompt=prompt,
            heading=heading,
        )
        
        # Run the workflow directly with the user input
        try:
            result = self.app.invoke(user_input)
            return {
                "success": True,
                "final_state": result,
                "workflow_completed": result.get("workflow_completed", False),
                "messages": result.get("messages", [])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get information about available agents and their capabilities"""
        return self.config["agents"]


def main():
    """Example usage of the system"""
    
    # Initialize the system
    system = AgenticContentProductionSystem()
    
    # Example 1: Simple prompt input
    print("=== Example 1: Simple Prompt Input ===")
    result1 = system.create_project_from_prompt(
        prompt="Create a YouTube video about the latest AI trends in 2025",
        heading="AI Trends Video",
    )
    
    if result1["success"]:
        print("✅ Workflow completed successfully!")
        print(f"Workflow completed: {result1['workflow_completed']}")
        if result1["messages"]:
            print(f"Last message: {result1['messages'][-1].content[:100]}...")
    else:
        print(f"❌ Workflow failed: {result1['error']}")
    
    # Example 2: (Removed) Document/website/video inputs are now tool-driven; see tools.
    
    # Example 3: See README for advanced, tool-based flows.
    
    # Display agent capabilities
    capabilities = system.get_agent_capabilities()
    print("\n🤖 Available agents:")
    for agent_type, agent_info in capabilities.items():
        print(f"- {agent_info['name']}: {agent_info['description']}")


if __name__ == "__main__":
    main()
