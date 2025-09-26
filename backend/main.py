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
    
    def create_project_from_statement(self, 
                                    content: str,
                                    project_name: str = "Demo Project",
                                    content_type: ContentType = ContentType.YOUTUBE_VIDEO,
                                    target_audience: str = "general audience",
                                    deadline: str = None,
                                    input_type: InputType = InputType.STATEMENT) -> Dict[str, Any]:
        """Create and run a project based on a simple statement from the user"""
        
        # Create user input from the statement
        user_input = UserInput(
            input_type=input_type,
            content=content,
            project_name=project_name,
            content_type=content_type,
            target_audience=target_audience,
            deadline=deadline
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
    
    # Example 1: Simple statement input
    print("=== Example 1: Simple Statement Input ===")
    result1 = system.create_project_from_statement(
        content="Create a YouTube video about the latest AI trends in 2025",
        project_name="AI Trends Video",
        content_type=ContentType.YOUTUBE_VIDEO,
        target_audience="tech enthusiasts"
    )
    
    if result1["success"]:
        print("✅ Workflow completed successfully!")
        print(f"Workflow completed: {result1['workflow_completed']}")
        if result1["messages"]:
            print(f"Last message: {result1['messages'][-1].content[:100]}...")
    else:
        print(f"❌ Workflow failed: {result1['error']}")
    
    # Example 2: Document processing
    print("\n=== Example 2: Document Processing ===")
    document_content = """
    # Research Summary: AI in Healthcare
    
    Recent advances in artificial intelligence have revolutionized healthcare...
    Key findings include improved diagnostic accuracy and patient outcomes...
    """
    result2 = system.create_project_from_statement(
        content="Please create a blog article based on this research document",
        project_name="Healthcare AI Content",
        content_type=ContentType.BLOG_ARTICLE,
        target_audience="medical professionals",
        input_type=InputType.DOCUMENT,
        # Pass the document content via context
    )
    
    # For document input, we need to handle it specially
    from langgraph_workflow import UserInput
    user_input_doc = UserInput(
        input_type=InputType.DOCUMENT,
        content="Please create a blog article based on this research document",
        project_name="Healthcare AI Content",
        document_content=document_content,
        content_type=ContentType.BLOG_ARTICLE,
        target_audience="medical professionals"
    )
    
    result2 = system.app.invoke(user_input_doc)
    
    if result2:
        print("✅ Document processing completed successfully!")
        print(f"Workflow completed: {result2.get('workflow_completed', False)}")
        if result2.get("messages"):
            print(f"Last message: {result2['messages'][-1].content[:100]}...")
    
    # Example 3: Website processing
    print("\n=== Example 3: Website Processing ===")
    user_input_website = UserInput(
        input_type=InputType.WEBSITE,
        content="Create a presentation about this research paper",
        project_name="Website Content",
        website_url="https://example.com/ai-research-2025",
        content_type=ContentType.PRESENTATION,
        target_audience="researchers"
    )
    
    result3 = system.app.invoke(user_input_website)
    
    if result3:
        print("✅ Website processing completed successfully!")
        print(f"Workflow completed: {result3.get('workflow_completed', False)}")
        if result3.get("messages"):
            print(f"Last message: {result3['messages'][-1].content[:100]}...")
    
    # Display agent capabilities
    capabilities = system.get_agent_capabilities()
    print("\n🤖 Available agents:")
    for agent_type, agent_info in capabilities.items():
        print(f"- {agent_info['name']}: {agent_info['description']}")


if __name__ == "__main__":
    main()
