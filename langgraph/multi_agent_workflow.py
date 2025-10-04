"""
New LangGraph workflow with multi-agent architecture similar to backend
"""
import asyncio
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import yaml
import json
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from enum import Enum

from agents.supervisor_agent import SupervisorAgent, AgentRouter
from agents.research_agent import ResearchAgent
from agents.content_agent import ContentAgent
from agents.asset_generation_agent import AssetGenerationAgent
from agents.storage_agent import StorageAgent
from agents.project_management_agent import ProjectManagementAgent
from agents.specialized_agent import AgentType, LangGraphState

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_multi_agent_workflow():
    """Create the multi-agent workflow graph"""
    logger.info("Creating multi-agent workflow graph")
    
    # Initialize agents
    supervisor_agent = SupervisorAgent()
    research_agent = ResearchAgent()
    content_agent = ContentAgent()
    asset_agent = AssetGenerationAgent()
    storage_agent = StorageAgent()
    project_agent = ProjectManagementAgent()
    
    # Initialize router
    router = AgentRouter()
    
    # Create the graph
    workflow = StateGraph(LangGraphState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_agent.process)
    workflow.add_node("agent_research", research_agent.process)
    workflow.add_node("agent_content", content_agent.process)
    workflow.add_node("agent_asset_generation", asset_agent.process)
    workflow.add_node("agent_storage", storage_agent.process)
    workflow.add_node("agent_project_management", project_agent.process)
    workflow.add_node("error_handler", lambda state: state)  # Placeholder for error handling
    workflow.add_node("router", lambda state: state)  # Router node

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
            "agent_supervisor": "supervisor",  # For finalize tasks
            "error_handler": "error_handler",
            "end": END
        }
    )

    # Set entry point - supervisor starts the workflow
    def init_node(state: LangGraphState) -> LangGraphState:
        """Initialize the workflow state"""
        logger.info("Initializing multi-agent workflow")
        
        # Set initial phase
        if 'workflow_phase' not in state or state['workflow_phase'] in ["", "initial"]:
            state['workflow_phase'] = "initial"
        
        # Initialize task results if not present
        if 'task_results' not in state:
            state['task_results'] = {}
        
        # Initialize other required fields
        if 'errors' not in state:
            state['errors'] = []
        
        if 'completed_phases' not in state:
            state['completed_phases'] = []
        
        if 'supervisor_decisions' not in state:
            state['supervisor_decisions'] = []
        
        return state

    # Add init node
    workflow.add_node("init", init_node)
    workflow.add_edge("__start__", "init")  # Correct entry point
    workflow.add_edge("init", "supervisor")
    
    logger.info("Multi-agent workflow graph created successfully")
    return workflow

# Enhanced system prompt that maintains the original functionality
ENHANCED_SYSTEM_PROMPT = """
You are Rocket Reels AI News Research Assistant - a specialized agent for discovering, analyzing, and storing trending technology news, including images for social media content.

🔧 **YOUR SPECIALIZED CAPABILITIES:**

🔍 **SEARCH FUNCTION:** Uses Research Agent to search for the latest trending technology news
🕷️ **CRAWL FUNCTION:** Uses Research Agent to extract full article content and images
🗄️ **STORAGE FUNCTION:** Uses Storage Agent to store content in Supabase database
🎬 **SCRIPTING FUNCTION:** Uses Content Agent to generate viral social media scripts
🎨 **PROMPT GENERATION FUNCTION:** Uses Content Agent to generate image prompts
🖼️ **IMAGE GENERATION FUNCTION:** Uses Asset Generation Agent to create images
🎬 **VIDEO GENERATION FUNCTION:** Uses Asset Generation Agent for video creation
🎙️ **VOICE GENERATION FUNCTION:** Uses Asset Generation Agent for voiceovers

🎯 **AGENTIC WORKFLOW WITH PHASES:**

**PHASE 1 - INITIALIZATION:** Supervisor Agent understands user request completely
**PHASE 2 - RESEARCH & DISCOVERY:** Research Agent executes comprehensive search
**PHASE 3 - SELECTION & EXTRACTION:** Research Agent extracts full content
**PHASE 4 - CONTENT GENERATION:** Content Agent creates viral scripts
**PHASE 5 - ASSET CREATION:** Asset Generation Agent creates images, voice, video
**PHASE 6 - ASSEMBLY & VALIDATION:** Supervisor Agent coordinates final assembly

This multi-agent system coordinates specialized agents to accomplish your content creation goals while maintaining all the functionality of the original single-agent system.
"""

# Preserve the original agent functionality for backward compatibility
def create_original_react_agent():
    """Create the original react agent for backward compatibility"""
    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI
    import os
    
    # Import existing tools
    from agents.search_agent import search_tools
    from agents.crawl_agent import crawl_tools
    from agents.supabase_agent import supabase_tools_sync_wrapped
    from agents.prompt_generation_agent import prompt_generation_tools
    from agents.image_generation_agent import image_generation_tools
    from agents.voice_generation_agent import voice_tools
    from agents.voice_cloning_setup import voice_cloning_tools
    
    # Combine all available tools
    all_tools = (search_tools + crawl_tools + supabase_tools_sync_wrapped + 
                prompt_generation_tools + image_generation_tools + 
                voice_tools + voice_cloning_tools)
    
    # Use ChatOpenAI
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=4000,
        temperature=1.0
    )
    
    # Create the original agent with comprehensive tools
    original_agent = create_react_agent(
        model, 
        all_tools, 
        prompt=ENHANCED_SYSTEM_PROMPT
    )
    
    return original_agent

# Create both workflow options
def get_workflow_app():
    """Get the appropriate workflow application based on configuration"""
    use_multi_agent = os.getenv("USE_MULTI_AGENT", "false").lower() == "true"
    
    if use_multi_agent:
        # Use the new multi-agent workflow
        workflow_graph = create_multi_agent_workflow()
        return workflow_graph.compile()
    else:
        # Use the original react agent (for backward compatibility)
        return create_original_react_agent()

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        print("Creating multi-agent workflow...")
        
        # Create the multi-agent workflow
        workflow_graph = create_multi_agent_workflow()
        app = workflow_graph.compile()
        
        # Example: Simple prompt input
        initial_state = {
            "messages": [HumanMessage(content="Create a YouTube video about the latest AI trends in 2025")],
            "user_request": "Create a YouTube video about the latest AI trends in 2025",
            "content_type": "youtube",
            "content_generated": None,
            "assets_generated": [],
            "workflow_phase": "initial",
            "completed_phases": [],
            "error": None,
            "final_result": None,
            "user_preferences": {},
            "project_metadata": {
                "project_name": "AI Trends Video",
                "created_at": datetime.now().isoformat()
            },
            "research_data": None,
            "script": None,
            "image_prompts": [],
            "voiceover": None,
            "task_results": {},
            "errors": [],
            "workflow_completed": False
        }
        
        print("Running multi-agent workflow...")
        result = app.invoke(initial_state)
        
        print("\nMulti-agent workflow completed!")
        print(f"Final workflow phase: {result.get('workflow_phase')}")
        print(f"Workflow completed: {result.get('workflow_completed', False)}")
        print(f"Completed phases: {result.get('completed_phases', [])}")
        print(f"Errors: {len(result.get('errors', []))}")
        
        if result.get('messages'):
            last_message = result['messages'][-1]
            if hasattr(last_message, 'content'):
                print(f"Last message preview: {last_message.content[:100]}...")

    asyncio.run(main())