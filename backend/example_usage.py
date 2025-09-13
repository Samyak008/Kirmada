"""
Example usage of the Agentic Content Production System
"""

from main import AgenticContentProductionSystem
from models import ContentType
import asyncio


async def example_workflow():
    """Example of running a complete content production workflow"""
    
    print("🚀 Initializing Agentic Content Production System...")
    
    # Initialize the system
    system = AgenticContentProductionSystem()
    
    # Create a new project
    print("\n📝 Creating new project...")
    project_id = system.create_project(
        project_name="AI Trends 2024 Deep Dive",
        content_request="Create a comprehensive YouTube video about the top 10 AI trends that will shape 2024, including practical examples and expert insights",
        content_type=ContentType.YOUTUBE_VIDEO,
        target_audience="tech professionals, AI researchers, and business leaders",
        deadline="2024-12-31T23:59:59"
    )
    
    print(f"✅ Project created with ID: {project_id}")
    
    # Get initial project status
    print("\n📊 Initial project status:")
    status = system.get_project_status(project_id)
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Display available agents
    print("\n🤖 Available agents and their capabilities:")
    capabilities = system.get_agent_capabilities()
    for agent_type, agent_info in capabilities.items():
        print(f"\n  {agent_info['name']}:")
        print(f"    Description: {agent_info['description']}")
        print(f"    Capabilities: {', '.join(agent_info['capabilities'])}")
    
    # Run the complete workflow
    print("\n🔄 Starting content production workflow...")
    print("This will orchestrate all agents through the complete production pipeline:")
    print("  1. Planning & Project Setup")
    print("  2. Research & Information Gathering")
    print("  3. Content Creation & Script Writing")
    print("  4. Asset Generation (Images, Audio, B-roll)")
    print("  5. Storage & Organization")
    print("  6. Quality Control & Review")
    print("  7. Finalization & Handoff")
    
    result = system.run_workflow(project_id)
    
    # Display results
    print("\n📋 Workflow Results:")
    if result["success"]:
        print("✅ Workflow completed successfully!")
        print(f"   Project ID: {result['project_id']}")
        print(f"   Workflow completed: {result['workflow_completed']}")
        
        # Display final state information
        final_state = result["final_state"]
        agent_state = final_state["agent_state"]
        
        print(f"\n📈 Final Project Statistics:")
        print(f"   Tasks completed: {len(agent_state.completed_tasks)}")
        print(f"   Assets created: {len(agent_state.assets)}")
        print(f"   Research sources: {len(agent_state.research_data.sources) if agent_state.research_data else 0}")
        print(f"   Script shots: {len(agent_state.script.shots) if agent_state.script else 0}")
        print(f"   Database records: {len(agent_state.database_records)}")
        print(f"   Notion pages: {len(agent_state.notion_pages)}")
        
        if agent_state.script:
            print(f"\n📝 Generated Content:")
            print(f"   Title: {agent_state.script.title}")
            print(f"   Duration: {agent_state.script.duration_minutes} minutes")
            print(f"   Target Audience: {agent_state.script.target_audience}")
            print(f"   Content Type: {agent_state.script.content_type}")
            print(f"   Hook: {agent_state.script.hook}")
            print(f"   Call to Action: {agent_state.script.call_to_action}")
        
    else:
        print("❌ Workflow failed!")
        print(f"   Error: {result['error']}")
    
    # Get final project status
    print("\n📊 Final project status:")
    final_status = system.get_project_status(project_id)
    for key, value in final_status.items():
        print(f"  {key}: {value}")


def example_agent_interaction():
    """Example of how agents interact within the system"""
    
    print("\n🔗 Agent Interaction Flow:")
    print("""
    Supervisor Agent
    ├── Analyzes content request
    ├── Creates production plan
    ├── Assigns tasks to specialized agents
    │
    ├── Research Agent
    │   ├── Searches for trending articles
    │   ├── Extracts and analyzes content
    │   └── Provides structured research data
    │
    ├── Content Agent
    │   ├── Creates engaging scripts
    │   ├── Breaks down into shots
    │   └── Plans visual elements
    │
    ├── Asset Generation Agent
    │   ├── Generates AI images
    │   ├── Creates voice recordings
    │   └── Finds b-roll footage
    │
    ├── Storage Agent
    │   ├── Stores data in Supabase
    │   ├── Organizes files in Google Drive
    │   └── Manages metadata
    │
    └── Project Management Agent
        ├── Sets up Notion workspace
        ├── Tracks progress
        └── Generates reports
    """)


def example_tool_usage():
    """Example of how tools are used by agents"""
    
    print("\n🛠️ Tool Usage Examples:")
    
    # Research Agent Tools
    print("\n  Research Agent Tools:")
    print("    - search_articles(query='AI trends 2024', max_results=10)")
    print("    - extract_article_content(urls=['https://example.com/article1'])")
    print("    - analyze_content(content='...', analysis_type='summary')")
    print("    - verify_information(claims=['AI will transform healthcare'])")
    
    # Content Agent Tools
    print("\n  Content Agent Tools:")
    print("    - create_script(research_data=..., content_type='youtube_video')")
    print("    - break_down_shots(script=..., shot_duration_range=(3, 15))")
    print("    - plan_visual_elements(shots=..., style_preferences={})")
    print("    - create_hook(topic='AI trends', target_audience='tech professionals')")
    
    # Asset Generation Agent Tools
    print("\n  Asset Generation Agent Tools:")
    print("    - generate_image_prompt(shot_description='Opening title shot')")
    print("    - create_image(prompts=['...'], style='realistic')")
    print("    - generate_voiceover(script_text='...', voice_type='professional_male')")
    print("    - find_broll(keywords=['AI', 'technology'], duration_range=(10, 60))")
    
    # Storage Agent Tools
    print("\n  Storage Agent Tools:")
    print("    - store_in_supabase(table_name='articles', data={...})")
    print("    - create_google_drive_folder(folder_name='AI_Trends_2024')")
    print("    - upload_to_google_drive(file_paths=['...'], destination_folder_id='...')")
    print("    - tag_content(content_id='...', tags=['AI', 'trends', '2024'])")
    
    # Project Management Agent Tools
    print("\n  Project Management Agent Tools:")
    print("    - create_notion_workspace(project_name='AI Trends 2024')")
    print("    - create_notion_page(title='Project Overview', content={...})")
    print("    - update_project_status(project_id='...', status='in_progress')")
    print("    - generate_progress_report(project_id='...', report_type='comprehensive')")


if __name__ == "__main__":
    print("🎬 Agentic Content Production System - Example Usage")
    print("=" * 60)
    
    # Run the main example
    asyncio.run(example_workflow())
    
    # Show additional examples
    example_agent_interaction()
    example_tool_usage()
    
    print("\n🎉 Example completed! Check the generated files and data structures.")
    print("\nNext steps:")
    print("1. Set up your environment variables (see env_example.txt)")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Implement the actual tool functionality")
    print("4. Customize agent prompts and workflows as needed")
    print("5. Add your specific integrations (APIs, databases, etc.)")
