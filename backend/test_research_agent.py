# Simple test to verify the research agent works

from langgraph_workflow import create_workflow_graph, UserInput, InputType
from models import ContentType


def main():
    print("Testing the improved research agent...")
    
    # Create the workflow
    try:
        graph = create_workflow_graph()
        app = graph.compile()
        
        # Create a simple user input
        user_input = UserInput(
            input_type=InputType.STATEMENT,
            content="I want to create a YouTube video about the latest AI trends in 2025",
            project_name="AI Trends Video",
            content_type=ContentType.YOUTUBE_VIDEO,
            target_audience="tech enthusiasts"
        )
        
        print(f"Input created: {user_input.content_type} for {user_input.target_audience}")
        
        # Run the workflow
        result = app.invoke(user_input)
        print("Workflow completed successfully!")
        print(f"Project: {result['agent_state'].project_name}")
        print(f"Content request: {result['agent_state'].content_request}")
        print(f"Current phase: {result['agent_state'].current_phase}")
        print(f"Workflow completed: {result['workflow_completed']}")
        
        if result['messages']:
            print(f"Last message: {result['messages'][-1].content[:100]}...")
    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()