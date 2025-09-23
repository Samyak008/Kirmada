from pathlib import Path
import sys

# Ensure we can import sibling modules when loaded from project root
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from langgraph_workflow import create_workflow_graph, UserInput  # see: d:\Kirmada\backend\langgraph_workflow.py

def production_workflow():
    # Ensure the workflow loads the YAML from the backend folder
    cfg = Path(__file__).with_name("agent_prompts.yaml")  # see: d:\Kirmada\backend\agent_prompts.yaml
    graph = create_workflow_graph(config_path=str(cfg))
    # Compile without extra kwargs for older LangGraph versions
    app = graph.compile()
    # Attach input schema attribute for Studio to render input form
    try:
        setattr(app, "input_schema", UserInput)
    except Exception:
        pass
    return app