from pathlib import Path
from langgraph_workflow import create_workflow_graph  # see: d:\Kirmada\backend\langgraph_workflow.py

def production_workflow():
    # Ensure the workflow loads the YAML from the backend folder
    cfg = Path(__file__).with_name("agent_prompts.yaml")  # see: d:\Kirmada\backend\agent_prompts.yaml
    graph = create_workflow_graph(config_path=str(cfg))
    return graph.compile()