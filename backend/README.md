# Agentic Content Production System

A sophisticated multi-agent system built with LangGraph for automated content production workflows. The system orchestrates specialized agents to handle research, content creation, asset generation, storage management, and project coordination.

## System Architecture

### Core Components

1. **Supervisor Agent**: Orchestrates the entire workflow and makes strategic decisions
2. **Research Agent**: Finds and processes high-quality source material
3. **Content Agent**: Creates engaging scripts and plans visual elements
4. **Asset Generation Agent**: Creates images, voice recordings, and finds b-roll footage
5. **Storage Agent**: Manages data storage in Supabase and Google Drive
6. **Project Management Agent**: Handles Notion integration and project tracking

### Workflow Phases

1. **Planning**: Initial project setup and task assignment
2. **Research**: Gathering and analyzing source material
3. **Content Creation**: Script development and shot planning
4. **Asset Generation**: Creating visual and audio elements
5. **Storage Management**: Organizing and storing project data
6. **Quality Control**: Final review and quality assurance
7. **Finalization**: Project completion and handoff

## Project Structure

```
├── models.py                 # Pydantic data models and schemas
├── tools.py                  # Tool interfaces for each agent
├── langgraph_workflow.py     # LangGraph workflow implementation
├── agent_prompts.yaml        # Agent configurations and prompts
├── main.py                   # Main system entry point
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your environment variables for external services:
   - Supabase credentials
   - Google Drive API credentials
   - Notion API token
   - OpenAI/Anthropic API keys

## Usage

### Basic Usage

```python
from main import AgenticContentProductionSystem
from models import ContentType

# Initialize the system
system = AgenticContentProductionSystem()

# Create a new project
project_id = system.create_project(
    project_name="AI Trends 2024 Video",
    content_request="Create an engaging YouTube video about AI trends",
    content_type=ContentType.YOUTUBE_VIDEO,
    target_audience="tech enthusiasts"
)

# Run the complete workflow
result = system.run_workflow(project_id)
```

### Advanced Usage

```python
# Get project status
status = system.get_project_status(project_id)

# Get agent capabilities
capabilities = system.get_agent_capabilities()

# Run specific workflow phases
# (Implementation depends on your specific needs)
```

## Data Models

The system uses comprehensive Pydantic models for type safety and validation:

- **AgentState**: Main state container for the entire workflow
- **ResearchData**: Structured research information
- **Script**: Content scripts with shot breakdowns
- **Asset**: Visual and audio assets
- **Task**: Individual workflow tasks
- **QualityCheck**: Quality assurance tracking

## Agent Tools

Each agent has access to specialized tools:

### Research Agent
- `search_articles`: Search for trending articles
- `extract_article_content`: Extract full content from URLs
- `analyze_content`: Analyze and summarize content
- `verify_information`: Verify information accuracy

### Content Agent
- `create_script`: Generate compelling scripts
- `break_down_shots`: Break scripts into shots
- `plan_visual_elements`: Plan visual components
- `create_hook`: Create engaging hooks
- `create_call_to_action`: Generate CTAs

### Asset Generation Agent
- `generate_image_prompt`: Create AI image prompts
- `create_image`: Generate AI images
- `generate_voiceover`: Create voice recordings
- `find_broll`: Find relevant footage
- `organize_assets`: Organize generated assets

### Storage Agent
- `store_in_supabase`: Store data in database
- `create_google_drive_folder`: Organize files
- `upload_to_google_drive`: Upload files
- `tag_content`: Add metadata tags
- `retrieve_content`: Retrieve stored content

### Project Management Agent
- `create_notion_workspace`: Set up Notion workspace
- `create_notion_page`: Create project pages
- `update_project_status`: Track progress
- `generate_progress_report`: Create reports
- `assign_task`: Assign tasks to agents
- `track_milestone`: Track project milestones

## Configuration

The system is configured through `agent_prompts.yaml`, which contains:

- Agent system prompts and capabilities
- Workflow phase definitions
- Quality standards
- Communication protocols
- Error handling strategies

## Error Handling

The system includes comprehensive error handling:

- Task retry mechanisms
- Agent reassignment on failure
- Error escalation to supervisor
- Graceful degradation

## Quality Assurance

Built-in quality checks ensure:

- Content accuracy and relevance
- Asset quality and consistency
- Technical compliance
- Brand alignment

## Extensibility

The system is designed for easy extension:

- Add new agents by implementing the `SpecializedAgent` base class
- Add new tools by extending the tool registry
- Modify workflow phases in the configuration
- Customize quality standards and criteria

## Future Enhancements

- Real-time collaboration features
- Advanced AI model integration
- Custom workflow templates
- Performance analytics
- Integration with additional platforms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.