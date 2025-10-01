# Backend Folder Documentation

## Overview
The backend folder contains the implementation of an Agentic Content Production System built with LangGraph. It orchestrates multiple specialized agents to create content from user prompts through a structured workflow. The system handles everything from research and content creation to asset generation and project management.

## Key Components

### 1. System Architecture
The system follows a supervisor-based agent architecture with specialized agents for different tasks:

- **Supervisor Agent**: Orchestrates the workflow and makes strategic decisions
- **Research Agent**: Handles information gathering and content research
- **Content Agent**: Creates scripts and plans visual elements
- **Asset Generation Agent**: Generates images, voiceovers, and finds b-roll footage
- **Storage Agent**: Manages data persistence in Supabase and Google Drive
- **Project Management Agent**: Handles Notion integration and tracking

### 2. Core Files

#### `main.py`
Entry point for the system. Contains the `AgenticContentProductionSystem` class that initializes the workflow and provides methods to create projects from user prompts. It also handles agent capabilities and system initialization.

#### `langgraph_workflow.py`
Implements the LangGraph workflow structure with:
- State management using TypedDict
- Agent routing logic
- Node definitions for each agent type
- Conditional edge routing based on workflow phases
- Error handling and workflow progression

#### `models.py`
Defines all data models and enumerations:
- `AgentState`: Core state model containing project information, research data, scripts, assets, etc.
- `Task`: Model for tracking agent tasks with status, priority, and dependencies
- `AgentType`: Enum for different agent types (supervisor, research, content, etc.)
- `TaskStatus`: Enum for task statuses (pending, in_progress, completed, etc.)
- `ContentType`: Enum for different content types (youtube_video, social_media_post, etc.)
- Supporting models for research data, scripts, assets, project folders, and Notion pages

#### `tools.py`
Contains tool definitions and implementations:
- Tool schemas for each agent type (research, content, asset_gen, storage, project_management)
- Tool registry mapping agent types to available tools
- Validation functions for tool inputs
- Implementation of core tools like search_articles_tool and extract_article_content_tool

### 3. Agent Implementations

#### `agents/supervisor_agent.py`
The supervisor agent coordinates the workflow and makes strategic decisions:
- Analyzes user content requests to determine the appropriate workflow path
- Creates tasks for other agents based on current phase
- Manages state transitions between different workflow phases
- Handles initialization and finalization of projects

#### `agents/research_agent.py`
Handles research and information gathering:
- Searches for relevant content based on user requests
- Crawls and extracts content from sources
- Analyzes and summarizes research findings
- Verifies information accuracy

#### `agents/content_agent.py`
Responsible for content creation:
- Generates scripts based on research data
- Creates shot breakdowns and timing analysis
- Plans visual elements for content
- Creates hooks and calls to action

#### `agents/asset_generation_agent.py`
Handles creation of visual and audio assets:
- Generates image prompts for AI creation
- Creates images using AI tools
- Generates voiceovers from scripts
- Finds relevant b-roll footage
- Organizes and manages assets

#### `agents/storage_agent.py`
Manages data persistence:
- Stores content in Supabase database
- Creates organized Google Drive folder structures
- Uploads files to Google Drive
- Adds tags and metadata to content
- Retrieves stored content

#### `agents/project_management_agent.py`
Handles project coordination:
- Creates Notion workspaces for project collaboration
- Updates project status and milestones
- Generates progress reports
- Assigns tasks to team members
- Tracks project milestones

### 4. Workflow Phases

The system operates through multiple distinct phases:

1. **Search Phase**: Search for trending articles and content
2. **Crawl Phase**: Extract content from identified URLs
3. **Store Article Phase**: Store article data in database
4. **Generate Script Phase**: Create engaging script from research
5. **Store Script Phase**: Store generated script
6. **Shot Analysis Phase**: Analyze script and break down into shots
7. **Parallel Generation Phase**: Generate prompts, images, voice, and b-roll in parallel
8. **Visual Table Generation Phase**: Generate visual table for organization
9. **Asset Gathering Phase**: Gather and organize all project assets
10. **Notion Integration Phase**: Set up Notion workspace and tracking
11. **Finalize Phase**: Final review and project completion

### 5. Configuration

#### `agent_prompts.yaml`
Contains configuration for all agents including:
- System prompts for each agent type
- Agent descriptions and capabilities
- Workflow phase definitions
- Agent communication protocols
- Quality standards and error handling strategies

#### `config.py`
Application configuration settings (if present)

### 6. Execution Flow

The system follows this general flow:

1. **Initialization**: System initializes with configuration from agent_prompts.yaml
2. **User Input**: Accepts content requests from users
3. **Supervisor Decision**: Supervisor analyzes request and determines workflow path
4. **Task Creation**: Creates appropriate tasks based on current phase
5. **Agent Processing**: Relevant agents process assigned tasks
6. **State Updates**: Updates AgentState with results
7. **Routing**: AgentRouter determines next agent based on current state
8. **Iteration**: Continues until workflow is completed
9. **Finalization**: Supervisor handles final review and completion

### 7. State Management

The system uses a `WorkflowState` TypedDict that includes:
- `agent_state`: Core application state (AgentState model)
- `messages`: Communication between agents (LangChain messages)
- `current_agent`: Currently active agent
- `next_agent`: Agent to process next
- `task_results`: Results from completed tasks
- `errors`: Error tracking
- `workflow_completed`: Completion status

### 8. Testing Files

- `test_basic.py`: Basic functionality tests
- `test_crawler.py`: Crawler-specific tests
- `test_research_agent.py`: Research agent tests
- Other test files for specific components

### 9. Dependencies

The system relies on:
- LangGraph for workflow orchestration
- Pydantic for data validation
- LangChain for LLM interactions
- Supabase for database storage
- Various tools for external service integration

## Usage

To use the system:
1. Initialize the AgenticContentProductionSystem
2. Call create_project_from_prompt() with user content request
3. The system will automatically execute the appropriate workflow phases
4. Results are returned in the final state with all generated content