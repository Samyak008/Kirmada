from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from models import AgentState, ResearchData, Script, Asset, ProjectFolder, DatabaseRecord, NotionPage, Task


class ToolResult(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


# Research Agent Tools
class SearchArticlesTool(BaseModel):
    """Search for trending articles on a given topic"""
    query: str = Field(..., description="Search query for articles")
    max_results: int = Field(default=10, description="Maximum number of results to return")
    date_range: Optional[str] = Field(default=None, description="Date range for search (e.g., 'last_week', 'last_month')")
    language: str = Field(default="en", description="Language for search results")


class ExtractArticleContentTool(BaseModel):
    """Extract full content from article URLs"""
    urls: List[str] = Field(..., description="List of article URLs to extract content from")
    include_metadata: bool = Field(default=True, description="Whether to include article metadata")


class AnalyzeContentTool(BaseModel):
    """Analyze and summarize article content"""
    content: str = Field(..., description="Article content to analyze")
    analysis_type: str = Field(default="summary", description="Type of analysis (summary, sentiment, key_points)")
    max_key_points: int = Field(default=5, description="Maximum number of key points to extract")


class VerifyInformationTool(BaseModel):
    """Verify the accuracy of information"""
    claims: List[str] = Field(..., description="List of claims to verify")
    sources: List[str] = Field(default=[], description="Sources to check against")


# Content Agent Tools
class CreateScriptTool(BaseModel):
    """Create a script based on research data"""
    research_data: ResearchData = Field(..., description="Research data to base script on")
    content_type: str = Field(..., description="Type of content (youtube_video, social_media_post, etc.)")
    target_audience: str = Field(..., description="Target audience for the content")
    duration_minutes: float = Field(..., description="Desired duration in minutes")
    tone: str = Field(default="professional", description="Tone of the content")


class BreakDownShotsTool(BaseModel):
    """Break down script into individual shots"""
    script: Script = Field(..., description="Script to break down into shots")
    shot_duration_range: tuple = Field(default=(3, 15), description="Range of shot durations in seconds")


class ShotAnalysisTool(BaseModel):
    """Analyze shots for timing, types, and visual requirements"""
    shots: List[Dict[str, Any]] = Field(..., description="List of shots to analyze")
    analyze_timing: bool = Field(default=True, description="Whether to analyze shot timing")
    analyze_types: bool = Field(default=True, description="Whether to categorize shot types")
    analyze_visual_requirements: bool = Field(default=True, description="Whether to analyze visual requirements")


class GenerateVisualTableTool(BaseModel):
    """Generate visual table for shot organization"""
    shot_breakdown: List[Dict[str, Any]] = Field(..., description="Shot breakdown data")
    include_timing: bool = Field(default=True, description="Include timing information")
    include_assets: bool = Field(default=True, description="Include asset references")
    format: str = Field(default="table", description="Output format (table, json, markdown)")


class PlanVisualElementsTool(BaseModel):
    """Plan visual elements for each shot"""
    shots: List[Dict[str, Any]] = Field(..., description="List of shots to plan visuals for")
    style_preferences: Dict[str, Any] = Field(default={}, description="Visual style preferences")


class CreateHookTool(BaseModel):
    """Create an engaging hook for the content"""
    topic: str = Field(..., description="Main topic of the content")
    target_audience: str = Field(..., description="Target audience")
    content_type: str = Field(..., description="Type of content")


class CreateCallToActionTool(BaseModel):
    """Create a compelling call to action"""
    content_summary: str = Field(..., description="Summary of the content")
    desired_action: str = Field(..., description="Action you want the audience to take")
    platform: str = Field(..., description="Platform where content will be published")


# Asset Generation Agent Tools
class GenerateImagePromptTool(BaseModel):
    """Generate detailed prompts for AI image creation"""
    shot_description: str = Field(..., description="Description of the shot")
    style: str = Field(default="realistic", description="Visual style for the image")
    mood: str = Field(default="professional", description="Mood or atmosphere")
    technical_specs: Dict[str, Any] = Field(default={}, description="Technical specifications")


class CreateImageTool(BaseModel):
    """Create AI-generated images"""
    prompts: List[str] = Field(..., description="List of image prompts")
    style: str = Field(default="realistic", description="Visual style")
    resolution: str = Field(default="1024x1024", description="Image resolution")
    quality: str = Field(default="high", description="Image quality setting")


class GenerateVoiceoverTool(BaseModel):
    """Generate voiceover recordings from script"""
    script_text: str = Field(..., description="Text to convert to voiceover")
    voice_type: str = Field(default="professional_male", description="Type of voice")
    emotion: str = Field(default="neutral", description="Emotional tone")
    speed: float = Field(default=1.0, description="Speech speed multiplier")
    language: str = Field(default="en", description="Language for voiceover")


class FindBrollTool(BaseModel):
    """Find relevant b-roll footage"""
    keywords: List[str] = Field(..., description="Keywords to search for")
    duration_range: tuple = Field(default=(10, 60), description="Duration range in seconds")
    quality: str = Field(default="hd", description="Video quality requirement")
    license_type: str = Field(default="royalty_free", description="License type for footage")


class OrganizeAssetsTool(BaseModel):
    """Organize generated assets"""
    assets: List[Asset] = Field(..., description="Assets to organize")
    organization_scheme: str = Field(default="by_shot", description="Organization scheme")
    naming_convention: str = Field(default="shot_number_type", description="Naming convention")


class AssetGatheringTool(BaseModel):
    """Gather and organize all project assets"""
    project_folder_path: str = Field(..., description="Path to project folder")
    assets: List[Asset] = Field(..., description="Assets to gather")
    create_folder_structure: bool = Field(default=True, description="Create organized folder structure")
    generate_manifest: bool = Field(default=True, description="Generate asset manifest")


class ParallelSyncTool(BaseModel):
    """Synchronize parallel processing tasks"""
    task_results: Dict[str, Any] = Field(..., description="Results from parallel tasks")
    wait_for_all: bool = Field(default=True, description="Wait for all tasks to complete")
    timeout_seconds: int = Field(default=300, description="Timeout for parallel tasks")


# Storage Agent Tools
class StoreInSupabaseTool(BaseModel):
    """Store data in Supabase database"""
    table_name: str = Field(..., description="Supabase table name")
    data: Dict[str, Any] = Field(..., description="Data to store")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class CreateGoogleDriveFolderTool(BaseModel):
    """Create organized folder structure in Google Drive"""
    folder_name: str = Field(..., description="Name of the folder to create")
    parent_folder_id: Optional[str] = Field(default=None, description="Parent folder ID")
    folder_structure: Dict[str, Any] = Field(default={}, description="Nested folder structure")


class UploadToGoogleDriveTool(BaseModel):
    """Upload files to Google Drive"""
    file_paths: List[str] = Field(..., description="Local file paths to upload")
    destination_folder_id: str = Field(..., description="Google Drive folder ID")
    file_names: Optional[List[str]] = Field(default=None, description="Custom file names")


class TagContentTool(BaseModel):
    """Add tags and metadata to content"""
    content_id: str = Field(..., description="ID of the content to tag")
    tags: List[str] = Field(..., description="Tags to add")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class RetrieveContentTool(BaseModel):
    """Retrieve stored content"""
    content_id: str = Field(..., description="ID of content to retrieve")
    include_metadata: bool = Field(default=True, description="Whether to include metadata")


# Project Management Agent Tools
class CreateNotionWorkspaceTool(BaseModel):
    """Create Notion workspace for project"""
    project_name: str = Field(..., description="Name of the project")
    workspace_template: str = Field(default="content_production", description="Template to use")
    team_members: List[str] = Field(default=[], description="List of team member emails")


class CreateNotionPageTool(BaseModel):
    """Create Notion page"""
    title: str = Field(..., description="Title of the page")
    parent_page_id: Optional[str] = Field(default=None, description="Parent page ID")
    content: Dict[str, Any] = Field(..., description="Page content structure")
    page_type: str = Field(default="page", description="Type of page (page, database, etc.)")


class UpdateProjectStatusTool(BaseModel):
    """Update project status and milestones"""
    project_id: str = Field(..., description="Project ID to update")
    status: str = Field(..., description="New status")
    milestone_updates: Dict[str, Any] = Field(default={}, description="Milestone updates")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class GenerateProgressReportTool(BaseModel):
    """Generate project progress report"""
    project_id: str = Field(..., description="Project ID")
    report_type: str = Field(default="comprehensive", description="Type of report")
    include_metrics: bool = Field(default=True, description="Whether to include metrics")
    format: str = Field(default="json", description="Report format (json, markdown, pdf)")


class AssignTaskTool(BaseModel):
    """Assign task to team member"""
    task_description: str = Field(..., description="Description of the task")
    assigned_to: str = Field(..., description="Agent type to assign to")
    priority: int = Field(default=1, description="Task priority (1-5)")
    deadline: Optional[str] = Field(default=None, description="Task deadline")
    dependencies: List[str] = Field(default=[], description="Task dependencies")


class TrackMilestoneTool(BaseModel):
    """Track project milestone"""
    milestone_name: str = Field(..., description="Name of the milestone")
    status: str = Field(..., description="Milestone status")
    completion_percentage: float = Field(default=0.0, description="Completion percentage")
    notes: Optional[str] = Field(default=None, description="Milestone notes")


# Tool Registry
TOOL_REGISTRY = {
    "research": {
        "search_articles": SearchArticlesTool,
        "extract_article_content": ExtractArticleContentTool,
        "analyze_content": AnalyzeContentTool,
        "verify_information": VerifyInformationTool,
    },
    "content": {
        "create_script": CreateScriptTool,
        "break_down_shots": BreakDownShotsTool,
        "shot_analysis": ShotAnalysisTool,
        "generate_visual_table": GenerateVisualTableTool,
        "plan_visual_elements": PlanVisualElementsTool,
        "create_hook": CreateHookTool,
        "create_call_to_action": CreateCallToActionTool,
    },
    "asset_generation": {
        "generate_image_prompt": GenerateImagePromptTool,
        "create_image": CreateImageTool,
        "generate_voiceover": GenerateVoiceoverTool,
        "find_broll": FindBrollTool,
        "organize_assets": OrganizeAssetsTool,
        "asset_gathering": AssetGatheringTool,
        "parallel_sync": ParallelSyncTool,
    },
    "storage": {
        "store_in_supabase": StoreInSupabaseTool,
        "create_google_drive_folder": CreateGoogleDriveFolderTool,
        "upload_to_google_drive": UploadToGoogleDriveTool,
        "tag_content": TagContentTool,
        "retrieve_content": RetrieveContentTool,
    },
    "project_management": {
        "create_notion_workspace": CreateNotionWorkspaceTool,
        "create_notion_page": CreateNotionPageTool,
        "update_project_status": UpdateProjectStatusTool,
        "generate_progress_report": GenerateProgressReportTool,
        "assign_task": AssignTaskTool,
        "track_milestone": TrackMilestoneTool,
    }
}


def get_tools_for_agent(agent_type: str) -> Dict[str, Any]:
    """Get available tools for a specific agent type"""
    return TOOL_REGISTRY.get(agent_type, {})


def validate_tool_input(agent_type: str, tool_name: str, input_data: Dict[str, Any]) -> bool:
    """Validate tool input against the tool schema"""
    tools = get_tools_for_agent(agent_type)
    if tool_name not in tools:
        return False
    
    tool_class = tools[tool_name]
    try:
        tool_class(**input_data)
        return True
    except Exception:
        return False


# Tool Implementations

def search_articles_tool(query: str, max_results: int = 8) -> ToolResult:
    """Search for articles using the ResearchAgent"""
    try:
        from langgraph_workflow import ResearchAgent
        from models import AgentType

        # Create a temporary config for the research agent
        config = {
            "agents": {
                "research": {
                    "system_prompt": "You are a research agent specialized in finding high-quality information."
                }
            }
        }

        # Initialize research agent
        agent = ResearchAgent(config)

        # Execute search
        result = agent._execute_task(
            type('Task', (), {'task_type': 'search'})(),
            type('AgentState', (), {'content_request': query})()
        )

        if result.get("success", True):
            return ToolResult(
                success=True,
                data=result,
                metadata={"tool": "search_articles", "query": query}
            )
        else:
            return ToolResult(
                success=False,
                error_message=result.get("error", "Search failed"),
                metadata={"tool": "search_articles", "query": query}
            )

    except Exception as e:
        return ToolResult(
            success=False,
            error_message=str(e),
            metadata={"tool": "search_articles", "query": query}
        )


def extract_article_content_tool(urls: List[str], include_metadata: bool = True) -> ToolResult:
    """Extract content from article URLs using the ResearchAgent"""
    try:
        from langgraph_workflow import ResearchAgent
        from models import AgentType, AgentState, ResearchData, Source

        # Create a temporary config for the research agent
        config = {
            "agents": {
                "research": {
                    "system_prompt": "You are a research agent specialized in extracting content from articles."
                }
            }
        }

        # Initialize research agent
        agent = ResearchAgent(config)

        # Create mock agent state with sources
        sources = [Source(url=url, title="", content="", relevance_score=0.8) for url in urls]
        agent_state = AgentState(
            project_id="temp",
            project_name="temp",
            content_request="temp",
            research_data=ResearchData(topic="temp", sources=sources)
        )

        # Execute crawl
        result = agent._execute_task(
            type('Task', (), {'task_type': 'crawl'})(),
            agent_state
        )

        if result.get("crawl_completed", False):
            return ToolResult(
                success=True,
                data=result,
                metadata={"tool": "extract_article_content", "urls": urls}
            )
        else:
            return ToolResult(
                success=False,
                error_message="Content extraction failed",
                metadata={"tool": "extract_article_content", "urls": urls}
            )

    except Exception as e:
        return ToolResult(
            success=False,
            error_message=str(e),
            metadata={"tool": "extract_article_content", "urls": urls}
        )
