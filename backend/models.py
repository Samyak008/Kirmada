from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContentType(str, Enum):
    YOUTUBE_VIDEO = "youtube_video"
    SOCIAL_MEDIA_POST = "social_media_post"
    BLOG_ARTICLE = "blog_article"
    PODCAST = "podcast"
    PRESENTATION = "presentation"


class AgentType(str, Enum):
    SUPERVISOR = "supervisor"
    RESEARCH = "research"
    CONTENT = "content"
    ASSET_GENERATION = "asset_generation"
    STORAGE = "storage"
    PROJECT_MANAGEMENT = "project_management"


class Source(BaseModel):
    url: str
    title: str
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    content: str
    summary: Optional[str] = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    tags: List[str] = []


class ResearchData(BaseModel):
    topic: str
    sources: List[Source] = []
    key_findings: List[str] = []
    trends: List[str] = []
    statistics: Dict[str, Any] = {}
    competitor_analysis: Optional[Dict[str, Any]] = None
    research_notes: str = ""


class Shot(BaseModel):
    shot_number: int
    description: str
    duration_seconds: float
    visual_elements: List[str] = []
    audio_notes: Optional[str] = None
    camera_angle: Optional[str] = None
    lighting_notes: Optional[str] = None
    shot_type: str = "medium"  # close_up, medium, wide, establishing
    timing_start: float = 0.0
    timing_end: float = 0.0
    visual_style: Optional[str] = None
    transition_type: Optional[str] = None


class Script(BaseModel):
    title: str
    content_type: ContentType
    target_audience: str
    duration_minutes: float
    hook: str
    main_content: str
    call_to_action: str
    shots: List[Shot] = []
    visual_plan: List[str] = []
    tone: str = "professional"
    keywords: List[str] = []


class Asset(BaseModel):
    asset_id: str
    asset_type: str  # image, audio, video, broll
    file_path: str
    description: str
    shot_number: Optional[int] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)
    file_size: Optional[int] = None
    duration: Optional[float] = None  # for audio/video assets


class ProjectFolder(BaseModel):
    folder_id: str
    folder_name: str
    parent_folder_id: Optional[str] = None
    google_drive_path: str
    created_at: datetime = Field(default_factory=datetime.now)
    files: List[str] = []  # file IDs


class DatabaseRecord(BaseModel):
    record_id: str
    table_name: str
    data: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    tags: List[str] = []


class NotionPage(BaseModel):
    page_id: str
    title: str
    content: Dict[str, Any]
    parent_page_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_edited: Optional[datetime] = None


class Task(BaseModel):
    task_id: str
    agent_type: AgentType
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    priority: int = Field(default=1, ge=1, le=5)
    dependencies: List[str] = []  # task IDs this task depends on
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class AgentState(BaseModel):
    # Core project information
    project_id: str
    project_name: str
    content_request: str
    content_type: ContentType
    target_audience: str
    deadline: Optional[datetime] = None
    
    # Current workflow state
    current_phase: str = "planning"
    current_agent: Optional[AgentType] = None
    workflow_completed: bool = False
    
    # Research data
    research_data: Optional[ResearchData] = None
    
    # Content creation
    script: Optional[Script] = None
    
    # Shot analysis (from original workflow)
    shot_breakdown: List[Dict[str, Any]] = []
    shot_timing: List[Dict[str, Any]] = []
    shot_types: List[str] = []
    
    # Asset generation tracking
    prompts_generated: List[Dict[str, Any]] = []
    images_generated: List[str] = []
    image_prompt_mapping: Dict[str, Dict[str, Any]] = {}
    voice_files: List[str] = []
    broll_assets: Dict[str, Any] = {}
    
    # Visual table generation
    visual_table: Optional[Dict[str, Any]] = None
    
    # Assets
    assets: List[Asset] = []
    
    # Storage management
    project_folders: List[ProjectFolder] = []
    database_records: List[DatabaseRecord] = []
    project_folder_path: str = ""
    asset_organization_result: str = ""
    
    # Project management
    notion_pages: List[NotionPage] = []
    notion_project_id: str = ""
    notion_status: str = ""
    tasks: List[Task] = []
    project_status: str = "active"
    
    # Workflow tracking
    completed_tasks: List[str] = []  # task IDs
    failed_tasks: List[str] = []
    next_actions: List[str] = []
    current_step: str = "search"  # From original workflow
    
    # Quality control
    quality_checks: List[Dict[str, Any]] = []
    revisions_needed: List[str] = []
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    
    # Agent communication
    agent_messages: List[Dict[str, Any]] = []
    supervisor_decisions: List[Dict[str, Any]] = []
    
    # Error handling
    errors: List[Dict[str, Any]] = []
    retry_count: int = 0
    max_retries: int = 3


class AgentMessage(BaseModel):
    from_agent: AgentType
    to_agent: Optional[AgentType] = None
    message_type: str  # task_assignment, result, error, question
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str


class SupervisorDecision(BaseModel):
    decision_id: str
    context: str
    chosen_agent: AgentType
    reasoning: str
    expected_outcome: str
    priority: int = Field(ge=1, le=5)
    timestamp: datetime = Field(default_factory=datetime.now)


class WorkflowPhase(BaseModel):
    phase_name: str
    description: str
    required_agents: List[AgentType]
    expected_duration_minutes: int
    success_criteria: List[str]
    dependencies: List[str] = []  # previous phase names


class QualityCheck(BaseModel):
    check_id: str
    check_type: str  # content_quality, asset_quality, technical_quality
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None
    performed_by: AgentType
    performed_at: Optional[datetime] = None
