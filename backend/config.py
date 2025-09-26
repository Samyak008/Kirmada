"""
Configuration management for the Agentic Content Production System
"""
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    # Preferred LLM model (set in .env: OPENAI_MODEL=gpt-4o or gpt-4o-mini)
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    
    # Supabase Configuration
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY")
    
    # Google Drive Configuration
    google_credentials_file: str = Field(..., env="GOOGLE_CREDENTIALS_FILE")
    google_drive_folder_id: str = Field(..., env="GOOGLE_DRIVE_FOLDER_ID")
    
    # Notion Configuration
    notion_token: str = Field(..., env="NOTION_TOKEN")
    notion_database_id: str = Field(..., env="NOTION_DATABASE_ID")
    
    # System Configuration
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    task_timeout: int = Field(default=300, env="TASK_TIMEOUT")  # 5 minutes
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Workflow Configuration
    enable_quality_checks: bool = Field(default=True, env="ENABLE_QUALITY_CHECKS")
    auto_retry_failed_tasks: bool = Field(default=True, env="AUTO_RETRY_FAILED_TASKS")
    parallel_task_execution: bool = Field(default=False, env="PARALLEL_TASK_EXECUTION")
    
    # Content Generation Settings
    default_content_duration: int = Field(default=300, env="DEFAULT_CONTENT_DURATION")  # 5 minutes
    max_research_sources: int = Field(default=10, env="MAX_RESEARCH_SOURCES")
    image_quality: str = Field(default="high", env="IMAGE_QUALITY")
    voice_speed: float = Field(default=1.0, env="VOICE_SPEED")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class AgentConfig:
    """Agent-specific configuration"""
    
    def __init__(self, config_data: Dict[str, Any]):
        self.data = config_data
    
    def get_agent_prompt(self, agent_type: str) -> str:
        """Get system prompt for an agent"""
        return self.data["agents"][agent_type]["system_prompt"]
    
    def get_agent_capabilities(self, agent_type: str) -> list:
        """Get capabilities for an agent"""
        return self.data["agents"][agent_type]["capabilities"]
    
    def get_workflow_phases(self) -> Dict[str, Any]:
        """Get workflow phase configuration"""
        return self.data["workflow_phases"]
    
    def get_quality_standards(self) -> Dict[str, Any]:
        """Get quality standards configuration"""
        return self.data["quality_standards"]


def load_agent_config(config_path: str = "agent_prompts.yaml") -> AgentConfig:
    """Load agent configuration from YAML file"""
    import yaml
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return AgentConfig(config_data)


# Global settings instance
settings = Settings()

# Global agent config instance
agent_config = load_agent_config()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def get_agent_config() -> AgentConfig:
    """Get agent configuration"""
    return agent_config
