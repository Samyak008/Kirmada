"""
Base crawler abstraction for the agentic content production system.

This module defines the common interface and data models for all crawler implementations,
allowing for easy swapping between different crawling backends (Firecrawl, Parallel.ai, etc.)
while maintaining consistent data structures and error handling.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class Reference(BaseModel):
    """Represents a reference or citation found during crawling."""
    title: str
    url: str
    type: str = "article"  # e.g., article, video, image, dataset, documentation, other


class CrawlOptions(BaseModel):
    """Configuration options for crawling operations."""
    depth: int = 1
    js_render: bool = False
    timeout_s: int = 30
    crawl_scope: Literal["page", "site"] = "page"
    extract_images: bool = False
    extract_metadata: bool = True
    max_pages: Optional[int] = None
    # Default desktop UA to avoid basic blocks
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )


class CrawlResult(BaseModel):
    """Structured result from a crawling operation."""
    url: str
    canonical_url: Optional[str] = None
    title: str = ""
    text: str = ""
    html: Optional[str] = None
    json_ld: Optional[Dict[str, Any]] = None
    images: List[str] = Field(default_factory=list)
    published_at: Optional[datetime] = None
    author: Optional[str] = None
    grounding: Optional[List[Reference]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = False
    error_message: Optional[str] = None


class Crawler(ABC):
    """Abstract base class for all crawler implementations."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the crawler with API key and configuration."""
        self.api_key = api_key
        self.config = config or {}

    @abstractmethod
    async def crawl(self, url: str, options: CrawlOptions) -> CrawlResult:
        """Crawl a URL and return structured results."""
        raise NotImplementedError

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the crawler provider."""
        raise NotImplementedError

    def estimate_cost(self, url: str, options: CrawlOptions) -> Optional[float]:
        """Optionally estimate cost for a crawl."""
        return None

    async def health_check(self) -> bool:
        """Optionally implement provider health check."""
        return True