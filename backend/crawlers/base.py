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
    title: str = Field(..., description="Title of the referenced content")
    url: Optional[str] = Field(None, description="URL of the reference")
    type: Literal["article", "video", "image", "document", "other"] = Field("article", description="Type of reference")
    relevance_score: Optional[float] = Field(None, description="Relevance score (0-1)")


class CrawlOptions(BaseModel):
    """Configuration options for crawling operations."""
    depth: int = Field(1, description="Crawling depth (1 = single page, 2+ = follow links)")
    js_render: bool = Field(True, description="Whether to render JavaScript")
    user_agent: str = Field("Mozilla/5.0 (compatible; ResearchAgent/1.0)", description="User agent string")
    timeout_s: int = Field(30, description="Request timeout in seconds")
    crawl_scope: Literal["page", "site"] = Field("page", description="Scope of crawling")
    max_pages: Optional[int] = Field(None, description="Maximum pages to crawl")
    follow_external_links: bool = Field(False, description="Whether to follow external links")
    extract_images: bool = Field(True, description="Whether to extract image URLs")
    extract_metadata: bool = Field(True, description="Whether to extract metadata")


class CrawlResult(BaseModel):
    """Structured result from a crawling operation."""
    url: str = Field(..., description="Original URL that was crawled")
    canonical_url: Optional[str] = Field(None, description="Canonical URL if different")
    title: str = Field(..., description="Page title")
    text: str = Field(..., description="Extracted text content")
    html: Optional[str] = Field(None, description="Raw HTML content")
    json_ld: Optional[Dict[str, Any]] = Field(None, description="Extracted JSON-LD structured data")
    images: List[str] = Field(default_factory=list, description="List of image URLs found")
    published_at: Optional[datetime] = Field(None, description="Publication date if available")
    author: Optional[str] = Field(None, description="Author name if available")
    grounding: Optional[List[Reference]] = Field(None, description="List of references/grounding data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    success: bool = Field(True, description="Whether the crawl was successful")
    error_message: Optional[str] = Field(None, description="Error message if crawl failed")


class Crawler(ABC):
    """Abstract base class for all crawler implementations."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the crawler with API key and configuration."""
        self.api_key = api_key
        self.config = config or {}

    @abstractmethod
    async def crawl(self, url: str, options: CrawlOptions) -> CrawlResult:
        """Crawl a URL and return structured results.

        Args:
            url: The URL to crawl
            options: Crawling configuration options

        Returns:
            CrawlResult: Structured crawling results

        Raises:
            Exception: If crawling fails
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the crawler provider."""
        pass

    def estimate_cost(self, url: str, options: CrawlOptions) -> Optional[float]:
        """Estimate the cost of crawling a URL (if applicable).

        Args:
            url: The URL to estimate cost for
            options: Crawling options that might affect cost

        Returns:
            Cost estimate in USD, or None if not applicable
        """
        return None

    async def health_check(self) -> bool:
        """Check if the crawler service is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        return True</content>
<parameter name="filePath">c:\Users\djsma\Downloads\Github_Desktop\prism\new_code\backend\crawlers\base.py