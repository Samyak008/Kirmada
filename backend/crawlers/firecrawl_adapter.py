"""
Firecrawl adapter for the agentic content production system.

This module implements the Crawler interface using the Firecrawl API for web scraping
and content extraction. Firecrawl provides structured data extraction with JavaScript
rendering support and anti-bot bypass capabilities.
"""

import asyncio
import os
from typing import Optional, Dict, Any
import httpx
from datetime import datetime

from .base import Crawler, CrawlOptions, CrawlResult, Reference


class FirecrawlAdapter(Crawler):
    """Firecrawl implementation of the Crawler interface."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize Firecrawl adapter.

        Args:
            api_key: Firecrawl API key. If None, will try to get from environment.
            config: Additional configuration options.
        """
        super().__init__(api_key, config)
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("Firecrawl API key is required. Set FIRECRAWL_API_KEY environment variable.")

        self.base_url = "https://api.firecrawl.dev"
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "firecrawl"

    async def crawl(self, url: str, options: CrawlOptions) -> CrawlResult:
        """Crawl a URL using Firecrawl API.

        Args:
            url: The URL to crawl
            options: Crawling configuration options

        Returns:
            CrawlResult: Structured crawling results
        """
        try:
            # Prepare Firecrawl request payload
            payload = {
                "url": url,
                "formats": ["markdown", "html"],
                "onlyMainContent": True,
                "includeTags": ["title", "meta", "description"] if options.extract_metadata else [],
                "waitFor": 3000 if options.js_render else 0,  # Wait for JS if enabled
                "mobile": False,
                "skipTlsVerification": False,
                "timeout": options.timeout_s * 1000,  # Convert to milliseconds
            }

            # Add depth crawling if requested
            if options.depth > 1:
                payload["limit"] = options.max_pages or 10
                payload["scrapeOptions"] = {
                    "formats": ["markdown"],
                    "onlyMainContent": True
                }

            # Make the API request
            response = await self.client.post(
                f"{self.base_url}/v1/scrape",
                json=payload
            )

            if response.status_code != 200:
                error_msg = f"Firecrawl API error: {response.status_code} - {response.text}"
                return CrawlResult(
                    url=url,
                    title="",
                    text="",
                    success=False,
                    error_message=error_msg
                )

            data = response.json()

            # Extract data from Firecrawl response
            page_data = data.get("data", {})

            # Parse markdown content
            markdown_content = page_data.get("markdown", "")
            html_content = page_data.get("html", "")

            # Extract title
            title = page_data.get("title", "")
            if not title and html_content:
                # Fallback title extraction from HTML
                import re
                title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()

            # Extract metadata
            metadata = page_data.get("metadata", {})

            # Extract images if requested
            images = []
            if options.extract_images and "images" in page_data:
                images = page_data["images"]

            # Extract structured data
            json_ld = None
            if "structuredData" in page_data:
                json_ld = page_data["structuredData"]

            # Extract publication date and author if available
            published_at = None
            author = None
            if metadata:
                if "publishedTime" in metadata:
                    try:
                        published_at = datetime.fromisoformat(metadata["publishedTime"].replace('Z', '+00:00'))
                    except:
                        pass
                if "author" in metadata:
                    author = metadata["author"]

            # Create grounding references if available
            grounding = None
            if "links" in page_data:
                grounding = []
                for link in page_data["links"][:5]:  # Limit to top 5 references
                    grounding.append(Reference(
                        title=link.get("title", link.get("text", "")),
                        url=link.get("href", ""),
                        type="article"
                    ))

            # Add provider metadata
            result_metadata = {
                "provider": "firecrawl",
                "elapsed_ms": data.get("processingTimeMs", 0),
                "cost_estimate": self.estimate_cost(url, options),
                "crawl_depth": options.depth,
                "js_rendered": options.js_render
            }

            return CrawlResult(
                url=url,
                canonical_url=metadata.get("canonicalUrl"),
                title=title,
                text=markdown_content,
                html=html_content if options.crawl_scope == "page" else None,
                json_ld=json_ld,
                images=images,
                published_at=published_at,
                author=author,
                grounding=grounding,
                metadata=result_metadata,
                success=True
            )

        except Exception as e:
            error_msg = f"Firecrawl crawl failed: {str(e)}"
            return CrawlResult(
                url=url,
                title="",
                text="",
                success=False,
                error_message=error_msg
            )

    def estimate_cost(self, url: str, options: CrawlOptions) -> Optional[float]:
        """Estimate the cost of crawling with Firecrawl.

        Firecrawl pricing is typically $0.10 per page for basic scraping.
        """
        # Basic estimation - can be refined based on actual pricing
        base_cost_per_page = 0.10
        pages_to_crawl = 1

        if options.depth > 1:
            pages_to_crawl = min(options.max_pages or 5, 10)  # Estimate based on depth

        return base_cost_per_page * pages_to_crawl

    async def health_check(self) -> bool:
        """Check if Firecrawl API is accessible."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/health")
            return response.status_code == 200
        except:
            return False
