"""
Parallel.ai adapter for the agentic content production system.

This module implements the Crawler interface using the Parallel.ai API for web scraping
and content extraction. Parallel.ai provides fast, scalable web crawling with structured
data extraction capabilities.
"""

import asyncio
import os
import httpx
from typing import Optional, Dict, Any
from datetime import datetime

from .base import Crawler, CrawlOptions, CrawlResult, Reference


class ParallelAIAdapter(Crawler):
    """Parallel.ai implementation of the Crawler interface."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize Parallel.ai adapter.

        Args:
            api_key: Parallel.ai API key. If None, will try to get from environment.
            config: Additional configuration options.
        """
        super().__init__(api_key, config)
        self.api_key = api_key or os.getenv("PARALLEL_API_KEY")
        if not self.api_key:
            raise ValueError("Parallel.ai API key is required. Set PARALLEL_API_KEY environment variable.")

        # Allow configuring base URL, crawl path, health path, and auth scheme
        self.base_url = os.getenv("PARALLEL_API_BASE_URL", "https://api.parallel.ai")
        self.crawl_path = os.getenv("PARALLEL_API_CRAWL_PATH", "/v1/crawl")
        self.health_path = os.getenv("PARALLEL_API_HEALTH_PATH", "/v1/health")
        auth_scheme = os.getenv("PARALLEL_API_AUTH_SCHEME", "Bearer")

        headers = (
            {"Authorization": f"{auth_scheme} {self.api_key}"}
            if auth_scheme.lower() == "bearer"
            else {"X-API-Key": self.api_key}
        )
        self.client = httpx.AsyncClient(timeout=60.0, headers=headers)

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "parallel_ai"

    async def crawl(self, url: str, options: CrawlOptions) -> CrawlResult:
        """Crawl a URL using Parallel.ai API."""
        # Mock mode: avoid external API calls during tests
        if (self.api_key or "").lower().startswith("mock"):
            try:
                resp = await self.client.get(
                    url,
                    headers={"User-Agent": options.user_agent},
                    timeout=options.timeout_s,
                )
                html = resp.text
                # naive title extract
                title = ""
                start = html.lower().find("<title>")
                end = html.lower().find("</title>") if start != -1 else -1
                if start != -1 and end != -1 and end > start:
                    title = html[start + 7 : end].strip()

                return CrawlResult(
                    url=url,
                    title=title,
                    text=html,
                    html=html if options.crawl_scope == "page" else None,
                    images=[],
                    metadata={"provider": "parallel_ai", "mode": "mock"},
                    success=True,
                )
            except Exception as e:
                return CrawlResult(
                    url=url, title="", text="", success=False, error_message=str(e)
                )

        try:
            # Prepare Parallel.ai request payload
            payload = {
                "url": url,
                "render_js": options.js_render,
                "wait_for": 3000 if options.js_render else 0,
                "timeout": options.timeout_s,
                "extract_text": True,
                "extract_html": True,
                "extract_metadata": options.extract_metadata,
                "extract_images": options.extract_images,
                "follow_links": options.depth > 1,
                "max_pages": options.max_pages or 5 if options.depth > 1 else 1,
                "user_agent": options.user_agent
            }

            # Make the API request
            response = await self.client.post(
                f"{self.base_url.rstrip('/')}{self.crawl_path}",
                json=payload
            )

            if response.status_code != 200:
                error_msg = (
                    f"Parallel.ai API error: {response.status_code} - {response.text} "
                    f"(endpoint: {self.base_url.rstrip('/')}{self.crawl_path})"
                )
                return CrawlResult(
                    url=url,
                    title="",
                    text="",
                    success=False,
                    error_message=error_msg
                )

            data = response.json()

            # Extract data from Parallel.ai response
            page_data = data.get("result", {})

            # Extract text content
            text_content = page_data.get("text", "")

            # Extract HTML content
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
            if "structured_data" in page_data:
                json_ld = page_data["structured_data"]

            # Extract publication date and author if available
            published_at = None
            author = None
            if metadata:
                if "published_time" in metadata:
                    try:
                        published_at = datetime.fromisoformat(metadata["published_time"].replace('Z', '+00:00'))
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
                "provider": "parallel_ai",
                "elapsed_ms": data.get("processing_time_ms", 0),
                "cost_estimate": self.estimate_cost(url, options),
                "crawl_depth": options.depth,
                "js_rendered": options.js_render
            }

            return CrawlResult(
                url=url,
                canonical_url=metadata.get("canonical_url"),
                title=title,
                text=text_content,
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
            error_msg = f"Parallel.ai crawl failed: {str(e)}"
            return CrawlResult(
                url=url,
                title="",
                text="",
                success=False,
                error_message=error_msg
            )

    def estimate_cost(self, url: str, options: CrawlOptions) -> Optional[float]:
        """Estimate the cost of crawling with Parallel.ai.

        Parallel.ai pricing is typically $0.05-$0.15 per page depending on complexity.
        """
        # Basic estimation - can be refined based on actual pricing
        base_cost_per_page = 0.08
        pages_to_crawl = 1

        if options.depth > 1:
            pages_to_crawl = min(options.max_pages or 5, 10)  # Estimate based on depth

        # Add premium for JS rendering
        if options.js_render:
            base_cost_per_page *= 1.5

        return base_cost_per_page * pages_to_crawl

    async def health_check(self) -> bool:
        """Check if Parallel.ai API is accessible."""
        if (self.api_key or "").lower().startswith("mock"):
            return True

        try:
            response = await self.client.get(f"{self.base_url.rstrip('/')}{self.health_path}")
            return response.status_code == 200
        except:
            return False