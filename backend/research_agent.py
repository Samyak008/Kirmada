from __future__ import annotations
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import yaml
import json
from datetime import datetime
import logging
from models import (
    AgentState, AgentType, TaskStatus, WorkflowPhase, 
    AgentMessage, SupervisorDecision, QualityCheck, Task
)
from tools import get_tools_for_agent, validate_tool_input, ToolResult
from specialized_agent import SpecializedAgent
from typing import TypedDict


class WorkflowState(TypedDict):
    """LangGraph state for the agentic workflow"""
    agent_state: AgentState
    messages: List[BaseMessage]
    current_agent: Optional[AgentType]
    next_agent: Optional[AgentType]
    task_results: Dict[str, Any]
    errors: List[Dict[str, Any]]
    workflow_completed: bool


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResearchAgent(SpecializedAgent):
    """Research agent implementation"""
    
    def __init__(self, config_path: str = "agent_prompts.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        super().__init__(AgentType.RESEARCH, config)
        # Initialize search and crawl services
        self.search_service = self._init_search_service()
        self.crawl_service = self._init_crawl_service()

    def _init_search_service(self):
        """Initialize the intelligent search service"""
        # TODO: Install langchain packages
        # from langchain_community.tools.tavily_search import TavilySearchResults
        # from langchain_openai import ChatOpenAI
        import os

        # Placeholder for OpenAI LLM - will be replaced when packages are installed
        class MockChatOpenAI:
            def invoke(self, messages):
                class MockResponse:
                    content = '{"intent_type": "general_news", "key_entities": ["AI"], "time_sensitivity": "recent", "content_preferences": ["analysis"], "priority_sources": ["tech publications"], "search_focus": "Find recent AI content"}'
                return MockResponse()

        # Placeholder for Tavily search - will be replaced when packages is installed
        class MockTavilySearch:
            def invoke(self, query_dict):
                # Mock search results
                return {
                    "results": [
                        {
                            "url": "https://example.com/ai-article",
                            "title": "Latest AI Developments",
                            "content": "Recent advances in artificial intelligence...",
                            "domain": "example.com"
                        }
                    ]
                }

        return {
            'llm': MockChatOpenAI(),
            'search_provider': MockTavilySearch(),
            'firecrawl_api_key': os.getenv("FIRECRAWL_API_KEY")
        }

    def _init_crawl_service(self):
        """Initialize crawl service components"""
        import os
        from crawlers.firecrawl_adapter import FirecrawlAdapter
        from crawlers.parallelai_adapter import ParallelAIAdapter

        return {
            'firecrawl_adapter': FirecrawlAdapter(),
            'parallelai_adapter': ParallelAIAdapter(),
            'firecrawl_api_key': os.getenv("FIRECRAWL_API_KEY"),
            'parallel_api_key': os.getenv("PARALLEL_API_KEY")
        }

    async def _intelligent_search(self, query: str, max_results: int = 8) -> Dict[str, Any]:
        """Execute intelligent search with contextual understanding"""
        try:
            logger.info(f"Starting intelligent search for query: {query}")
            # Analyze query intent
            intent_analysis = await self._analyze_query_intent(query)

            # Generate optimized search queries
            search_queries = await self._generate_search_queries(query, intent_analysis)
            logger.info(f"Generated {len(search_queries)} search queries: {search_queries}")

            # Execute searches
            all_results = []
            for search_query in search_queries:
                try:
                    logger.info(f"Executing search for query: {search_query}")
                    results = await asyncio.to_thread(
                        self.search_service['search_provider'].invoke,
                        {"query": search_query}
                    )
                    extracted_results = self._extract_structured_results(results)
                    all_results.extend(extracted_results)
                    logger.info(f"Search for '{search_query}' returned {len(extracted_results)} results")
                except Exception as e:
                    logger.error(f"Search failed for query '{search_query}': {e}")
                    continue

            logger.info(f"Total results collected: {len(all_results)}")
            
            # Filter and rank results
            filtered_results = await self._intelligent_filter_and_rank(query, intent_analysis, all_results)
            logger.info(f"Filtered and ranked results: {len(filtered_results)}")

            # Ensure we return the correct number of results
            final_results = filtered_results[:max_results] if isinstance(filtered_results, list) else []
            
            result = {
                "success": True,
                "query": query,
                "results": final_results,
                "total_found": len(all_results),
                "filtered_count": len(final_results),
                "intent_analysis": intent_analysis
            }
            
            logger.info(f"Search completed successfully. Returning {len(final_results)} results")
            return result

        except Exception as e:
            logger.error(f"Error during intelligent search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }

    async def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Use LLM to analyze user query and determine search strategy"""
        logger.info(f"Analyzing query intent for: {query}")
        prompt = f"""Analyze this search query and determine the user's intent: "{query}"

Your task is to understand what the user is looking for and provide a search strategy.

Consider:
1. Is this asking for latest/general tech news or something specific?
2. Are there specific companies, products, or technologies mentioned?
3. What time sensitivity does this have? (breaking news, recent developments, etc.)
4. What type of content would be most valuable? (announcements, analysis, tutorials, etc.)

Return a JSON object with this structure:
{{
  "intent_type": "general_news|specific_topic|company_news|product_news|breaking_news",
  "key_entities": ["list", "of", "important", "keywords"],
  "time_sensitivity": "breaking|recent|any",
  "content_preferences": ["announcements", "analysis", " tutorials", "reviews"],
  "priority_sources": ["list", "of", "preferred", "source", "types"],
  "search_focus": "brief description of what to focus on"
}}

Return ONLY the JSON, no additional text."""

        try:
            response = await asyncio.to_thread(
                self.search_service['llm'].invoke,
                [HumanMessage(content=prompt)]
            )

            result = json.loads(response.content.strip())
            logger.info(f"Query intent analysis complete: {result.get('intent_type', 'unknown')}")
            return result
        except Exception as e:
            logger.error(f"Error during query intent analysis: {str(e)}")
            # Fallback analysis
            return {
                "intent_type": "general_news",
                "key_entities": [query],
                "time_sensitivity": "recent",
                "content_preferences": ["analysis", "announcements"],
                "priority_sources": ["tech publications"],
                "search_focus": f"Find recent content about {query}"
            }

    async def _generate_search_queries(self, original_query: str, strategy: Dict[str, Any]) -> List[str]:
        """Generate multiple optimized search queries based on strategy"""
        logger.info(f"Generating search queries for: {original_query}")
        prompt = f"""Based on this search strategy, generate 3-4 optimized search queries that will find high-quality, standalone tech articles.

Original query: "{original_query}"
Strategy: {json.dumps(strategy, indent=2)}

Requirements for search queries:
1. Focus on finding standalone articles, not aggregated content
2. Prioritize official sources for companies/products mentioned
3. Include recency indicators if time-sensitive
4. Avoid terms that lead to roundups, newsletters, or category pages

Generate queries that will find:
- Official announcements and press releases
- Specific news articles about individual topics
- Recent developments from authoritative sources

Return a JSON array of 3-4 search query strings:
["query1", "query2", "query3", "query4"]

Return ONLY the JSON array, no additional text."""

        try:
            response = await asyncio.to_thread(
                self.search_service['llm'].invoke,
                [HumanMessage(content=prompt)]
            )

            queries = json.loads(response.content.strip())
            logger.info(f"Generated {len(queries)} search queries: {queries}")
            return queries
        except Exception as e:
            logger.error(f"Error during search query generation: {str(e)}")
            # Fallback queries
            return [
                f"{original_query} latest news",
                f"{original_query} recent developments",
                f"{original_query} analysis",
                f"{original_query} official announcement"
            ]

    def _extract_structured_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        """Extract structured data from search results"""
        logger.info("Extracting structured results from raw data")
        structured_results = []

        try:
            if hasattr(raw_results, 'get'):
                results = raw_results.get('results', [])
            else:
                results = []

            for result in results:
                structured_results.append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "domain": result.get("domain", ""),
                    "published_date": result.get("published_date"),
                    "score": 0.5  # Default score
                })

        except Exception as e:
            logger.error(f"Error extracting structured results: {e}")

        logger.info(f"Extracted {len(structured_results)} structured results")
        return structured_results

    async def _intelligent_filter_and_rank(self, query: str, strategy: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to intelligently filter and rank results"""
        logger.info(f"Intelligently filtering and ranking {len(results)} results")
        if not results:
            return []

        # Limit results to analyze
        results_to_analyze = results[:15]

        # Create analysis prompt
        results_text = ""
        for i, result in enumerate(results_to_analyze, 1):
            results_text += f"{i}. {result['title']}\n   URL: {result['url']}\n   Content: {result['content'][:200]}...\n\n"

        prompt = f"""Analyze these search results for the query: "{query}"

Search Strategy: {json.dumps(strategy, indent=2)}

Results to analyze:
{results_text}

Your task is to filter and rank these results to find the BEST standalone tech articles that match the user's intent.

PRIORITIZE results that are:
1. Standalone articles about specific topics (not category pages, author pages, or aggregated content)
2. From official sources when relevant (company blogs, press releases, official announcements)
3. Recent and newsworthy content
4. Directly relevant to the user's query
5. From authoritative tech publications

EXCLUDE results that are:
- Newsletter articles or daily/weekly digests
- Author profile pages or category landing pages
- Aggregated roundup content
- Forum posts or community discussions
- Generic section pages (like /tech/, /science/, /category/)

Return a JSON array of the TOP 8 results, ranked from best to worst. Include the complete result data:

[
  {{
    "url": "full_url_here",
    "title": "article_title_here",
    "content": "content_snippet_here",
    "score": 0.9,
    "domain": "domain.com",
    "published_date": "date_if_available",
    "relevance_reason": "brief explanation of why this is relevant and high-quality"
  }}
]

Return ONLY the JSON array with no additional text."""

        try:
            response = await asyncio.to_thread(
                self.search_service['llm'].invoke,
                [HumanMessage(content=prompt)]
            )

            ranked_results = json.loads(response.content.strip())
            logger.info(f"Successfully ranked and filtered results. Got {len(ranked_results)} top results.")
            return ranked_results

        except Exception as e:
            logger.error(f"Error during intelligent filtering and ranking: {str(e)}")
            # Fallback: return top results with default scores
            return results[:8]

    async def _crawl_url(self, url: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Crawl a URL using available adapters with fallback"""
        logger.info(f"Crawling URL: {url}")
        from crawlers.base import CrawlOptions

        crawl_options = CrawlOptions(
            depth=options.get("depth", 1) if options else 1,
            js_render=options.get("js_render", True) if options else True,
            timeout_s=options.get("timeout_s", 30) if options else 30,
            crawl_scope=options.get("crawl_scope", "page") if options else "page",
            extract_images=options.get("extract_images", True) if options else True,
            extract_metadata=options.get("extract_metadata", True) if options else True
        )

        # Try Firecrawl first
        try:
            logger.info("Attempting crawl with Firecrawl adapter")
            result = await self.crawl_service['firecrawl_adapter'].crawl(url, crawl_options)
            if result.success:
                logger.info(f"Successfully crawled {url} with Firecrawl")
                return {
                    "success": True,
                    "url": url,
                    "title": result.title,
                    "content": result.text,
                    "html": result.html,
                    "images": result.images,
                    "metadata": result.metadata,
                    "provider": "firecrawl"
                }
        except Exception as e:
            logger.error(f"Firecrawl failed for {url}: {e}")

        # Fallback to Parallel.ai
        try:
            logger.info("Attempting crawl with Parallel.ai adapter")
            result = await self.crawl_service['parallelai_adapter'].crawl(url, crawl_options)
            if result.success:
                logger.info(f"Successfully crawled {url} with Parallel.ai")
                return {
                    "success": True,
                    "url": url,
                    "title": result.title,
                    "content": result.text,
                    "html": result.html,
                    "images": result.images,
                    "metadata": result.metadata,
                    "provider": "parallel_ai"
                }
        except Exception as e:
            logger.error(f"Parallel.ai failed for {url}: {e}")

        # Final fallback: basic HTML parsing
        logger.info("Using enhanced HTML parsing as final fallback")
        return await self._enhanced_html_parsing(url)

    async def _enhanced_html_parsing(self, url: str) -> Dict[str, Any]:
        """Enhanced HTML parsing as final fallback"""
        logger.info(f"Using enhanced HTML parsing for: {url}")
        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract title
                title = ""
                title_selectors = ['h1', 'title', '.headline', '.article-title']
                for selector in title_selectors:
                    element = soup.select_one(selector)
                    if element:
                        title = element.get_text().strip()
                        break

                # Extract content
                content_selectors = ['article', '.content', '.post-content', '.entry-content', 'main']
                content = ""
                for selector in content_selectors:
                    element = soup.select_one(selector)
                    if element:
                        content = element.get_text().strip()
                        break

                if not content:
                    # Fallback to all paragraphs
                    paragraphs = soup.find_all('p')
                    content = ' '.join([p.get_text().strip() for p in paragraphs[:20]])  # Limit to first 20 paragraphs

                logger.info(f"Successfully parsed HTML for {url} - title: {title[:50]}... content length: {len(content)} chars")
                return {
                    "success": True,
                    "url": url,
                    "title": title or "No title found",
                    "content": content,
                    "html": response.text,
                    "images": [],
                    "metadata": {"provider": "html_parsing"},
                    "provider": "html_parsing"
                }

        except Exception as e:
            logger.error(f"Error during enhanced HTML parsing for {url}: {str(e)}")
            return {
                "success": False,
                "url": url,
                "title": "",
                "content": "",
                "error": str(e),
                "provider": "html_parsing"
            }
            
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute research tasks with real search and crawling using existing infrastructure"""
        logger.info(f"Executing research task: {task.description}. Current phase: {agent_state.current_phase}")
        import asyncio
        
        async def execute_async():
            try:
                # Accept either description or legacy task_type
                task_desc_raw = getattr(task, "description", None) or getattr(task, "task_type", "")
                task_desc = str(task_desc_raw).lower()
                
                if "search" in task_desc or agent_state.current_phase == "search":
                    logger.info("Starting search phase")
                    # Use the actual search service
                    query = agent_state.content_request or (getattr(task, "description", None) or str(task_desc_raw))
                    logger.info(f"Initiating search for query: {query}")
                    search_results = await self._intelligent_search(query, max_results=8)
                    
                    if not search_results.get("success"):
                        logger.error(f"Search failed: {search_results.get('error', 'Search failed')}")
                        return {
                            "success": False,
                            "error": search_results.get("error", "Search failed"),
                            "search_completed": False
                        }
                    
                    # Initialize research_data if needed
                    if not getattr(agent_state, "research_data", None):
                        logger.info("Initializing research data structure")
                        from models import ResearchData
                        agent_state.research_data = ResearchData(
                            topic=query,
                            sources=[],
                            key_findings=[],
                            trends=[],
                            statistics={},
                            research_notes=""
                        )
                    
                    # Convert search results to Source objects
                    from models import Source
                    sources = []
                    for result in search_results.get("results", []):
                        source = Source(
                            url=result.get("url", ""),
                            title=result.get("title", ""),
                            content=result.get("content", ""),
                            relevance_score=result.get("score", 0.5),
                            tags=["tech", "news"]
                        )
                        sources.append(source)
                    
                    agent_state.research_data.sources = sources
                    logger.info(f"Search completed successfully. Found {len(sources)} sources.")
                    
                    return {
                        "success": True,
                        "search_completed": True,
                        "sources_found": len(sources),
                        "urls": [s.url for s in sources if s.url],
                        "next_phase": "crawl"
                    }
                
                elif "crawl" in task_desc or agent_state.current_phase == "crawl":
                    logger.info("Starting crawl phase")
                    # Use your existing crawl infrastructure
                    if not agent_state.research_data or not agent_state.research_data.sources:
                        logger.error("No search results to crawl")
                        return {
                            "success": False,
                            "error": "No search results to crawl",
                            "crawl_completed": False
                        }
                    
                    # Get URLs from search results (limit to 3 for testing)
                    urls = [source.url for source in agent_state.research_data.sources[:3] if source.url]
                    logger.info(f"Found {len(urls)} URLs to crawl")
                    
                    if not urls:
                        logger.error("No valid URLs found in search results")
                        return {
                            "success": False,
                            "error": "No valid URLs found in search results",
                            "crawl_completed": False
                        }
                    
                    crawled_articles = []
                    crawl_errors = []
                    
                    # Use your existing _crawl_url method which handles Firecrawl/ParallelAI fallback
                    for url in urls:
                        try:
                            logger.info(f"Crawling URL: {url}")
                            crawl_result = await self._crawl_url(url)
                            
                            if crawl_result.get("success"):
                                # Update the corresponding Source object with crawled content
                                for source in agent_state.research_data.sources:
                                    if source.url == url:
                                        source.content = crawl_result.get("content", "")
                                        source.title = crawl_result.get("title", source.title)
                                        if crawl_result.get("content"):
                                            # Extract summary from first 200 chars
                                            source.summary = crawl_result["content"][:200] + "..."
                                        break
                                
                                crawled_articles.append(crawl_result)
                                logger.info(f"Successfully crawled: {url} ({len(crawl_result.get('content', ''))} chars)")
                            else:
                                error_msg = crawl_result.get("error", "Unknown crawl error")
                                crawl_errors.append({"url": url, "error": error_msg})
                                logger.error(f"Failed to crawl {url}: {error_msg}")
                                
                        except Exception as e:
                            error_msg = f"Exception while crawling {url}: {str(e)}"
                            crawl_errors.append({"url": url, "error": error_msg})
                            logger.error(error_msg)
                            continue
                    
                    # Generate key findings from crawled content
                    if crawled_articles:
                        key_findings = []
                        for article in crawled_articles[:3]:
                            content = article.get("content", "")
                            if content:
                                # Extract first sentence as a key finding
                                sentences = content.split('. ')
                                if sentences:
                                    key_findings.append(sentences[0] + ".")
                        
                        agent_state.research_data.key_findings = key_findings[:5]
                    
                    logger.info(f"Crawl completed. Articles crawled: {len(crawled_articles)}, Errors: {len(crawl_errors)}")
                    return {
                        "success": True,
                        "crawl_completed": True,
                        "articles_crawled": len(crawled_articles),
                        "crawl_errors": len(crawl_errors),
                        "total_content_length": sum(len(article.get("content", "")) for article in crawled_articles),
                        "crawled_content": crawled_articles,
                        "errors": crawl_errors,
                        "next_phase": "store_article"
                    }
                
                else:
                    logger.error(f"Unknown research task: {task.description}")
                    return {
                        "success": False,
                        "error": f"Unknown research task: {task.description}"
                    }
                    
            except Exception as e:
                logger.error(f"Error during research execution: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "research_completed": False
                }
        
        # Execute async function
        try:
            try:
                loop = asyncio.get_running_loop()
                is_running = True
            except RuntimeError:
                is_running = False

            if is_running:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, execute_async())
                    result = future.result()
                    logger.info(f"Research task execution completed with success: {result.get('success', False)}")
                    return result
            else:
                result = asyncio.run(execute_async())
                logger.info(f"Research task execution completed with success: {result.get('success', False)}")
                return result
        except Exception as e:
            logger.error(f"Critical error during research task execution: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with research results"""
        logger.info(f"Updating agent state after research. Result success: {result.get('success', False)}")
        
        # Create mock research data
        from models import ResearchData, Source
        
        sources = [
            Source(
                url="https://example.com/article1",
                title="Sample Article 1",
                content="Sample content from article 1",
                relevance_score=0.9,
                tags=["tech", "ai"]
            )
        ]
        
        research_data = ResearchData(
            topic=agent_state.content_request,
            sources=sources,
            key_findings=result.get("key_findings", []),
            research_notes="Research completed successfully"
        )
        
        agent_state.research_data = research_data
        
        # Update current phase to advance the workflow
        if agent_state.current_phase == "search":
            agent_state.current_phase = "crawl"
            logger.info("Advanced phase from search to crawl")
        elif agent_state.current_phase == "crawl":
            agent_state.current_phase = "store_article"
            logger.info("Advanced phase from crawl to store_article")