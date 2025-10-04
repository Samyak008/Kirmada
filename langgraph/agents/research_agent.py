from __future__ import annotations
import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import yaml
import json
from datetime import datetime
import logging
from enum import Enum
from typing import TypedDict
from langchain_openai import ChatOpenAI
import os

from .specialized_agent import SpecializedAgent, AgentType, Task, TaskStatus, LangGraphState

# Import the existing tools
from .search_agent import search_trending_tech_news, extract_article_urls
from .crawl_agent import crawl_article_content
from .supabase_agent import store_article_content_sync_wrapped

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchAgent(SpecializedAgent):
    """Research agent implementation that works with existing tools"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.RESEARCH, config)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=4000,
            temperature=0.1
        )
        
        # Register tools that will be used by this agent
        self.tools = [search_trending_tech_news, extract_article_urls, crawl_article_content, store_article_content_sync_wrapped]
    
    def _execute_task(self, task: Dict[str, Any], state: LangGraphState) -> Dict[str, Any]:
        """Execute research tasks with real search and crawling"""
        logger.info(f"Executing research task: {task.get('description', 'Unknown task')}. Current phase: {state['workflow_phase']}")
        
        try:
            # Accept either description or legacy task_type
            task_desc_raw = task.get("description", "") or task.get("task_type", "")
            task_desc = str(task_desc_raw).lower()
            
            if "search" in task_desc or state['workflow_phase'] == "search":
                logger.info("Starting search phase")
                # Get search queries from content request
                query = state['user_request'] or str(task_desc_raw)
                logger.info(f"Initiating search for query: {query}")
                
                # Use the existing search tool
                search_results = asyncio.run(search_trending_tech_news.ainvoke({"query": query}))
                
                # Parse the search results and extract URLs
                import re
                urls = re.findall(r'https?://[^\s\n,\'\"]+', search_results)
                
                # Initialize research_data if needed
                if 'research_data' not in state:
                    state['research_data'] = {
                        'topic': query,
                        'sources': [],
                        'key_findings': [],
                        'trends': [],
                        'statistics': {},
                        'research_notes': ""
                    }
                
                # Update research_data with sources
                sources = []
                for url in urls:
                    sources.append({
                        'url': url,
                        'title': f"Article about {query}",
                        'content': "",
                        'relevance_score': 0.8,
                        'tags': [url.split('/')[2]]
                    })
                
                state['research_data']['sources'] = sources
                logger.info(f"Search completed successfully. Found {len(sources)} sources.")
                
                return {
                    "success": True,
                    "search_completed": True,
                    "sources_found": len(sources),
                    "urls": [s['url'] for s in sources if s.get('url')],
                    "next_phase": "crawl"
                }
            
            elif "crawl" in task_desc or state['workflow_phase'] == "crawl":
                logger.info("Starting crawl phase")
                # Use existing crawl infrastructure
                if not state.get('research_data') or not state.get('research_data', {}).get('sources'):
                    logger.error("No search results to crawl")
                    return {
                        "success": False,
                        "error": "No search results to crawl",
                        "crawl_completed": False
                    }
                
                # Get URLs from search results (limit to 3 for efficiency)
                urls = [source['url'] for source in state['research_data']['sources'][:3] if source.get('url')]
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
                
                # Use the existing _crawl_url method (from the backend implementation)
                for url in urls:
                    try:
                        logger.info(f"Crawling URL: {url}")
                        # Use the existing crawl tool
                        crawl_result = asyncio.run(crawl_article_content.ainvoke({"url": url}))
                        
                        if "❌" not in crawl_result and len(crawl_result) > 50:  # Check if successful
                            # Update the corresponding source with crawled content
                            for source in state['research_data']['sources']:
                                if source['url'] == url:
                                    # Extract content from the crawl result
                                    source['content'] = crawl_result
                                    source['title'] = f"Title from {url.split('/')[2]}"
                                    if crawl_result:
                                        # Extract summary from first 150 chars
                                        source['summary'] = crawl_result[:150] + "..." if len(crawl_result) > 150 else crawl_result
                                    break
                            
                            crawled_articles.append({
                                'url': url,
                                'content_length': len(crawl_result),
                                'success': True
                            })
                            logger.info(f"Successfully crawled: {url} ({len(crawl_result)} chars)")
                        else:
                            crawl_errors.append({"url": url, "error": "Crawl failed or returned minimal content"})
                            logger.error(f"Failed to crawl {url}: Returned minimal content")
                            
                    except Exception as e:
                        error_msg = f"Exception while crawling {url}: {str(e)}"
                        crawl_errors.append({"url": url, "error": error_msg})
                        logger.error(error_msg)
                        continue
                
                logger.info(f"Crawl completed. Articles crawled: {len(crawled_articles)}, Errors: {len(crawl_errors)}")
                return {
                    "success": True,
                    "crawl_completed": True,
                    "articles_crawled": len(crawled_articles),
                    "crawl_errors": len(crawl_errors),
                    "total_content_length": sum(len(article.get('content', '')) for article in crawled_articles),
                    "crawled_content": crawled_articles,
                    "errors": crawl_errors,
                    "next_phase": "store_article"
                }
            
            else:
                logger.error(f"Unknown research task: {task.get('description', 'Unknown task')}")
                return {
                    "success": False,
                    "error": f"Unknown research task: {task.get('description', 'Unknown task')}"
                }
                
        except Exception as e:
            logger.error(f"Error during research execution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "research_completed": False
            }

    def _update_agent_state(self, state: LangGraphState, result: Dict[str, Any]) -> None:
        """Update agent state with research results"""
        logger.info(f"Updating agent state after research. Result success: {result.get('success', False)}")
        
        # Update current phase to advance the workflow
        if state['workflow_phase'] == "search":
            state['workflow_phase'] = "crawl"
            logger.info("Advanced phase from search to crawl")
        elif state['workflow_phase'] == "crawl":
            state['workflow_phase'] = "store_article"
            logger.info("Advanced phase from crawl to store_article")