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
from .supabase_agent import (
    store_article_content_sync_wrapped, 
    store_multiple_articles, 
    retrieve_stored_articles, 
    get_article_by_url, 
    get_stored_article_by_keyword, 
    get_article_id_by_url,
    store_script_content,
    retrieve_stored_scripts,
    get_script_by_id,
    get_scripts_by_article_id,
    approve_script
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StorageAgent(SpecializedAgent):
    """Storage agent implementation that works with existing tools"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.STORAGE, config)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=4000,
            temperature=0.1
        )
        
        # Register tools that will be used by this agent
        self.tools = [
            store_article_content_sync_wrapped,
            store_multiple_articles,
            retrieve_stored_articles,
            get_article_by_url,
            get_stored_article_by_keyword,
            get_article_id_by_url,
            store_script_content,
            retrieve_stored_scripts,
            get_script_by_id,
            get_scripts_by_article_id,
            approve_script
        ]
    
    def _execute_task(self, task: Dict[str, Any], state: LangGraphState) -> Dict[str, Any]:
        """Execute storage tasks"""
        logger.info(f"Storage agent executing task: {task.get('description', 'Unknown task')}. Current phase: {state['workflow_phase']}")
        
        try:
            task_desc_raw = task.get("description", "") or task.get("task_type", "")
            task_desc = str(task_desc_raw).lower()
            
            if "store_article" in task_desc or state['workflow_phase'] == "store_article":
                logger.info("Starting article storage phase")
                
                # Get research data to store
                research_data = state.get('research_data', {})
                if not research_data or not research_data.get('sources'):
                    logger.error("No research data available for storage")
                    return {
                        "success": False,
                        "error": "No research data available for storage",
                        "storage_completed": False
                    }
                
                # Use the first source to store
                source = research_data['sources'][0] if research_data.get('sources') else {}
                
                # Prepare article data for storage
                article_data = {
                    'url': source.get('url', ''),
                    'title': source.get('title', ''),
                    'content': source.get('content', '')[:5000],  # Limit content length
                    'image_urls': [],  # Extract from content if needed
                    'domain': source.get('url', '').split('/')[2] if source.get('url', '') else 'unknown',
                    'word_count': len(source.get('content', '').split()),
                    'image_metadata': {},
                    'metadata': {
                        'content_type': state.get('content_type', 'unknown'),
                        'source_type': 'research'
                    }
                }
                
                # Use existing storage tool
                try:
                    result = asyncio.run(store_article_content_sync_wrapped.ainvoke({"article_data": article_data}))
                    
                    result_dict = {
                        "success": True,
                        "data_stored": True,
                        "article_id": "unknown",  # Extract from result if possible
                        "storage_result": result
                    }
                    
                    # Update state with storage info
                    if 'stored_articles' not in state:
                        state['stored_articles'] = []
                    state['stored_articles'].append({
                        'url': article_data['url'],
                        'title': article_data['title'],
                        'id': "unknown"  # Extract from result
                    })
                    
                    logger.info("Article storage completed successfully")
                    return result_dict
                except Exception as e:
                    logger.error(f"Error storing article: {str(e)}")
                    return {
                        "success": False,
                        "error": str(e),
                        "storage_completed": False
                    }
            
            elif "store_script" in task_desc or state['workflow_phase'] == "store_script":
                logger.info("Starting script storage phase")
                
                # Get script content to store
                script_content = state.get('script', '')
                if not script_content:
                    logger.error("No script content available for storage")
                    return {
                        "success": False,
                        "error": "No script content available for storage",
                        "script_storage_completed": False
                    }
                
                # Get article ID if available
                article_id = None
                if state.get('stored_articles'):
                    article_id = state['stored_articles'][-1].get('id', '')
                
                # Prepare script data for storage
                script_data = {
                    'article_id': article_id or 'unknown',
                    'platform': state.get('content_type', 'youtube'),
                    'script_content': script_content,
                    'hook': script_content[:100] if script_content else '',  # First 100 chars as hook
                    'visual_suggestions': state.get('visual_suggestions', []),
                    'metadata': {
                        'word_count': len(script_content.split()) if script_content else 0,
                        'estimated_duration': len(script_content.split()) * 0.5 if script_content else 0,
                        'platform': state.get('content_type', 'youtube'),
                        'image_count': len(state.get('images_generated', []))
                    }
                }
                
                # Use existing script storage tool
                try:
                    from .supabase_agent import store_script_content
                    result = store_script_content(script_data)
                    
                    result_dict = {
                        "success": True,
                        "script_stored": True,
                        "script_id": "unknown",  # Extract from result if possible
                        "storage_result": result
                    }
                    
                    # Update state with storage info
                    if 'stored_scripts' not in state:
                        state['stored_scripts'] = []
                    state['stored_scripts'].append({
                        'id': "unknown",  # Extract from result
                        'platform': script_data['platform'],
                        'content_preview': script_content[:100]
                    })
                    
                    logger.info("Script storage completed successfully")
                    return result_dict
                except Exception as e:
                    logger.error(f"Error storing script: {str(e)}")
                    return {
                        "success": False,
                        "error": str(e),
                        "script_storage_completed": False
                    }
            
            else:
                logger.error(f"Unknown storage task: {task.get('description', 'Unknown task')}")
                return {
                    "success": False,
                    "error": f"Unknown storage task: {task.get('description', 'Unknown task')}"
                }
                
        except Exception as e:
            logger.error(f"Error during storage execution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "storage_completed": False
            }
    
    def _update_agent_state(self, state: LangGraphState, result: Dict[str, Any]) -> None:
        """Update agent state with storage results"""
        logger.info(f"Storage agent updating agent state. Current phase: {state['workflow_phase']}")
        
        # Update current phase to advance the workflow
        if state['workflow_phase'] == "store_article":
            state['workflow_phase'] = "generate_script"
            logger.info("Advanced phase from store_article to generate_script")
        elif state['workflow_phase'] == "store_script":
            state['workflow_phase'] = "shot_analysis"
            logger.info("Advanced phase from store_script to shot_analysis")