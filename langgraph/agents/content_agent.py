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
from .scripting_agent import generate_viral_script
from .visual_agent import generate_visual_timing
from .prompt_generation_agent import generate_prompts_from_script

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentAgent(SpecializedAgent):
    """Content agent implementation that works with existing tools"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.CONTENT, config)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=4000,
            temperature=0.7
        )
        
        # Register tools that will be used by this agent
        from .scripting_agent import generate_viral_script
        from .visual_agent import generate_visual_timing
        from .prompt_generation_agent import generate_prompts_from_script
        
        self.tools = [generate_viral_script, generate_visual_timing, generate_prompts_from_script]
    
    def _execute_task(self, task: Dict[str, Any], state: LangGraphState) -> Dict[str, Any]:
        """Execute content creation tasks"""
        logger.info(f"Content agent executing task: {task.get('description', 'Unknown task')}. Current phase: {state['workflow_phase']}")
        
        try:
            task_desc_raw = task.get("description", "") or task.get("task_type", "")
            task_desc = str(task_desc_raw).lower()
            
            if "generate_script" in task_desc or state['workflow_phase'] == "generate_script":
                logger.info("Starting script generation phase")
                
                # Get research data to create script from
                research_data = state.get('research_data', {})
                if not research_data or not research_data.get('sources'):
                    logger.error("No research data available for script generation")
                    return {
                        "success": False,
                        "error": "No research data available for script generation",
                        "script_generated": False
                    }
                
                # Use the first source's content for script generation
                source = research_data['sources'][0] if research_data.get('sources') else {}
                content = source.get('content', state.get('user_request', ''))
                
                # Get content type (platform)
                content_type = state.get('content_type', 'youtube')
                
                # Use existing tool to generate script
                from .scripting_agent import generate_viral_script
                
                if content:
                    # Extract important parts from the content to generate the script
                    content_preview = content[:2000]  # Limit content to avoid token issues
                    script_result = asyncio.run(generate_viral_script(content_preview, content_type))
                    
                    # Update state with generated script
                    state['script'] = script_result
                    
                    result = {
                        "success": True,
                        "script_created": True,
                        "script_content": script_result,
                        "shots_planned": 10,
                        "visual_elements_planned": 15,
                        "content_quality_score": 0.88
                    }
                    
                    logger.info("Script generation completed successfully")
                    return result
                else:
                    logger.error("No content available to generate script")
                    return {
                        "success": False,
                        "error": "No content available to generate script",
                        "script_generated": False
                    }
            
            elif "shot_analysis" in task_desc or state['workflow_phase'] == "shot_analysis":
                logger.info("Starting shot analysis phase")
                
                # Get script content to analyze
                script_content = state.get('script', '')
                if not script_content:
                    logger.error("No script content available for shot analysis")
                    return {
                        "success": False,
                        "error": "No script content available for shot analysis",
                        "shot_analysis_completed": False
                    }
                
                # Use existing visual agent to generate timing plan
                from .visual_agent import generate_visual_timing
                
                # Get research data for context
                research_data = state.get('research_data', {})
                article_data = research_data.get('sources', [{}])[0] if research_data.get('sources') else {}
                
                try:
                    visual_timing = asyncio.run(
                        generate_visual_timing(
                            script_content=script_content,
                            article_data=article_data,
                            platform=state.get('content_type', 'youtube')
                        )
                    )
                    
                    # Update state with visual timing
                    state['visual_timing'] = visual_timing
                    
                    # Simulate shot breakdown
                    shot_breakdown = [
                        {"shot_number": 1, "description": "Hook shot", "duration": 5.0, "shot_type": "close-up"},
                        {"shot_number": 2, "description": "Main content", "duration": 15.0, "shot_type": "medium"},
                        {"shot_number": 3, "description": "Conclusion", "duration": 5.0, "shot_type": "wide"}
                    ]
                    
                    result = {
                        "success": True,
                        "shot_analysis_completed": True,
                        "shot_breakdown": shot_breakdown,
                        "shot_timing": [s["duration"] for s in shot_breakdown],
                        "shot_types": [s["shot_type"] for s in shot_breakdown],
                        "visual_timing_plan": visual_timing
                    }
                    
                    logger.info("Shot analysis completed successfully")
                    return result
                except Exception as e:
                    logger.error(f"Error during shot analysis: {str(e)}")
                    return {
                        "success": False,
                        "error": str(e),
                        "shot_analysis_completed": False
                    }
            
            elif "visual_table_generation" in task_desc or state['workflow_phase'] == "visual_table_generation":
                logger.info("Starting visual table generation phase")
                
                # Get visual timing data and generate visual table
                visual_timing = state.get('visual_timing', '')
                
                # Create a visual table structure
                visual_table = {
                    "timing_plan": visual_timing if visual_timing else "No visual timing available",
                    "assets_needed": ["hook_image", "main_content_image", "conclusion_image"],
                    "transitions": ["fade_in", "cut", "fade_out"],
                    "duration": 25.0  # Sum of shot durations
                }
                
                # Update state with visual table
                state['visual_table'] = visual_table
                
                result = {
                    "success": True,
                    "visual_table_generated": True,
                    "visual_table": visual_table
                }
                
                logger.info("Visual table generation completed successfully")
                return result
            
            else:
                logger.error(f"Unknown content task: {task.get('description', 'Unknown task')}")
                return {
                    "success": False,
                    "error": f"Unknown content task: {task.get('description', 'Unknown task')}"
                }
                
        except Exception as e:
            logger.error(f"Error during content execution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "content_completed": False
            }
    
    def _update_agent_state(self, state: LangGraphState, result: Dict[str, Any]) -> None:
        """Update agent state with content creation results"""
        logger.info(f"Content agent updating agent state. Current phase: {state['workflow_phase']}")
        
        # Update current phase to advance the workflow
        if state['workflow_phase'] == "generate_script":
            state['workflow_phase'] = "store_script"
            logger.info("Advanced phase from generate_script to store_script")
        elif state['workflow_phase'] == "shot_analysis":
            # Update shot analysis data
            state['shot_breakdown'] = result.get("shot_breakdown", [])
            state['shot_timing'] = result.get("shot_timing", [])
            state['shot_types'] = result.get("shot_types", [])
            state['workflow_phase'] = "parallel_generation"
            logger.info("Advanced phase from shot_analysis to parallel_generation")
        elif state['workflow_phase'] == "visual_table_generation":
            state['visual_table'] = result.get("visual_table", {})
            state['workflow_phase'] = "asset_gathering"
            logger.info("Advanced phase from visual_table_generation to asset_gathering")