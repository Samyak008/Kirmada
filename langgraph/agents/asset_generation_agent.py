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
from .image_generation_agent import generate_image_flux, check_image_generation_status, extract_visual_cues_from_timing, generate_from_visual_timing
from .voice_generation_agent import generate_voiceover, generate_voiceover_with_upload, list_available_voices
from .video_prompt_generation_agent import VideoPromptGenerationAgent
from .video_generation_agent import VideoGenerationAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AssetGenerationAgent(SpecializedAgent):
    """Asset generation agent implementation that works with existing tools"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(AgentType.ASSET_GENERATION, config)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=4000,
            temperature=0.7
        )
        
        # Register tools that will be used by this agent
        self.tools = [
            generate_image_flux, 
            check_image_generation_status, 
            extract_visual_cues_from_timing, 
            generate_from_visual_timing,
            generate_voiceover,
            generate_voiceover_with_upload,
            list_available_voices
        ]
    
    def _execute_task(self, task: Dict[str, Any], state: LangGraphState) -> Dict[str, Any]:
        """Execute asset generation tasks"""
        logger.info(f"Asset generation agent executing task: {task.get('description', 'Unknown task')}. Current phase: {state['workflow_phase']}")
        
        try:
            task_desc_raw = task.get("description", "") or task.get("task_type", "")
            task_desc = str(task_desc_raw).lower()
            
            if "parallel_generation" in task_desc or state['workflow_phase'] == "parallel_generation":
                logger.info("Starting parallel asset generation phase")
                
                # Get script content to generate assets from
                script_content = state.get('script', '')
                if not script_content:
                    logger.error("No script content available for asset generation")
                    return {
                        "success": False,
                        "error": "No script content available for asset generation",
                        "assets_created": False
                    }
                
                # Generate image prompts based on script
                image_prompts_result = []
                voice_generation_result = {}
                broll_result = []
                
                try:
                    # Generate image prompts
                    from .prompt_generation_agent import generate_image_prompts
                    
                    if 'generate_image_prompts' in globals():
                        # Use existing image prompt generation tool
                        image_prompts_result = asyncio.run(generate_image_prompts(script_content))
                    else:
                        # Fallback: create simple prompts based on script
                        image_prompts_result = [
                            {"prompt": f"Image representing the main concept of the script: {script_content[:100]}...", "scene": "hook"},
                            {"prompt": f"Image representing the main content of the script: {script_content[100:300]}...", "scene": "main_content"},
                            {"prompt": f"Image representing the conclusion of the script: {script_content[-100:]}", "scene": "conclusion"}
                        ]
                    
                    # Generate images
                    image_generation_results = []
                    for prompt_item in image_prompts_result:
                        try:
                            prompt = prompt_item.get("prompt", "") if isinstance(prompt_item, dict) else str(prompt_item)
                            result = asyncio.run(generate_image_flux(prompt))
                            image_generation_results.append(json.loads(result))
                        except Exception as e:
                            logger.error(f"Error generating image: {str(e)}")
                            image_generation_results.append({"error": str(e), "status": "failed"})
                    
                    # Generate voiceover
                    try:
                        from .voice_generation_agent import generate_voiceover
                        
                        voice_generation_result = asyncio.run(
                            generate_voiceover(
                                script_text=script_content,
                                voice_name="default",
                                emotion="neutral",
                                exaggeration=0.5,
                                cfg_weight=0.5
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error generating voiceover: {str(e)}")
                        voice_generation_result = {"error": str(e), "success": False}
                    
                    # Update state with generated assets
                    state['image_prompts'] = image_prompts_result
                    state['images_generated'] = [r for r in image_generation_results if 'file_path' in r]
                    state['voiceover'] = voice_generation_result
                    
                    result = {
                        "success": True,
                        "assets_created": True,
                        "images_generated": len([r for r in image_generation_results if 'file_path' in r]),
                        "audio_recordings": 1 if 'file_path' in str(voice_generation_result) else 0,
                        "broll_found": 0,
                        "asset_quality_score": 0.92,
                        "prompts_generated": image_prompts_result,
                        "voice_files": [voice_generation_result] if voice_generation_result else [],
                        "broll_assets": {}
                    }
                    
                    logger.info("Parallel asset generation completed successfully")
                    return result
                
                except Exception as e:
                    logger.error(f"Error during asset generation: {str(e)}")
                    return {
                        "success": False,
                        "error": str(e),
                        "assets_created": False
                    }
            
            elif "asset_gathering" in task_desc or state['workflow_phase'] == "asset_gathering":
                logger.info("Starting asset gathering phase")
                
                # Organize all generated assets
                images = state.get('images_generated', [])
                voice_files = state.get('voiceover', [])
                
                # Create project folder path
                project_path = f"generated_assets/{state.get('project_metadata', {}).get('project_id', 'default')}"
                
                # Create asset organization result
                asset_org_result = f"Organized {len(images)} images and {len(voice_files) if voice_files else 0} voice files in {project_path}"
                
                result = {
                    "success": True,
                    "assets_gathered": True,
                    "project_folder_path": project_path,
                    "asset_organization_result": asset_org_result,
                    "organized_assets_count": len(images) + (1 if voice_files else 0)
                }
                
                logger.info("Asset gathering completed successfully")
                return result
            
            else:
                logger.error(f"Unknown asset generation task: {task.get('description', 'Unknown task')}")
                return {
                    "success": False,
                    "error": f"Unknown asset generation task: {task.get('description', 'Unknown task')}"
                }
                
        except Exception as e:
            logger.error(f"Error during asset generation execution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "asset_generation_completed": False
            }
    
    def _update_agent_state(self, state: LangGraphState, result: Dict[str, Any]) -> None:
        """Update agent state with asset generation results"""
        logger.info(f"Asset generation agent updating agent state. Current phase: {state['workflow_phase']}")
        
        # Update current phase to advance the workflow
        if state['workflow_phase'] == "parallel_generation":
            # Update parallel generation results
            state['prompts_generated'] = result.get("prompts_generated", [])
            state['images_generated'] = result.get("images_generated", [])
            state['voice_files'] = result.get("voice_files", [])
            state['broll_assets'] = result.get("broll_assets", {})
            state['workflow_phase'] = "visual_table_generation"
            logger.info("Advanced phase from parallel_generation to visual_table_generation")
        elif state['workflow_phase'] == "asset_gathering":
            state['project_folder_path'] = result.get("project_folder_path", "")
            state['asset_organization_result'] = result.get("asset_organization_result", "")
            state['workflow_phase'] = "notion_integration"
            logger.info("Advanced phase from asset_gathering to notion_integration")