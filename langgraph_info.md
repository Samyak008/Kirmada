# LangGraph Folder Documentation

## Overview
The langgraph folder contains the implementation of an AI-powered content creation system that focuses on automating the entire content production process. It includes agents specialized in different aspects of content creation: research, scripting, image generation, video creation, voiceover generation, and storage management.

## Key Components

### 1. Agent Structure
The core of the system is `agent1.py` which serves as the main orchestrator for the entire content creation workflow. It uses LangGraph to manage the flow between different specialized agents:

- **Search Agent**: Searches for trending tech news using Tavily API
- **Crawl Agent**: Extracts full content from selected URLs using Mistral OCR
- **Supabase Agent**: Stores content and metadata in Supabase database
- **Scripting Agent**: Generates viral social media scripts optimized for different platforms
- **Prompt Generation Agent**: Creates detailed image prompts for visual content
- **Image Generation Agent**: Uses Together AI FLUX API for high-quality image generation
- **Video Generation Agent**: Creates video content from static images using motion effects
- **Voice Generation Agent**: Converts script text to speech using Chatterbox TTS
- **Visual Agent**: Creates timing plans and visual elements for content
- **Google Drive Storage**: Handles uploading and organizing generated content

### 2. System Architecture
The system follows a multi-phase workflow:

1. **Search & Discovery**: Uses search agents to find trending tech news
2. **Content Selection**: Waits for user to select articles to process
3. **Content Extraction**: Crawls and extracts full content using OCR
4. **Database Storage**: Automatically stores content in Supabase with metadata
5. **Script Generation**: Creates platform-specific scripts
6. **Visual Planning**: Generates visual timing and shot breakdown
7. **Image Generation**: Creates AI-generated images using FLUX API
8. **Voice Generation**: Converts scripts to voiceover
9. **Video Creation**: Combines images and voiceovers to create videos

### 3. Agent Files

#### `agent1.py`
Main orchestrator that combines all agent tools and implements the workflow logic. Uses a comprehensive system prompt to guide the AI agent through the complete content production process.

#### `search_agent.py`
Contains tools for searching trending tech news from official sources like TechCrunch, The Verge, and Arstechnica. Filters out aggregated content to focus on single articles with specific publication dates.

#### `crawl_agent.py`
Implements web crawling functionality to extract full article content and images using OCR technology. Returns structured data including title, content, metadata, and image URLs.

#### `supabase_agent.py`
Handles database storage operations in Supabase, including storing articles, scripts, and their metadata. Implements duplicate detection using URL hashing.

#### `scripting_agent.py`
Generates viral social media scripts using multiple templates (announcement, reaction, tutorial, trend analysis, comparison, storytelling) optimized for different platforms like YouTube, TikTok, Instagram, and LinkedIn.

#### `image_generation_agent.py`
Implements image generation using Together AI FLUX API as primary method, with fallbacks to other services like DALL-E, Canva, and Leonardo.AI. Handles prompt optimization and retry strategies.

#### `gdrive_storage.py`
Manages Google Drive organization and file uploads, creating structured folder hierarchies for different content types (images, voiceovers, videos, crawl data).

#### `voice_generation_agent.py`
Uses Chatterbox TTS for voiceover generation with emotion control, voice cloning capabilities, and automatic upload to Google Drive.

#### `video_generation_agent.py`
Creates videos from static images using motion effects, with multiple generation methods including Google Gemini, Replicate Stable Video Diffusion, and OpenCV transitions.

## Tools and Integration

### External APIs
- **Tavily API**: For searching trending news
- **Together AI FLUX**: For image generation
- **Google APIs**: For Drive storage
- **Supabase**: For database storage
- **OpenAI/DeepSeek**: For LLM processing

### File Organization
The system automatically organizes generated content with:
- Main project folder: "RocketReelsAI"
- Subfolders: generated_images, voiceover, generated_videos, crawl_data, search_results, prompts
- Topic-based subfolders for better organization
- Automatic upload and synchronization

## Workflow Process

### Phase 1 - Search & Discovery
- Execute comprehensive search for trending tech news
- Present top 8 curated articles with source domains
- Wait for user selection before proceeding

### Phase 2 - Content Extraction & Storage
- Extract full content and images from selected URLs
- Automatically store in Supabase database with metadata
- Present comprehensive content package

### Phase 3 - Script Generation
- Generate viral script optimized for selected platform
- Create engaging hooks, structured content, and CTAs

### Phase 4 - Visual Elements
- Create image prompts from script
- Generate AI images with detailed visual descriptions
- Plan motion and transitions for video creation

### Phase 5 - Audio and Video Production
- Convert script to voiceover
- Generate videos from static images with motion effects
- Combine all elements into final content

## Error Handling

The system implements comprehensive error handling with:
- Fallback strategies for failed image generation
- Database connection recovery
- Retry mechanisms with exponential backoff
- Clear alternatives when automatic processes fail

## Execution

The system is designed to be executed as part of a LangGraph workflow, with the `agent1.py` serving as the main entry point. It can be invoked with different content requests and will automatically execute the complete content production workflow.