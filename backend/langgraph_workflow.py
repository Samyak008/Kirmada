from __future__ import annotations  # ensure annotations aren't evaluated at import time

import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Union
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import yaml
import json
from datetime import datetime
from models import (
    AgentState, AgentType, TaskStatus, WorkflowPhase, 
    AgentMessage, SupervisorDecision, QualityCheck, Task  # add Task here
)
from tools import get_tools_for_agent, validate_tool_input, ToolResult


class WorkflowState(TypedDict):
    """LangGraph state for the agentic workflow"""
    agent_state: AgentState
    messages: List[BaseMessage]
    current_agent: Optional[AgentType]
    next_agent: Optional[AgentType]
    task_results: Dict[str, Any]
    errors: List[Dict[str, Any]]
    workflow_completed: bool
class UserInput(BaseModel):
    """Schema for initial input shown in LangGraph Studio."""
    project_id: str = Field(default="proj_demo")
    project_name: str = Field(default="Demo Project")
    content_request: str = Field(default="Create a short video about AI trends.")
    content_type: str = Field(default="youtube_video")
    target_audience: str = Field(default="tech enthusiasts")
    deadline: Optional[str] = Field(default=None, description="ISO datetime string")
    current_phase: str = Field(default="search")
    current_step: str = Field(default="initialize")
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class AgentRouter:
    """Handles routing between different agents based on current state"""
    
    def __init__(self, config_path: str = "agent_prompts.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def should_continue(self, state: WorkflowState) -> str:
        """Determine if workflow should continue and which agent to call next"""
        agent_state = state["agent_state"]
        
        # Check if workflow is completed
        if agent_state.workflow_completed:
            return "end"
        
        # Check for errors that need handling
        if state["errors"]:
            return "error_handler"
        
        # Determine next agent based on current phase and completed tasks
        next_agent = self._determine_next_agent(agent_state)
        
        if next_agent is None:
            return "end"
        
        return f"agent_{next_agent.value}"
    
    def _determine_next_agent(self, agent_state: AgentState) -> Optional[AgentType]:
        """Determine which agent should be called next based on current state"""
        current_phase = agent_state.current_phase
        
        # Phase-based routing (matching original workflow)
        phase_routing = {
            "search": [AgentType.RESEARCH],
            "crawl": [AgentType.RESEARCH],
            "store_article": [AgentType.STORAGE],
            "generate_script": [AgentType.CONTENT],
            "store_script": [AgentType.STORAGE],
            "shot_analysis": [AgentType.CONTENT],
            "parallel_generation": [AgentType.ASSET_GENERATION],
            "visual_table_generation": [AgentType.CONTENT],
            "asset_gathering": [AgentType.STORAGE],
            "notion_integration": [AgentType.PROJECT_MANAGEMENT],
            "finalize": [AgentType.SUPERVISOR]
        }
        
        available_agents = phase_routing.get(current_phase, [])
        
        # Check which agents have pending tasks
        for agent in available_agents:
            agent_tasks = [task for task in agent_state.tasks 
                          if task.agent_type == agent and task.status == TaskStatus.PENDING]
            if agent_tasks:
                return agent
        
        # If no pending tasks, check if phase is complete and move to next phase
        if self._is_phase_complete(agent_state, current_phase):
            next_phase = self._get_next_phase(current_phase)
            if next_phase:
                agent_state.current_phase = next_phase
                return self._determine_next_agent(agent_state)
        
        return None
    
    def _is_phase_complete(self, agent_state: AgentState, phase: str) -> bool:
        """Check if current phase is complete"""
        phase_agents = {
            "search": [AgentType.RESEARCH],
            "crawl": [AgentType.RESEARCH],
            "store_article": [AgentType.STORAGE],
            "generate_script": [AgentType.CONTENT],
            "store_script": [AgentType.STORAGE],
            "shot_analysis": [AgentType.CONTENT],
            "parallel_generation": [AgentType.ASSET_GENERATION],
            "visual_table_generation": [AgentType.CONTENT],
            "asset_gathering": [AgentType.STORAGE],
            "notion_integration": [AgentType.PROJECT_MANAGEMENT],
            "finalize": [AgentType.SUPERVISOR]
        }
        
        required_agents = phase_agents.get(phase, [])
        for agent in required_agents:
            agent_tasks = [task for task in agent_state.tasks 
                          if task.agent_type == agent and task.status != TaskStatus.COMPLETED]
            if agent_tasks:
                return False
        
        return True
    
    def _get_next_phase(self, current_phase: str) -> Optional[str]:
        """Get the next phase in the workflow"""
        phase_sequence = [
            "search", "crawl", "store_article", "generate_script", 
            "store_script", "shot_analysis", "parallel_generation",
            "visual_table_generation", "asset_gathering", 
            "notion_integration", "finalize"
        ]
        
        try:
            current_index = phase_sequence.index(current_phase)
            if current_index < len(phase_sequence) - 1:
                return phase_sequence[current_index + 1]
        except ValueError:
            pass
        
        return None


class SupervisorAgent:
    """Supervisor agent that makes decisions and coordinates the workflow"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_prompt = config["agents"]["supervisor"]["system_prompt"]
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Process supervisor decisions and task assignments"""
        ################ToDO -> Pass it to the language model when making decisions or processing tasks
        ################Verify that system prompt is being used correctly
        system_message = SystemMessage(content=self.system_prompt)
        state["messages"].append(system_message)
        agent_state = state["agent_state"]
        
        # Analyze current state and make decisions
        decisions = self._make_decisions(agent_state)
        
        # Update agent state with decisions
        for decision in decisions:
            agent_state.supervisor_decisions.append(decision)
            
            # Create task if needed
            if decision.chosen_agent != AgentType.SUPERVISOR:
                task = self._create_task(decision, agent_state)
                agent_state.tasks.append(task)
        
        # Update messages
        supervisor_message = AIMessage(
            content=f"Supervisor decisions made: {len(decisions)} tasks assigned"
        )
        state["messages"].append(supervisor_message)
        
        return state
    
    def _make_decisions(self, agent_state: AgentState) -> List[SupervisorDecision]:
        """Make strategic decisions about next steps"""
        decisions = []
        
        # Analyze current phase and determine what needs to be done
        if agent_state.current_phase == "search":
            decisions.append(SupervisorDecision(
                decision_id=f"search_{datetime.now().timestamp()}",
                context="Search phase needs to find trending articles",
                chosen_agent=AgentType.RESEARCH,
                reasoning="Research agent needed to search for relevant content",
                expected_outcome="Search results and URLs identified",
                priority=1
            ))
        
        elif agent_state.current_phase == "crawl":
            decisions.append(SupervisorDecision(
                decision_id=f"crawl_{datetime.now().timestamp()}",
                context="Crawl phase needs to extract article content",
                chosen_agent=AgentType.RESEARCH,
                reasoning="Research agent needed to extract and structure content",
                expected_outcome="Article content extracted and structured",
                priority=1
            ))
        
        elif agent_state.current_phase == "store_article":
            decisions.append(SupervisorDecision(
                decision_id=f"store_article_{datetime.now().timestamp()}",
                context="Store article phase needs database storage",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to store article data",
                expected_outcome="Article stored in database with ID",
                priority=1
            ))
        
        elif agent_state.current_phase == "generate_script":
            decisions.append(SupervisorDecision(
                decision_id=f"generate_script_{datetime.now().timestamp()}",
                context="Script generation phase needs content creation",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to create engaging script",
                expected_outcome="Script content and hook generated",
                priority=1
            ))
        
        elif agent_state.current_phase == "store_script":
            decisions.append(SupervisorDecision(
                decision_id=f"store_script_{datetime.now().timestamp()}",
                context="Store script phase needs database storage",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to store script data",
                expected_outcome="Script stored in database with ID",
                priority=1
            ))
        
        elif agent_state.current_phase == "shot_analysis":
            decisions.append(SupervisorDecision(
                decision_id=f"shot_analysis_{datetime.now().timestamp()}",
                context="Shot analysis phase needs shot breakdown",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to analyze shots and timing",
                expected_outcome="Shot breakdown, timing, and types analyzed",
                priority=1
            ))
        
        elif agent_state.current_phase == "parallel_generation":
            decisions.append(SupervisorDecision(
                decision_id=f"parallel_generation_{datetime.now().timestamp()}",
                context="Parallel generation phase needs asset creation",
                chosen_agent=AgentType.ASSET_GENERATION,
                reasoning="Asset generation agent needed for parallel asset creation",
                expected_outcome="Prompts, images, voice, and b-roll generated",
                priority=1
            ))
        
        elif agent_state.current_phase == "visual_table_generation":
            decisions.append(SupervisorDecision(
                decision_id=f"visual_table_{datetime.now().timestamp()}",
                context="Visual table generation phase needs organization",
                chosen_agent=AgentType.CONTENT,
                reasoning="Content agent needed to generate visual table",
                expected_outcome="Visual table created for organization",
                priority=1
            ))
        
        elif agent_state.current_phase == "asset_gathering":
            decisions.append(SupervisorDecision(
                decision_id=f"asset_gathering_{datetime.now().timestamp()}",
                context="Asset gathering phase needs organization",
                chosen_agent=AgentType.STORAGE,
                reasoning="Storage agent needed to gather and organize assets",
                expected_outcome="Assets gathered and organized",
                priority=1
            ))
        
        elif agent_state.current_phase == "notion_integration":
            decisions.append(SupervisorDecision(
                decision_id=f"notion_integration_{datetime.now().timestamp()}",
                context="Notion integration phase needs project setup",
                chosen_agent=AgentType.PROJECT_MANAGEMENT,
                reasoning="Project management agent needed for Notion setup",
                expected_outcome="Notion project created and integrated",
                priority=1
            ))
        
        elif agent_state.current_phase == "finalize":
            decisions.append(SupervisorDecision(
                decision_id=f"finalize_{datetime.now().timestamp()}",
                context="Finalization phase needs project completion",
                chosen_agent=AgentType.SUPERVISOR,
                reasoning="Supervisor needed for final review and completion",
                expected_outcome="Project finalized and ready for handoff",
                priority=1
            ))
        
        return decisions
    
    def _create_task(self, decision: SupervisorDecision, agent_state: AgentState) -> Task:
        """Create a task based on supervisor decision"""
        return Task(
            task_id=f"task_{decision.decision_id}",
            agent_type=decision.chosen_agent,
            description=decision.expected_outcome,
            status=TaskStatus.PENDING,
            priority=decision.priority,
            assigned_at=datetime.now()
        )


class SpecializedAgent:
    """Base class for specialized agents (Research, Content, Asset Generation, etc.)"""
    
    def __init__(self, agent_type: AgentType, config: Dict[str, Any]):
        self.agent_type = agent_type
        self.config = config
        self.system_prompt = config["agents"][agent_type.value]["system_prompt"]
        self.tools = get_tools_for_agent(agent_type.value)
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """Process agent-specific tasks"""
        ################ToDO -> Pass it to the language model when making decisions or processing tasks
        ################Verify that system prompt is being used correctly
        system_message = SystemMessage(content=self.system_prompt)
        state["messages"].append(system_message)
        
        agent_state = state["agent_state"]

        
        # Get pending tasks for this agent
        pending_tasks = [task for task in agent_state.tasks 
                        if task.agent_type == self.agent_type and task.status == TaskStatus.PENDING]
        
        if not pending_tasks:
            return state
        
        # Process the highest priority task
        task = max(pending_tasks, key=lambda t: t.priority)
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Execute agent-specific logic
            result = self._execute_task(task, agent_state)
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Update agent state with results
            self._update_agent_state(agent_state, result)
            
            # Add completion message
            completion_message = AIMessage(
                content=f"{self.agent_type.value} agent completed task: {task.description}"
            )
            state["messages"].append(completion_message)
            
        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            
            error_message = AIMessage(
                content=f"{self.agent_type.value} agent failed task: {str(e)}"
            )
            state["messages"].append(error_message)
            state["errors"].append({
                "agent": self.agent_type.value,
                "task": task.task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute agent-specific task logic - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_task")
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with task results - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _update_agent_state")


class ResearchAgent(SpecializedAgent):
    """Research agent implementation"""
    
    def __init__(self, config: Dict[str, Any]):
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

        # Placeholder for Tavily search - will be replaced when packages are installed
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
            # Analyze query intent
            intent_analysis = await self._analyze_query_intent(query)

            # Generate optimized search queries
            search_queries = await self._generate_search_queries(query, intent_analysis)

            # Execute searches
            all_results = []
            for search_query in search_queries:
                try:
                    results = await asyncio.to_thread(
                        self.search_service['search_provider'].invoke,
                        {"query": search_query}
                    )
                    all_results.extend(self._extract_structured_results(results))
                except Exception as e:
                    print(f"Search failed for query '{search_query}': {e}")
                    continue

            # Filter and rank results
            filtered_results = await self._intelligent_filter_and_rank(query, intent_analysis, all_results)

            return {
                "success": True,
                "query": query,
                "results": filtered_results[:max_results],
                "total_found": len(all_results),
                "filtered_count": len(filtered_results),
                "intent_analysis": intent_analysis
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }

    async def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Use LLM to analyze user query and determine search strategy"""
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
  "content_preferences": ["announcements", "analysis", "tutorials", "reviews"],
  "priority_sources": ["list", "of", "preferred", "source", "types"],
  "search_focus": "brief description of what to focus on"
}}

Return ONLY the JSON, no additional text."""

        try:
            response = await asyncio.to_thread(
                self.search_service['llm'].invoke,
                [HumanMessage(content=prompt)]
            )

            return json.loads(response.content.strip())
        except Exception as e:
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

            return json.loads(response.content.strip())
        except Exception as e:
            # Fallback queries
            return [
                f"{original_query} latest news",
                f"{original_query} recent developments",
                f"{original_query} analysis",
                f"{original_query} official announcement"
            ]

    def _extract_structured_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        """Extract structured data from search results"""
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
            print(f"Error extracting structured results: {e}")

        return structured_results

    async def _intelligent_filter_and_rank(self, query: str, strategy: Dict[str, Any], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to intelligently filter and rank results"""
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

            # Add relevance scores and validate
            for result in ranked_results:
                if "score" not in result:
                    result["score"] = 0.8

            return ranked_results

        except Exception as e:
            # Fallback: return top results with default scores
            return results[:8]

    async def _crawl_url(self, url: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Crawl a URL using available adapters with fallback"""
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
            result = await self.crawl_service['firecrawl_adapter'].crawl(url, crawl_options)
            if result.success:
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
            print(f"Firecrawl failed for {url}: {e}")

        # Fallback to Parallel.ai
        try:
            result = await self.crawl_service['parallelai_adapter'].crawl(url, crawl_options)
            if result.success:
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
            print(f"Parallel.ai failed for {url}: {e}")

        # Final fallback: basic HTML parsing
        return await self._enhanced_html_parsing(url)

    async def _enhanced_html_parsing(self, url: str) -> Dict[str, Any]:
        """Enhanced HTML parsing as final fallback"""
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
        import asyncio
        
        async def execute_async():
            try:
                # Accept either description or legacy task_type
                task_desc_raw = getattr(task, "description", None) or getattr(task, "task_type", "")
                task_desc = str(task_desc_raw).lower()
                
                if "search" in task_desc or agent_state.current_phase == "search":
                    # Use the actual search service
                    query = agent_state.content_request or (getattr(task, "description", None) or str(task_desc_raw))
                    search_results = await self._intelligent_search(query, max_results=8)
                    
                    if not search_results.get("success"):
                        return {
                            "success": False,
                            "error": search_results.get("error", "Search failed"),
                            "search_completed": False
                        }
                    
                    # Initialize research_data if needed
                    if not getattr(agent_state, "research_data", None):
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
                    
                    return {
                        "success": True,
                        "search_completed": True,
                        "sources_found": len(sources),
                        "urls": [s.url for s in sources if s.url],
                        "next_phase": "crawl"
                    }
                
                elif "crawl" in task_desc or agent_state.current_phase == "crawl":
                    # Use your existing crawl infrastructure
                    if not agent_state.research_data or not agent_state.research_data.sources:
                        return {
                            "success": False,
                            "error": "No search results to crawl",
                            "crawl_completed": False
                        }
                    
                    # Get URLs from search results (limit to 3 for testing)
                    urls = [source.url for source in agent_state.research_data.sources[:3] if source.url]
                    
                    if not urls:
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
                            print(f"Crawling URL: {url}")
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
                                print(f"Successfully crawled: {url} ({len(crawl_result.get('content', ''))} chars)")
                            else:
                                error_msg = crawl_result.get("error", "Unknown crawl error")
                                crawl_errors.append({"url": url, "error": error_msg})
                                print(f"Failed to crawl {url}: {error_msg}")
                                
                        except Exception as e:
                            error_msg = f"Exception while crawling {url}: {str(e)}"
                            crawl_errors.append({"url": url, "error": error_msg})
                            print(error_msg)
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
                    return {
                        "success": False,
                        "error": f"Unknown research task: {task.description}"
                    }
                    
            except Exception as e:
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
                    return future.result()
            else:
                return asyncio.run(execute_async())
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
    """Update agent state with research results"""
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
    
    # Update current step based on phase
    if agent_state.current_phase == "search":
        agent_state.current_step = "crawl"
    elif agent_state.current_phase == "crawl":
        agent_state.current_step = "store_article"


class ContentAgent(SpecializedAgent):
    """Content agent implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.CONTENT, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute content creation tasks"""
        return {
            "script_created": True,
            "shots_planned": 10,
            "visual_elements_planned": 15,
            "content_quality_score": 0.88
        }
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with content creation results"""
        from models import Script, Shot
        
        # Create mock script
        shots = [
            Shot(
                shot_number=1,
                description="Opening hook shot",
                duration_seconds=5.0,
                visual_elements=["Title card", "Background animation"],
                shot_type="establishing",
                timing_start=0.0,
                timing_end=5.0
            )
        ]
        
        script = Script(
            title=f"Content about {agent_state.content_request}",
            content_type=agent_state.content_type,
            target_audience=agent_state.target_audience,
            duration_minutes=5.0,
            hook="Engaging opening hook",
            main_content="Main content here",
            call_to_action="Subscribe for more content",
            shots=shots
        )
        
        agent_state.script = script
        
        # Update current step based on phase
        if agent_state.current_phase == "generate_script":
            agent_state.current_step = "store_script"
        elif agent_state.current_phase == "shot_analysis":
            # Update shot analysis data
            agent_state.shot_breakdown = result.get("shot_breakdown", [])
            agent_state.shot_timing = result.get("shot_timing", [])
            agent_state.shot_types = result.get("shot_types", [])
            agent_state.current_step = "parallel_generation"
        elif agent_state.current_phase == "visual_table_generation":
            agent_state.visual_table = result.get("visual_table", {})
            agent_state.current_step = "asset_gathering"


class AssetGenerationAgent(SpecializedAgent):
    """Asset generation agent implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.ASSET_GENERATION, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute asset generation tasks"""
        return {
            "assets_created": 8,
            "images_generated": 5,
            "audio_recordings": 2,
            "broll_found": 3,
            "asset_quality_score": 0.92
        }
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with asset generation results"""
        from models import Asset
        
        # Create mock assets
        assets = [
            Asset(
                asset_id="asset_1",
                asset_type="image",
                file_path="/assets/images/opening_shot.png",
                description="Opening shot image",
                shot_number=1
            )
        ]
        
        agent_state.assets.extend(assets)
        
        # Update current step based on phase
        if agent_state.current_phase == "parallel_generation":
            # Update parallel generation results
            agent_state.prompts_generated = result.get("prompts_generated", [])
            agent_state.images_generated = result.get("images_generated", [])
            agent_state.voice_files = result.get("voice_files", [])
            agent_state.broll_assets = result.get("broll_assets", {})
            agent_state.current_step = "visual_table_generation"
        elif agent_state.current_phase == "asset_gathering":
            agent_state.project_folder_path = result.get("project_folder_path", "")
            agent_state.asset_organization_result = result.get("asset_organization_result", "")
            agent_state.current_step = "notion_integration"


class StorageAgent(SpecializedAgent):
    """Storage agent implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.STORAGE, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute storage tasks"""
        return {
            "data_stored": True,
            "folders_created": 3,
            "files_uploaded": 12,
            "metadata_tagged": True
        }
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with storage results"""
        from models import ProjectFolder, DatabaseRecord
        
        # Create mock storage records
        folder = ProjectFolder(
            folder_id="project_folder_1",
            folder_name=agent_state.project_name,
            google_drive_path="/Content Production/Projects/",
            files=["file1", "file2"]
        )
        
        agent_state.project_folders.append(folder)
        
        # Update current step based on phase
        if agent_state.current_phase == "store_article":
            agent_state.current_step = "generate_script"
        elif agent_state.current_phase == "store_script":
            agent_state.current_step = "shot_analysis"


class ProjectManagementAgent(SpecializedAgent):
    """Project management agent implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.PROJECT_MANAGEMENT, config)
    
    def _execute_task(self, task: Task, agent_state: AgentState) -> Dict[str, Any]:
        """Execute project management tasks"""
        return {
            "notion_workspace_created": True,
            "project_tracking_setup": True,
            "milestones_defined": 6,
            "team_coordination_complete": True
        }
    
    def _update_agent_state(self, agent_state: AgentState, result: Dict[str, Any]) -> None:
        """Update agent state with project management results"""
        from models import NotionPage
        
        # Create mock Notion page
        page = NotionPage(
            page_id="notion_page_1",
            title=agent_state.project_name,
            content={"status": "active", "progress": 25}
        )
        
        agent_state.notion_pages.append(page)
        
        # Update current step based on phase
        if agent_state.current_phase == "notion_integration":
            agent_state.notion_project_id = result.get("notion_project_id", "")
            agent_state.notion_status = result.get("notion_status", "active")
            agent_state.current_step = "finalize"


# Simple no-op router node so conditional routing has a source node
def router_node(state: WorkflowState) -> WorkflowState:
    return state

def _default_agent_state_from_input(inp: Union[UserInput, Dict[str, Any]]) -> AgentState:
    """Build a minimal AgentState if one wasn't provided."""
    inbound = inp.dict() if isinstance(inp, UserInput) else inp
    return AgentState(
        project_id=inbound.get("project_id", "proj_demo"),
        project_name=inbound.get("project_name", "Demo Project"),
        content_request=inbound.get("content_request", "Create a short video about AI trends."),
        content_type=inbound.get("content_type", "youtube_video"),
        target_audience=inbound.get("target_audience", "tech enthusiasts"),
        deadline=inbound.get("deadline", "2099-12-31T23:59:59"),
        current_phase=inbound.get("current_phase", "search"),
        current_step=inbound.get("current_step", "initialize"),
        tasks=[],
        supervisor_decisions=[],
        research_data=None,
        script=None,
        assets=[],
        project_folders=[],
        notion_pages=[],
        messages=[],
        errors=[],
        workflow_completed=False,
    )

def init_node(state: Union[UserInput, Dict[str, Any]]) -> WorkflowState:
    """Normalize user input into full WorkflowState."""
    inbound = state or {}
    # If a bare dict with agent_state is provided, use it; otherwise build from UserInput
    agent_state = inbound.get("agent_state") if isinstance(inbound, dict) else None
    agent_state = agent_state or _default_agent_state_from_input(inbound)
    messages = inbound.get("messages", []) if isinstance(inbound, dict) else []
    return WorkflowState(
        agent_state=agent_state,
        messages=messages,
        current_agent=None,
        next_agent=None,
        task_results={},
        errors=[],
        workflow_completed=False,
    )

def create_workflow_graph(config_path: str = "agent_prompts.yaml") -> StateGraph:
    """Create the LangGraph workflow"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize agents
    supervisor = SupervisorAgent(config)
    research_agent = ResearchAgent(config)
    content_agent = ContentAgent(config)
    asset_agent = AssetGenerationAgent(config)
    storage_agent = StorageAgent(config)
    project_agent = ProjectManagementAgent(config)
    
    # Initialize router
    router = AgentRouter(config_path)
    
    # Create the graph
    workflow = StateGraph(WorkflowState)
    workflow.add_node("init", init_node)
    # Add nodes
    workflow.add_node("supervisor", supervisor.process)
    workflow.add_node("agent_research", research_agent.process)
    workflow.add_node("agent_content", content_agent.process)
    workflow.add_node("agent_asset_generation", asset_agent.process)
    workflow.add_node("agent_storage", storage_agent.process)
    workflow.add_node("agent_project_management", project_agent.process)
    workflow.add_node("error_handler", lambda state: state)  # Placeholder for error handling
    workflow.add_node("router", router_node)  # NEW: define router node

    # Add edges
    workflow.add_edge("supervisor", "router")
    workflow.add_edge("agent_research", "router")
    workflow.add_edge("agent_content", "router")
    workflow.add_edge("agent_asset_generation", "router")
    workflow.add_edge("agent_storage", "router")
    workflow.add_edge("agent_project_management", "router")
    workflow.add_edge("error_handler", "router")
    workflow.add_edge("init", "supervisor")

    # Add conditional routing
    workflow.add_conditional_edges(
        "router",
        router.should_continue,
        {
            "agent_research": "agent_research",
            "agent_content": "agent_content",
            "agent_asset_generation": "agent_asset_generation",
            "agent_storage": "agent_storage",
            "agent_project_management": "agent_project_management",
            "agent_supervisor": "supervisor",  # NEW: matches f"agent_{AgentType.SUPERVISOR.value}"
            "error_handler": "error_handler",
            "end": END
        }
    )

    # Set entry point
    workflow.set_entry_point("init")
    return workflow


# Example usage
if __name__ == "__main__":
    # Create and compile the workflow
    graph = create_workflow_graph()
    app = graph.compile()
    
    # Example initial state
    initial_state = WorkflowState(
        agent_state=AgentState(
            project_id="proj_001",
            project_name="Tech Content Production",
            content_request="Create a video about AI trends in 2024",
            content_type="youtube_video",
            target_audience="tech enthusiasts"
        ),
        messages=[],
        current_agent=None,
        next_agent=None,
        task_results={},
        errors=[],
        workflow_completed=False
    )
    
    # Run the workflow
    result = app.invoke(initial_state)
    print("Workflow completed!")
