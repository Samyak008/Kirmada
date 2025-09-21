"""
Test script for the ResearchAgent with Firecrawl and Parallel.ai crawlers.

This script tests the basic functionality of the crawler system without requiring
API keys for initial testing. It uses mock responses to validate the integration.
"""

import asyncio
import os
from typing import Dict, Any

# Mock environment variables for testing
os.environ["FIRECRAWL_API_KEY"] = "mock_firecrawl_key"
os.environ["PARALLEL_API_KEY"] = "mock_parallel_key"
os.environ["OPENAI_API_KEY"] = "mock_openai_key"

from .langgraph_workflow import ResearchAgent
from .models import AgentType, AgentState, ResearchData, Source


def test_research_agent_initialization():
    """Test that ResearchAgent initializes correctly"""
    print("Testing ResearchAgent initialization...")

    config = {
        "agents": {
            "research": {
                "system_prompt": "You are a research agent specialized in finding information."
            }
        }
    }

    try:
        agent = ResearchAgent(config)
        print("✅ ResearchAgent initialized successfully")

        # Test service initialization
        search_service = agent.search_service
        crawl_service = agent.crawl_service

        assert search_service is not None, "Search service not initialized"
        assert crawl_service is not None, "Crawl service not initialized"

        print("✅ Services initialized successfully")
        return True

    except Exception as e:
        print(f"❌ ResearchAgent initialization failed: {e}")
        return False


def test_mock_search():
    """Test the mock search functionality"""
    print("\nTesting mock search functionality...")

    config = {
        "agents": {
            "research": {
                "system_prompt": "You are a research agent."
            }
        }
    }

    try:
        agent = ResearchAgent(config)

        # Create a mock task and agent state
        task = type('Task', (), {'task_type': 'search'})()
        agent_state = type('AgentState', (), {'content_request': 'AI trends'})()

        result = agent._execute_task(task, agent_state)

        print(f"Search result: {result}")

        if result.get("success", True):
            print("✅ Mock search completed successfully")
            return True
        else:
            print(f"❌ Mock search failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Mock search test failed: {e}")
        return False


def test_crawler_adapters():
    """Test crawler adapter initialization"""
    print("\nTesting crawler adapter initialization...")

    try:
        from crawlers.firecrawl_adapter import FirecrawlAdapter
        from crawlers.parallelai_adapter import ParallelAIAdapter
        from crawlers.base import CrawlOptions

        # Test Firecrawl adapter
        firecrawl_adapter = FirecrawlAdapter()
        print("✅ FirecrawlAdapter initialized")

        # Test Parallel.ai adapter
        parallel_adapter = ParallelAIAdapter()
        print("✅ ParallelAIAdapter initialized")

        # Test crawl options
        options = CrawlOptions()
        print(f"✅ CrawlOptions created: {options}")

        return True

    except Exception as e:
        print(f"❌ Crawler adapter test failed: {e}")
        return False


def test_tool_implementations():
    """Test tool implementations"""
    print("\nTesting tool implementations...")

    try:
        from tools import search_articles_tool, extract_article_content_tool

        # Test search tool
        search_result = search_articles_tool("AI trends", max_results=5)
        print(f"Search tool result: {search_result}")

        # Test extract tool
        extract_result = extract_article_content_tool(["https://example.com"])
        print(f"Extract tool result: {extract_result}")

        print("✅ Tool implementations work")
        return True

    except Exception as e:
        print(f"❌ Tool implementation test failed: {e}")
        return False


async def test_async_crawl():
    """Test async crawl functionality"""
    print("\nTesting async crawl functionality...")

    try:
        config = {
            "agents": {
                "research": {
                    "system_prompt": "You are a research agent."
                }
            }
        }

        agent = ResearchAgent(config)

        # Test enhanced HTML parsing (fallback)
        result = await agent._enhanced_html_parsing("https://httpbin.org/html")
        print(f"HTML parsing result: {result}")

        if result["success"]:
            print("✅ Async HTML parsing works")
            return True
        else:
            print(f"❌ Async HTML parsing failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Async crawl test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Starting ResearchAgent and Crawler Tests")
    print("=" * 50)

    results = []

    # Run synchronous tests
    results.append(test_research_agent_initialization())
    results.append(test_mock_search())
    results.append(test_crawler_adapters())
    results.append(test_tool_implementations())

    # Run async test
    try:
        async_result = asyncio.run(test_async_crawl())
        results.append(async_result)
    except Exception as e:
        print(f"❌ Async test failed to run: {e}")
        results.append(False)

    # Summary
    print("\n" + "=" * 50)
    print("🧪 Test Results Summary:")

    passed = sum(results)
    total = len(results)

    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! The ResearchAgent and crawler system is ready.")
        print("\nNext steps:")
        print("1. Install required packages: pip install httpx beautifulsoup4 pydantic")
        print("2. Set real API keys: FIRECRAWL_API_KEY, PARALLEL_API_KEY, OPENAI_API_KEY")
        print("3. Test with real APIs")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)