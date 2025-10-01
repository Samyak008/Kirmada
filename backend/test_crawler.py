"""
Comprehensive test script for the crawler system with Firecrawl and Parallel.ai.

This script tests the full crawler functionality including API integrations,
error handling, and performance comparisons.
"""

import asyncio
import os
import time
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Set mock API keys for testing (replace with real keys when available)
os.environ["FIRECRAWL_API_KEY"] = os.getenv("FIRECRAWL_API_KEY", "mock_firecrawl_key")
os.environ["PARALLEL_API_KEY"] = os.getenv("PARALLEL_API_KEY", "mock_parallel_key")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "mock_openai_key")

from crawlers.firecrawl_adapter import FirecrawlAdapter
from crawlers.parallelai_adapter import ParallelAIAdapter
from crawlers.base import CrawlOptions, CrawlResult


class CrawlerTester:
    """Test harness for crawler adapters"""

    def __init__(self):
        self.adapters = {
            'firecrawl': FirecrawlAdapter(),
            'parallel_ai': ParallelAIAdapter()
        }

    async def test_adapter(self, adapter_name: str, url: str, options: CrawlOptions) -> Dict[str, Any]:
        """Test a specific adapter"""
        adapter = self.adapters[adapter_name]

        start_time = time.time()
        try:
            result = await adapter.crawl(url, options)
            elapsed = time.time() - start_time

            return {
                'adapter': adapter_name,
                'success': result.success,
                'elapsed_seconds': elapsed,
                'title': result.title,
                'content_length': len(result.text) if result.text else 0,
                'images_count': len(result.images),
                'error': result.error_message,
                'provider': result.metadata.get('provider', 'unknown')
            }

        except Exception as e:
            elapsed = time.time() - start_time
            return {
                'adapter': adapter_name,
                'success': False,
                'elapsed_seconds': elapsed,
                'error': str(e)
            }

    async def compare_adapters(self, urls: List[str], options: CrawlOptions) -> Dict[str, Any]:
        """Compare performance of all adapters on multiple URLs"""
        results = {}

        for url in urls:
            print(f"\n🔍 Testing URL: {url}")
            url_results = {}

            for adapter_name in self.adapters.keys():
                print(f"  Testing {adapter_name}...")
                result = await self.test_adapter(adapter_name, url, options)
                url_results[adapter_name] = result

                status = "✅" if result['success'] else "❌"
                print(f"    {status} {adapter_name}: {result.get('elapsed_seconds', 0):.2f}s")

            results[url] = url_results

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a performance report"""
        report = []
        report.append("📊 Crawler Performance Report")
        report.append("=" * 50)

        for url, url_results in results.items():
            report.append(f"\n🔗 URL: {url}")
            report.append("-" * 30)

            for adapter_name, result in url_results.items():
                status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
                elapsed = result.get('elapsed_seconds', 0)
                content_len = result.get('content_length', 0)

                report.append(f"{adapter_name.upper()}: {status}")
                report.append(f"  Time: {elapsed:.2f}s")
                report.append(f"  Content: {content_len} chars")

                if result.get('error'):
                    report.append(f"  Error: {result['error']}")

                if result.get('title'):
                    report.append(f"  Title: {result['title'][:50]}...")

        return "\n".join(report)


async def test_crawler_system():
    """Main test function"""
    print("🧪 Testing Crawler System")
    print("=" * 50)

    tester = CrawlerTester()

    # Test URLs (using reliable test endpoints)
    test_urls = [
        "https://httpbin.org/html",  # Simple HTML page
        "https://example.com",       # Basic example page
    ]

    # Test options
    options = CrawlOptions(
        depth=1,
        js_render=False,  # Disable JS rendering for faster testing
        timeout_s=10,
        crawl_scope="page",
        extract_images=False,  # Disable image extraction for faster testing
        extract_metadata=True
    )

    print(f"Test Configuration:")
    print(f"  URLs: {len(test_urls)}")
    print(f"  Adapters: {list(tester.adapters.keys())}")
    print(f"  Options: {options}")
    print()

    # Run comparison
    results = await tester.compare_adapters(test_urls, options)

    # Generate and print report
    report = tester.generate_report(results)
    print("\n" + report)

    # Summary statistics
    print("\n📈 Summary Statistics:")
    print("-" * 30)

    total_tests = len(test_urls) * len(tester.adapters)
    successful_tests = sum(
        1 for url_results in results.values()
        for result in url_results.values()
        if result['success']
    )

    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")

    if successful_tests > 0:
        avg_times = {}
        for adapter_name in tester.adapters.keys():
            times = [
                result.get('elapsed_seconds', 0)
                for url_results in results.values()
                for result in url_results.values()
                if result['adapter'] == adapter_name and result['success']
            ]
            if times:
                avg_times[adapter_name] = sum(times) / len(times)

        if avg_times:
            print("\nAverage Response Times:")
            for adapter, avg_time in avg_times.items():
                print(f"  {adapter}: {avg_time:.2f}s")

    return successful_tests > 0


async def test_individual_adapters():
    """Test each adapter individually with detailed output"""
    print("\n🔧 Individual Adapter Tests")
    print("=" * 50)

    adapters = {
        'firecrawl': FirecrawlAdapter(),
        'parallel_ai': ParallelAIAdapter()
    }

    test_url = "https://httpbin.org/html"
    options = CrawlOptions(
        depth=1,
        js_render=False,
        timeout_s=15,
        extract_images=False,
        extract_metadata=True
    )

    for name, adapter in adapters.items():
        print(f"\nTesting {name.upper()}:")
        print("-" * 20)

        try:
            result = await adapter.crawl(test_url, options)

            print(f"Success: {result.success}")
            print(f"Title: {result.title[:50] if result.title else 'None'}")
            print(f"Content Length: {len(result.text) if result.text else 0}")
            print(f"Images Found: {len(result.images)}")
            print(f"Provider: {result.metadata.get('provider', 'unknown')}")

            if result.error_message:
                print(f"Error: {result.error_message}")

            # Test health check
            try:
                healthy = await adapter.health_check()
                print(f"Health Check: {'✅' if healthy else '❌'}")
            except Exception as e:
                print(f"Health Check Failed: {e}")

        except Exception as e:
            print(f"❌ Test failed: {e}")


def main():
    """Main entry point"""
    print("🚀 Crawler System Test Suite")
    print("=" * 60)

    async def run_tests():
        # Run individual adapter tests
        await test_individual_adapters()

        # Run comparative tests
        success = await test_crawler_system()

        print("\n" + "=" * 60)
        if success:
            print("🎉 Crawler tests completed successfully!")
            print("\n💡 Recommendations:")
            print("1. Set real API keys for production testing")
            print("2. Test with real websites for comprehensive validation")
            print("3. Monitor API usage and costs")
            print("4. Consider implementing caching for repeated requests")
        else:
            print("⚠️  Some tests failed. Check API keys and network connectivity.")
            print("\n🔧 Troubleshooting:")
            print("1. Verify API keys are set correctly")
            print("2. Check network connectivity")
            print("3. Ensure required packages are installed")
            print("4. Try with different test URLs")

        return success

    # Run async tests
    try:
        success = asyncio.run(run_tests())
        return success
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)