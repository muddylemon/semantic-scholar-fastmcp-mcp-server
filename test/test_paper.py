import unittest
import asyncio
import os
from typing import Optional, List, Dict

from .test_utils import make_request, create_error_response, ErrorType, Config

class TestPaperTools(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # You can set your API key here for testing
        os.environ["SEMANTIC_SCHOLAR_API_KEY"] = ""  # Optional
        
        # Create event loop for async tests
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Sample paper IDs for testing
        self.sample_paper_id = "649def34f8be52c8b66281af98ae884c09aef38b"
        self.sample_paper_ids = [
            self.sample_paper_id,
            "ARXIV:2106.15928"
        ]

    def tearDown(self):
        """Clean up after tests"""
        self.loop.close()

    def run_async(self, coro):
        """Helper to run async functions in tests"""
        return self.loop.run_until_complete(coro)

    async def async_test_with_delay(self, coro):
        """Helper to run async tests with delay to handle rate limiting"""
        await asyncio.sleep(1)  # Add 1 second delay between tests
        return await coro

    def test_paper_relevance_search(self):
        """Test paper relevance search functionality"""
        # Test basic search
        result = self.run_async(self.async_test_with_delay(make_request(
            "/paper/search",
            params={
                "query": "quantum computing",
                "fields": "title,abstract,year"
            }
        )))
        self.assertIn("data", result)
        self.assertIn("total", result)
        
        # Test with filters
        result = self.run_async(self.async_test_with_delay(make_request(
            "/paper/search",
            params={
                "query": "machine learning",
                "fields": "title,year",
                "minCitationCount": 100,
                "year": "2020-2023"
            }
        )))
        self.assertIn("data", result)

    def test_paper_bulk_search(self):
        """Test paper bulk search functionality"""
        result = self.run_async(self.async_test_with_delay(make_request(
            "/paper/search/bulk",
            params={
                "query": "neural networks",
                "fields": "title,year,authors",
                "sort": "citationCount:desc"
            }
        )))
        self.assertIn("data", result)

    def test_paper_details(self):
        """Test paper details functionality"""
        result = self.run_async(self.async_test_with_delay(make_request(
            f"/paper/{self.sample_paper_id}",
            params={
                "fields": "title,abstract,year,authors"
            }
        )))
        self.assertIn("paperId", result)
        self.assertIn("title", result)

    def test_paper_batch_details(self):
        """Test batch paper details functionality"""
        result = self.run_async(self.async_test_with_delay(make_request(
            "/paper/batch",
            method="POST",
            params={"fields": "title,year,authors"},
            json={"ids": self.sample_paper_ids}
        )))
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), len(self.sample_paper_ids))

if __name__ == '__main__':
    unittest.main()
