from typing import Dict, Any
import os
from relevance_checker import RelevanceChecker
from search_engine_router import SearchEngineRouterV2

current_dir = os.path.dirname(os.path.realpath(__file__))
log_file_path = os.path.join(current_dir, "logs", "query_router.log")

class QueryOrchestrator:
    def __init__(self, docs_metadata_path: str, table_desc_path: str, log_file_path: str=log_file_path):
        self.relevance_checker = RelevanceChecker(log_file_path)
        self.search_engine_router = SearchEngineRouterV2(docs_metadata_path, table_desc_path, log_file_path)

    def route_query(self, query: str) -> Dict[str, Any]:
        """Route the query and return both relevance status and response data."""
        # Check relevance
        relevance_result = self.relevance_checker.check_relevance(query)
        
        if not relevance_result.get("is_relevant"):
            return {
                "is_relevant": False,
                "response": None,
                "reasoning": relevance_result.get("reasoning", "Query is not related to NYC dining")
            }
        
        # Process query through search engines
        search_result = self.search_engine_router.process_query(query)
        
        return {
            "is_relevant": True,
            "response": search_result["response"],
            "reasoning": search_result["reasoning"],
            "search_strategy": search_result.get("search_strategy")
        }