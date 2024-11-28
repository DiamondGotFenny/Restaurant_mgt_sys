from enum import Enum
from typing import Dict, Any
from vectorDB_Agent.vectorDB_Engine import VectorDBEngine
from text_to_sql.text_to_sql_engine import TextToSQLEngine
from openai import AzureOpenAI
import os
import json
import csv
from logger_config import logger
current_dir = os.path.dirname(os.path.realpath(__file__))
log_file_path = os.path.join(current_dir,"logs" "query_router.log")

class QueryType(Enum):
    DOCUMENT_BASED = "document_based"
    DATABASE_BASED = "database_based"
    OFF_TOPIC = "off_topic"

class QueryRouter:
    def __init__(self, docs_metadata_path: str, table_desc_path: str, log_file_path: str=log_file_path):
        """
        Initialize QueryRouter with paths to metadata files
        
        Args:
            docs_metadata_path (str): Path to the JSON file containing document metadata
            table_desc_path (str): Path to the CSV file containing table descriptions
        """
        self.vector_engine = VectorDBEngine()
        self.sql_engine = TextToSQLEngine(log_file_path)
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.logger = logger(log_file_path)
        # Load metadata from files
        self.docs_metadata = self._load_docs_metadata(docs_metadata_path)
        self.table_descriptions = self._load_table_descriptions(table_desc_path)
        self.logger.info("QueryRouter initialized successfully.")
        
    def _load_docs_metadata(self, file_path: str) -> Dict:
        """Load document metadata from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data
        except Exception as e:
            self.logger.error(f"Error loading document metadata: {e}")
            return {}

    def _load_table_descriptions(self, file_path: str) -> Dict:
        """Load table descriptions from CSV file"""
        table_desc = {}
        try:
            with open(file_path, 'r') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    table_desc[row['Table']] = row['Description']
            return table_desc
        except Exception as e:
            self.logger.error(f"Error loading table descriptions: {e}")
            return {}

    def _determine_search_engine(self, query: str) -> Dict[str, Any]:
        """Determine which search engine to use for the query"""
        self.logger.info("Determining search engine for the query.")
        prompt = f"""As Sophie, the NYC restaurant expert, determine which data source would be best to answer this query.

Available Data Sources:

DOCUMENT-BASED SOURCES (Use for subjective information, reviews, guides, blog posts):
{json.dumps(self.docs_metadata, indent=2)}

DATABASE TABLES (Use for structured data like inspections, menus, ratings):
{json.dumps(self.table_descriptions, indent=2)}

Return a JSON with these fields:
{{
    "query_type": string,    // "document_based" or "database_based"
    "reasoning": string,     // explanation of the choice
    "relevant_sources": [    // specific documents or tables that would be useful
        string
    ]
}}"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_4o"),
                messages=messages,
                response_format={ "type": "json_object" }
            )
            self.logger.info(f"---------Search engine determined successfully---------------")
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Error in search engine determination: {e}")
            return {"query_type": None, "reasoning": "Error in analysis", "relevant_sources": []}

    def _check_relevance(self, query: str) -> Dict[str, Any]:
        """Check if the query is relevant to NYC dining."""
        self.logger.info("Checking relevance of the query.")
        prompt = """Determine if this query is relevant to NYC dining.

        RELEVANT TOPICS INCLUDE:
        - NYC restaurants and dining establishments
        - Restaurant reviews, ratings, or recommendations in NYC
        - Menu items, prices, or cuisine types in NYC restaurants
        - Restaurant locations, neighborhoods, or accessibility in NYC
        - Restaurant safety, inspections, or ratings in NYC
        - Specific NYC restaurants or dining experiences

        Return ONLY a JSON with these fields:
        {
            "is_relevant": boolean,
            "reasoning": string
        }"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_4o"),
                messages=messages,
                response_format={ "type": "json_object" }
            )
            self.logger.info("Relevance check completed.")
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Error in relevance check: {e}")
            return {"is_relevant": False, "reasoning": "Error in analysis"}
    
    def route_query(self, query: str) -> str:
        """
        Route the query and return both relevance status and response data.
        
        Returns:
            Dict with structure:
            {
                "is_relevant": boolean,
                "response": string,  # Either context data or None
                "reasoning": string  # Explanation of routing decision
            }
        """
        self.logger.info("Routing the query.")
        # First check relevance
        relevance_result = self._check_relevance(query)
        self.logger.info(f"\n-------------Relevance result: {relevance_result}-----------------\n")
        
        if not relevance_result.get("is_relevant"):
            return {
                "is_relevant": False,
                "response": None,
                "reasoning": relevance_result.get("reasoning", "Query is not related to NYC dining")
            }
        
        # If relevant, determine which search engine to use and get response
        try:
            engine_analysis = self._determine_search_engine(query)
            self.logger.info(f"\n-------------Engine analysis result: {engine_analysis}-----------------\n")
            query_type = engine_analysis.get("query_type")
            
            if query_type == "document_based":
                response = self.vector_engine.qa_chain(query)
            elif query_type == "database_based":
                generated_output = self.sql_engine.process_query(query)
                response = generated_output['result']
            else:
                response = "I'm having trouble finding the right information for your query."
                
            return {
                "is_relevant": True,
                "response": response,
                "reasoning": engine_analysis.get("reasoning", "")
            }
            
        except Exception as e:
            self.logger.error(f"Error in query routing: {e}")
            return {
                "is_relevant": True,
                "response": "I apologize, but I'm having trouble accessing the information you need.",
                "reasoning": "Error occurred during data retrieval"
            }