from enum import Enum
from typing import Dict, Any, List
from vectorDB_Agent.vectorDB_Engine import VectorDBEngine
from text_to_sql.text_to_sql_engine import TextToSQLEngine
from openai import AzureOpenAI
import os
import json
import csv
from logger_config import setup_logger
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
        self.vector_engine = VectorDBEngine(log_file_path)
        self.sql_engine = TextToSQLEngine(log_file_path)
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.logger = setup_logger(log_file_path)
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
        """Determine which search engine(s) to use for the query"""
        self.logger.info("Determining search engine(s) for the query.")
        prompt = f"""As Sophie, the NYC restaurant expert, analyze which data sources would be needed to provide a complete answer to this query.

        Available Data Sources:

        DOCUMENT-BASED SOURCES (Use for subjective information, reviews, guides, blog posts):
        {json.dumps(self.docs_metadata, indent=2)}

        DATABASE TABLES (Use for structured data like inspections, menus, ratings):
        {json.dumps(self.table_descriptions, indent=2)}

        Consider that many queries might need both types of sources for a complete answer. 

        For compound queries (questions about multiple different things), break them down into separate components.
        
        Return a JSON with these fields:
        {{
            "search_strategy": {{
                "use_documents": boolean,    // Whether to search document sources
                "use_database": boolean,     // Whether to search database
                "primary_source": string     // "hybrid", "documents", or "database"
            }},
            "reasoning": string,             // Explanation of the choice
            "relevant_sources": {{           // Specific sources that would be useful
                "documents": [string],       // Relevant document sources
                "tables": [string]           // Relevant database tables
            }},
            "query_components": {{           // Break down complex queries
                "is_compound": boolean,      // Whether this is a compound query
                "components": [              // Array of separate query components
                    {{
                        "sub_query": string,           // The individual query
                        "document_aspect": string | null,        // What to look for in documents, the rewrite of the sub_query
                        "database_aspect": string | null         // What to look for in database, the rewrite of the sub_query
                    }}
                ]
            }}
        }}"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_4o"),
                messages=messages,
                response_format={ "type": "json_object" },
                temperature=0.1
            )
            self.logger.info("Search engine(s) determined successfully")
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Error in search engine determination: {e}")
            return {
                "search_strategy": {
                    "use_documents": True,
                    "use_database": True,
                    "primary_source": "hybrid"
                },
                "reasoning": "Error in analysis, using all sources",
                "relevant_sources": {
                    "documents": [],
                    "tables": []
                },
                "query_components": {
                     "is_compound": False,
                    "document_aspect": None,
                    "database_aspect": None,
                    "components": [      { "sub_query": None,
                        "document_aspect": None,
                        "database_aspect": None}       
                       ]                
                }
            }


    def _combine_search_results(self, doc_response: str, db_response: str) -> str:
        """Combine results from different search engines intelligently"""
        prompt = f"""As Sophie, combine these search results into a coherent response.

        Document-based results:
        {doc_response}

        Database results:
        {db_response}

        Combine these results into a natural, conversational response that:
        1. Integrates information from both sources seamlessly
        2. Eliminates redundancy
        3. Presents information in a logical order
        4. Maintains a friendly, helpful tone

        Return ONLY the combined response, no explanations."""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Please combine these results."}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_4o"),
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error in combining results: {e}")
            return f"{doc_response}\n\nAdditional information:\n{db_response}"


    def _check_relevance(self, query: str) -> Dict[str, Any]:
        """Check if the query is relevant to NYC dining."""
        self.logger.info("Checking relevance of the query.")
        prompt = """Determine if this query is relevant to NYC dining.

        RELEVANT TOPICS INCLUDE:
        - NYC restaurants and dining establishments
        - Restaurant reviews, ratings, or recommendations in NYC
        - Food type or cuisine or specific dishes that users show interest in
        - Menu items, prices, or cuisine types in NYC restaurants
        - Restaurant locations, neighborhoods, or accessibility in NYC
        - Restaurant safety, inspections, or ratings in NYC
        - Restaurant business hours, reservations, or delivery options in NYC
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
                response_format={ "type": "json_object" },
                temperature=0.1
            )
            self.logger.info("Relevance check completed.")
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Error in relevance check: {e}")
            return {"is_relevant": False, "reasoning": "Error in analysis"}
    def _combine_component_responses(self, component_responses: List[str], original_query: str) -> str:
        """Combine responses from different components of a compound query"""
        prompt = f"""As Sophie, combine these separate response components into a coherent answer.

        Original Query: {original_query}

        Component Responses:
        {json.dumps(component_responses, indent=2)}

        Create a unified response that:
        1. Addresses all parts of the original query
        2. Maintains clear separation between different types of recommendations
        3. Flows naturally in conversation
        4. Preserves all relevant information
        5. Maintains a friendly, helpful tone

        Return ONLY the combined response, no explanations."""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Please combine these results."}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_4o"),
                messages=messages,
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error in combining component responses: {e}")
            return "\n\n".join(component_responses)
        
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route the query and return both relevance status and response data.
        Supports hybrid searches across multiple data sources and compound queries.
        """
        self.logger.info("Routing the query.")
        # First check relevance
        relevance_result = self._check_relevance(query)
        self.logger.info(f"\nQuery: {query}\nRelevance result: {relevance_result}")
        
        if not relevance_result.get("is_relevant"):
            return {
                "is_relevant": False,
                "response": None,
                "reasoning": relevance_result.get("reasoning", "Query is not related to NYC dining")
            }
        
        try:
            # Determine search strategy
            engine_analysis = self._determine_search_engine(query)
            self.logger.info(f"\n-------------Query: {query}\nEngine analysis: {engine_analysis}-----------------\n")
            
            search_strategy = engine_analysis["search_strategy"]
            query_components = engine_analysis["query_components"]
            
            # Handle compound queries
            if query_components.get("is_compound"):
                component_responses = []
                
                # Process each component separately
                for component in query_components["components"]:
                    doc_response = None
                    db_response = None
                    
                    if search_strategy["use_documents"]:
                        doc_aspect = component["document_aspect"] or component["sub_query"]
                        doc_response = self.vector_engine.qa_chain(doc_aspect)
                        
                    if search_strategy["use_database"]:
                        db_aspect = component["database_aspect"] or component["sub_query"]
                        db_result = self.sql_engine.process_query(db_aspect)
                        db_response = db_result['result']
                    
                    # Combine results for this component
                    if search_strategy["primary_source"] == "hybrid":
                        if doc_response and db_response:
                            component_response = self._combine_search_results(doc_response, db_response)
                        else:
                            component_response = doc_response or db_response
                    else:
                        component_response = doc_response if search_strategy["primary_source"] == "documents" else db_response
                    
                    component_responses.append(component_response)
                
                # Combine all component responses
                final_response = self._combine_component_responses(component_responses, query)
                
            else:
                # Handle single query (existing logic)
                doc_response = None
                db_response = None
                
                if search_strategy["use_documents"]:
                    doc_aspect = query_components.get("document_aspect") or query
                    doc_response = self.vector_engine.qa_chain(doc_aspect)
                    
                if search_strategy["use_database"]:
                    db_aspect = query_components.get("database_aspect") or query
                    db_result = self.sql_engine.process_query(db_aspect)
                    db_response = db_result['result']
                
                if search_strategy["primary_source"] == "hybrid":
                    if doc_response and db_response:
                        final_response = self._combine_search_results(doc_response, db_response)
                    else:
                        final_response = doc_response or db_response
                else:
                    final_response = doc_response if search_strategy["primary_source"] == "documents" else db_response
            
            return {
                "is_relevant": True,
                "response": final_response,
                "reasoning": engine_analysis.get("reasoning", ""),
                "search_strategy": search_strategy
            }
                
        except Exception as e:
            self.logger.error(f"Error in query routing: {e}")
            return {
                "is_relevant": True,
                "response": "I apologize, but I'm having trouble accessing the information you need.",
                "reasoning": f"Error occurred during data retrieval: {str(e)}"
            }