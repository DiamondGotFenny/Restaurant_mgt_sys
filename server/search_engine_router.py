from typing import Dict, Any, List
from vectorDB_Agent.vectorDB_Engine import VectorDBEngine
from text_to_sql.text_to_sql_engine import TextToSQLEngine
from openai import AzureOpenAI
import os
import json
import csv
from logger_config import setup_logger

class SearchEngineRouterV1:
    """Traditional implementation using multiple LLM calls and separate processing steps"""
    def __init__(self, docs_metadata_path: str, table_desc_path: str, log_file_path: str):
        self.vector_engine = VectorDBEngine(log_file_path)
        self.sql_engine = TextToSQLEngine(log_file_path)
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.logger = setup_logger(log_file_path)
        self.docs_metadata = self._load_docs_metadata(docs_metadata_path)
        self.table_descriptions = self._load_table_descriptions(table_desc_path)
        
    def _load_docs_metadata(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading document metadata: {e}")
            return {}

    def _load_table_descriptions(self, file_path: str) -> Dict:
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
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process the query and return response data."""
        try:
            engine_analysis = self._determine_search_engine(query)
            self.logger.info(f"\n-------------Query: {query}\nEngine analysis: {engine_analysis}-----------------\n")
            
            search_strategy = engine_analysis["search_strategy"]
            query_components = engine_analysis["query_components"]
            
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
                "response": final_response,
                "reasoning": engine_analysis.get("reasoning", ""),
                "search_strategy": search_strategy
            }
                
        except Exception as e:
            self.logger.error(f"Error in query processing: {e}")
            return {
                "response": "I apologize, but I'm having trouble accessing the information you need.",
                "reasoning": f"Error occurred during data retrieval: {str(e)}"
            }
class SearchEngineRouterV2:
    """Enhanced implementation using function calling for streamlined processing"""
    def __init__(self, docs_metadata_path: str, table_desc_path: str, log_file_path: str):
        self.vector_engine = VectorDBEngine(log_file_path)
        self.sql_engine = TextToSQLEngine(log_file_path)
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.logger = setup_logger(log_file_path)
        self.docs_metadata = self._load_docs_metadata(docs_metadata_path)
        self.table_descriptions = self._load_table_descriptions(table_desc_path)
        self.logger.info("Function calling Search Engine Router initialized successfully.")
    def _load_docs_metadata(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading document metadata: {e}")
            return {}

    def _load_table_descriptions(self, file_path: str) -> Dict:
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

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process the query using function calling"""
        try:
            functions = [
                {
                    "name": "process_and_respond",
                    "description": "Process the query and generate response using available search engines",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_plan": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "engine_type": {"type": "string", "enum": ["vector_search", "sql_search"]},
                                        "search_question": {
                                            "type": "string",
                                            "description": "The search question in natural language, not SQL.'"
                                        },
                                        "reasoning": {"type": "string"}
                                    },
                                    "required": ["engine_type", "search_question", "reasoning"]
                                }
                            },
                            "strategy_summary": {
                                "type": "object",
                                "properties": {
                                    "use_documents": {"type": "boolean"},
                                    "use_database": {"type": "boolean"},
                                    "primary_source": {"type": "string", "enum": ["hybrid", "documents", "database"]},
                                    "reasoning": {"type": "string"}
                                },
                                "required": ["use_documents", "use_database", "primary_source", "reasoning"]
                            }
                        },
                        "required": ["search_plan", "strategy_summary"]
                    }
                }
            ]

            messages = [
                {
                    "role": "system",
                    "content": f"""You are QueryPlanner, an intelligent search strategist specialized in organizing and planning information retrieval from multiple data sources.  You have access to two search engines:

                    1. Vector Search Engine (for documents):
                    - Purpose: Searches through unstructured text documents
                    - Available Documents: {json.dumps(self.docs_metadata, indent=2)}
                    - Best for: Reviews, subjective information, detailed descriptions
                    - Function: vector_search(question) -> returns relevant text passages

                    2. SQL Search Engine (for database):
                    - Purpose: Searches structured database tables
                    - Available Tables: {json.dumps(self.table_descriptions, indent=2)}
                    - Best for: Factual data, ratings, inspections, objective information
                    - Function: sql_search(question) -> returns structured data

                    Your task:
                    1. Analyze the user query
                    2. Create a search plan specifying which engines to use and what to search for
                    3. The search results will be automatically fetched and combined based on your plan

                    Important:
                    - Always use natural language questions in your search plan, NOT SQL queries
                    - The actual conversion to SQL will be handled by the search engines

                    Remember:
                    - You can use both engines if needed
                    - Break down complex queries into specific searchable components
                    - Keep all queries in natural language form
                    - The search engines will handle the appropriate conversions internally"""
                },
                {"role": "user", "content": query}
            ]

            # Get the search plan from GPT-4
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_4o"),
                temperature=0.3,
                messages=messages,
                functions=functions,
                function_call={"name": "process_and_respond"}
            )

            plan = json.loads(response.choices[0].message.function_call.arguments)
            search_results = []

            # Execute the search plan
            for step in plan["search_plan"]:
                if step["engine_type"] == "vector_search":
                    result = self.vector_engine.qa_chain(step["search_question"])
                else:  # sql_search
                    result = self.sql_engine.process_query(step["search_question"])["result"]
                search_results.append({
                    "engine_type": step["engine_type"],
                    "search_question": step["search_question"],
                    "result": result,
                    "reasoning": step["reasoning"]
                })

            self.logger.info(f"\ncombine_messages string_result: {json.dumps(search_results, indent=2)} \n")
            

            return {
                "response": search_results,
                "reasoning": plan["strategy_summary"]["reasoning"],
                "search_strategy": {
                    "use_documents": plan["strategy_summary"]["use_documents"],
                    "use_database": plan["strategy_summary"]["use_database"],
                    "primary_source": plan["strategy_summary"]["primary_source"]
                }
            }

        except Exception as e:
            self.logger.error(f"Error in query processing: {e}")
            return {
                "response": "I apologize, but I'm having trouble accessing the information you need.",
                "reasoning": f"Error occurred during data retrieval: {str(e)}"
            }



