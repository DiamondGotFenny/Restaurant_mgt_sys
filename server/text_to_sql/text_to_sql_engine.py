#text_to_sql_engine.py
import os
import json
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from fastapi import HTTPException
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from  logger_config import setup_logger
from  text_to_sql.dynamic_examples_store import DynamicExamplesStore
import json
from sqlalchemy import create_engine, text

class RequiredTables(BaseModel):
    """Output from Step 2"""
    tables: List[str] = Field(description="Tables that are directly needed for the main query")

class TableSchema(BaseModel):
    """Schema information for a single table"""
    columns: dict = Field(description="Column definitions")
    primary_key: List[str] = Field(description="Primary key columns")
    foreign_keys: Optional[dict] = Field(description="Foreign key relationships")

class TextToSQLEngine:
    def __init__(self,log_file: str = "text_to_sql_engine.log"):
        """Initialize the TextToSQLEngine with necessary configurations and components."""
        # Load environment variables
        load_dotenv(find_dotenv())

        # Define the base directory and paths
        self.BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        
        # Setup logger
        self.logger = setup_logger(log_file)
        
        # Load environment variables
        self._load_env_variables()
        
        # Initialize components
        self._initialize_components()
        
        # Setup prompts
        self._setup_prompts()
        self.logger.info("TextToSQLEngine initialized successfully")

    def _load_env_variables(self):
        """Load environment variables."""
        self.db_uri = os.getenv("NEON_RESTARANT_DB_STR")
        self.azure_openai_api_key = os.getenv("OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("OPENAI_API_BASE")
        self.azure_openai_deployment_mini = os.getenv("OPENAI_MODEL_4OMINI")
        self.azure_openai_deployment = os.getenv("OPENAI_MODEL_4O")
        self.azure_api_version = os.getenv("AZURE_API_VERSION")
        self.azure_openai_embedding_deployment = os.getenv("OPENAI_EMBEDDING_MODEL")

    def _initialize_components(self):
        """Initialize database and AI models."""
        self.logger.info("Initializing database and AI models")
        
        # Initialize database engine
        self.engine = create_engine(self.db_uri)
        self.logger.info("Database initialized successfully")

        # Initialize Azure OpenAI models
        self.model_4o_mini = AzureChatOpenAI(
            api_key=self.azure_openai_api_key,
            azure_endpoint=self.azure_openai_endpoint,
            deployment_name=self.azure_openai_deployment_mini,
            api_version=self.azure_api_version,
            temperature=0,
            max_tokens=3000
        )

        self.model_4o = AzureChatOpenAI(
            api_key=self.azure_openai_api_key,
            azure_endpoint=self.azure_openai_endpoint,
            deployment_name=self.azure_openai_deployment,
            api_version=self.azure_api_version,
            temperature=0.2,
            max_tokens=3000
        )
         # Get the absolute path of the current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the absolute path to dynamic_examples.json
        examples_file_path = os.path.join(current_dir, 'dynamic_examples.json')

        # Construct the absolute path to the dynamic_examples_vectorDB directory
        persist_dir_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'dynamic_examples_vectorDB'))
        
        # Initialize DynamicExamplesStore
        self.examples_store = DynamicExamplesStore(
            examples_file=examples_file_path,
            persist_directory=persist_dir_path,
            azure_openai_api_key=self.azure_openai_api_key,
            azure_openai_endpoint=self.azure_openai_endpoint,
            azure_openai_embedding_deployment=self.azure_openai_embedding_deployment
        )
        
        
    def execute_query(self, query: str) -> str:
        """Execute SQL query and return results in JSON format."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                # Convert result to list of dictionaries
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in result]
                format_data = json.dumps(data, default=str, indent=2)
                self.logger.info(f"Query executed successfully: {format_data}")
                return format_data
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error executing query: {error_msg}")
            
            # Check for specific types of SQL errors
            if "UndefinedColumn" in error_msg:
                return f"Error: Column referenced in query does not exist. Details: {error_msg}"
            elif "UndefinedTable" in error_msg:
                return f"Error: Table referenced in query does not exist. Details: {error_msg}"
            elif "syntax error" in error_msg.lower():
                return f"Error: SQL syntax error in query. Details: {error_msg}"
            else:
                return f"Error executing query: {error_msg}"
    
    def _setup_prompts(self):
        """Setup all prompts for the multi-step process"""
        
        #load table descriptions 
        table_descriptions_path = os.path.join(self.BASE_DIR, 'database_table_descriptions.csv')
        try:
            df = pd.read_csv(table_descriptions_path)
            # Convert to a formatted string with table name and description
            self.table_descriptions = "\n\n".join([
                f"Table: {row['Table']}\n"
                f"Description: {row['Description']}"
                for _, row in df.iterrows()
            ])
        except Exception as e:
            self.logger.error(f"Error loading table descriptions: {str(e)}")
            self.table_descriptions = "Error loading table descriptions"
            
        # Step 1: Analysis Prompt     
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL query analyzer specializing in restaurant and dining data analysis. Your task is to analyze natural language questions and provide a strategy for query construction.

            # Inputs:
            1. A user's question about restaurants/dining
            2. You have access to the following a brief list of available tables and their high-level descriptions: {table_descriptions}

            # Structure your response with only an "Analysis" section and a "Query Strategy" section, in the following format:
            **Analysis**:
            [Explain the logic and approach needed to answer the question]

            **Query Strategy** :
            1. [First step in query construction]
            2. [Second step]
            ...
            [Each step should be specific and actionable]
            --------------------------------

            # Requirements:
            - let's think step by step.
            - Focus on the logic rather than specific SQL syntax.
            - Identify the most relevant tables that can answer the user's question, considering relationships.
            - try to access multiple tables to get the required information, and consider the join conditions.
            - Consider data aggregations, groupings, and orderings needed.
            - Think about potential NULL values or data quality issues.
            - Consider performance implications.
            - Do not handle 'New York' or 'New York City' or 'NYC' that metioned in the question, as the default location is New York City.
            
            Here are some relevant examples:
            {examples}"""),
            ("human", "{question}")
        ])

        # Step 2: Table Extraction Prompt
        self.table_extraction_prompt = PromptTemplate.from_template("""
            You are a table identifier for a database system. Your task is to extract the exact table names needed based on the previous analysis and query strategy.

            Input analysis:
            {analysis}

            Output requirements:
            - Return ONLY a comma-separated list of table names
            - Include only tables that are absolutely necessary for the query
            - Don't include any explanations or additional text
            - Use exact table names as they appear in the database
            - Order tables based on their primary/dependency relationship

            Output format example:
            restaurant_inspections,restaurant_menu,restaurants_has_reviews
        """)

        # Step 3: Query Generation Prompt
        table_info_path = os.path.join(self.BASE_DIR, 'tables_info.json')
        try:
            with open(table_info_path, 'r') as f:
                table_info = json.load(f)
                self.all_table_schemas = {table['name']: table for table in table_info['tables']}
                
            self.query_generation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert PostgreSQL query generator. Generate a precise SQL query based on the provided strategy and table schemas.

                Requirements:
                1. Use only the necessary columns based on the strategy
                2. Include columns needed for joins, filters, and calculations
                3. Limit results to first 20 rows if the result is not already limited
                4. Use COALESCE for handling NULL values
                5. Use clear table aliases for readability
                6. Ensure proper JOIN conditions to avoid cartesian products
                7. Follow these join rules:
                    - Use exact name matching for most table joins
                    - Only use id-based joins when specifically required
                    - Use LEFT/RIGHT joins when data presence isn't guaranteed
                    - Consider case sensitivity in name-based joins
                8. Format the query with proper indentation
                9. End the query with a semicolon
                10. DO NOT include SQL markdown tags or comments
                11. When matching restaurant names between tables, ensure exact matching by:
                    - Removing any leading or trailing spaces (TRIM)
                    - Converting all letters to lowercase (LOWER)
                12. Check available columns and their examples in the table schemas before using them, don't use non-existing columns
                13. the query strategy may contain incorrect columns name or values, you should check the table schemas and correct them if any.
                
                The output should be only the PostgreSQL query that follows SQL best practices and meets all requirements."""),
                ("human", """Strategy: {strategy}
                Table Schemas: {table_schemas}
                Here are some relevant examples:
                {examples}""")
            ])

        except Exception as e:
            self.logger.error(f"Error in _setup_prompts step 3: {str(e)}")
            raise

        # Define answer prompt
        self.answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding PostgreSQL query, and PostgreSQL result, answer the user question.

            Question: {question}
            PostgreSQL Query: {query}
            PostgreSQL Result: {result}
            Answer: """
        )

    def _analyze_question(self, question: str) -> str:
        """Step 1: Analyze the question and generate strategy"""
        model = self.model_4o
        self.logger.info(f"-----------------_analyze_question question: {question} \n------------------\n")
        
        try:
            # Get relevant examples using DynamicExamplesStore
            relevant_examples = self.examples_store.get_relevant_examples(question, k=3)
            examples_text = ""
            for i, example in enumerate(relevant_examples, 1):
                examples_text += f"\nExample {i}:\nQuestion: {example['question']}\n"
                examples_text += f"Analysis: {example['analysis']}\n"
                examples_text += f"Query Strategy: {example['query_strategy']}\n"
                examples_text += "-" * 80 + "\n"
            
            self.logger.info(f"Dynamic Examples: {examples_text} \n-----------------\n")
            
            prompt_format = self.analysis_prompt.format_messages(
                table_descriptions=self.table_descriptions,
                examples=examples_text,
                question=question
            )
            
            result = model.invoke(prompt_format)
            return result.content
            
        except Exception as e:
            self.logger.error(f"Error in _analyze_question: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze question: {str(e)}"
            )

    def _extract_required_tables(self, analysis: str) -> RequiredTables:
        """Step 2: Extract required tables"""
        model = self.model_4o_mini.with_structured_output(RequiredTables)
        result = model.invoke(
            self.table_extraction_prompt.format(
                analysis=analysis,
            )
        )
        self.logger.info(f"_extract_required_tables result: {result}")
        return result
    
    def _load_table_schemas(self, required_tables: RequiredTables) -> dict:
        """Step 3: Load relevant table schemas"""
        schemas = {}
        try:
            for table in required_tables.tables:
                if table in self.all_table_schemas:
                    schemas[table] = self.all_table_schemas[table]
                else:
                    self.logger.warning(f"Schema not found for table: {table}")
            return schemas
        except Exception as e:
            self.logger.error(f"Error loading table schemas: {str(e)}")
            raise

    def _generate_sql_query(self, query: str, strategy: str, schemas: dict) -> str:
        """Step 4: Generate SQL query from natural language query."""
        try:
            # Get relevant examples for SQL query generation
            relevant_examples = self.examples_store.get_relevant_examples(query, k=2)
            examples_text = ""
            for i, example in enumerate(relevant_examples, 1):
                examples_text += f"\nExample {i}:\nQuestion: {example['question']}\n"
                examples_text += f"Query: {example.get('query', '')}\n"
                examples_text += "-" * 80 + "\n"

            formatted_prompt = self.query_generation_prompt.format_messages(
                strategy=strategy,
                table_schemas=json.dumps(schemas, indent=2),
                examples=examples_text
            )
            
            model = self.model_4o
            response = model.invoke(formatted_prompt)
            sql_query = response.content
            
            cleaned_query = self.clean_sql_query(sql_query)
            final_query = self.add_limit_to_query(cleaned_query)

            return final_query
            
        except Exception as e:
            self.logger.error(f"Error generating SQL query: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate SQL query: {str(e)}"
            )

    def clean_sql_query(self, query: str) -> str:
        """Clean SQL query by removing unnecessary SQL markers and fixing quote escaping."""
        query = query.replace("```sql", "").replace("```", "")
        query = query.replace("`", "")
        query = query.replace("\\'", "''")
        query = query.strip()
        if not query.endswith(';'):
            query += ';'
        return query
    
    def add_limit_to_query(self, sql_query: str, default_limit: int = 20) -> str:
        """Add LIMIT clause to SQL query if not present."""
        self.logger.debug(f"Processing query for LIMIT clause: {sql_query}")
        
        clean_query = sql_query.strip().rstrip(';')
        upper_query = clean_query.upper()
        
        try:
            if 'LIMIT' in upper_query:
                self.logger.debug("Query already contains LIMIT clause")
                return f"{clean_query};"
                
            if 'ORDER BY' in upper_query:
                return f"{clean_query} LIMIT {default_limit};"
            else:
                return f"{clean_query} LIMIT {default_limit};"
                
        except Exception as e:
            self.logger.error(f"Error processing LIMIT clause: {str(e)}")
            return f"{clean_query};"

    def _revise_sql_query(self, original_query: str, error_message: str, strategy: str, analysis: str, schemas: dict) -> str:
        """Use LLM to revise the SQL query based on error message and context."""
        try:
            revision_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a PostgreSQL expert. Your task is to fix SQL queries that have errors. 
                Use the provided context (analysis, strategy, and table schemas) to ensure the revised query maintains the original intent while fixing the errors.

                Requirements for the revised query:
                1. Fix the specific error mentioned in the error message
                2. Maintain consistency with the original analysis and strategy
                3. Use only columns that exist in the provided table schemas
                4. Follow proper PostgreSQL syntax and best practices
                5. Include appropriate JOIN conditions
                6. Handle NULL values appropriately
                7. Maintain any existing LIMIT clauses
                8. Ensure type compatibility in comparisons and calculations
                9. Use table aliases consistently
                10. End the query with a semicolon

                The output should be ONLY the corrected SQL query, no explanations."""),
                ("human", """Context:
                Analysis:
                {analysis}

                Strategy:
                {strategy}

                Table Schemas:
                {table_schemas}

                Original Query:
                {original_query}

                Error Message:
                {error_message}

                Please provide the corrected query:""")
            ])

            messages = revision_prompt.format_messages(
                analysis=analysis,
                strategy=strategy,
                table_schemas=json.dumps(schemas, indent=2),
                original_query=original_query,
                error_message=error_message
            )
            
            response = self.model_4o.invoke(messages)
            revised_query = self.clean_sql_query(response.content)
            
            self.logger.info(f"Original query: {original_query}")
            self.logger.info(f"Error message: {error_message}")
            self.logger.info(f"Revised query: {revised_query}")
            
            return revised_query
            
        except Exception as e:
            self.logger.error(f"Error in query revision: {str(e)}")
            raise

    def _execute_query_safely(self, query: str, max_retries: int = 3, strategy: str = None, analysis: str = None, schemas: dict = None) -> str:
        """Execute SQL query with error handling and automatic revision."""
        retries = 0
        current_query = query
        last_error = None
        
        while retries <= max_retries:
            try:
                self.logger.info(f"------current_query in _execute_query_safely: {current_query}\n-----------------\n")
                result = self.execute_query(current_query)
                
                # Check if result is a string containing error messages
                if isinstance(result, str) and (
                    "Error:" in result 
                    or "error" in result.lower() 
                    or "psycopg2.errors" in result
                ):
                    self.logger.warning(f"Query execution error: {result}")
                    last_error = result
                    
                    if retries == max_retries:
                        self.logger.error("Max retries reached, preparing error response")
                        error_response = {
                            "error": "Query execution failed after multiple attempts",
                            "details": {
                                "last_error": last_error,
                                "attempted_queries": [query, current_query],
                                "message": "There was an error executing your query. Please try rephrasing your question or simplifying your request.",
                                "suggestion": "Please verify the column names and table structure in your query."
                            }
                        }
                        return json.dumps(error_response, indent=2)
                    
                    self.logger.info("Attempting to revise query...")
                    try:
                        current_query = self._revise_sql_query(
                            current_query, 
                            result,
                            strategy,
                            analysis,
                            schemas
                        )
                    except Exception as revision_error:
                        self.logger.error(f"Error during query revision: {str(revision_error)}")
                        error_response = {
                            "error": "Failed to revise query",
                            "details": {
                                "original_error": str(last_error),
                                "revision_error": str(revision_error),
                                "message": "The system encountered an error while trying to fix the query. Please try rephrasing your question."
                            }
                        }
                        return json.dumps(error_response, indent=2)
                        
                    retries += 1
                    self.logger.info(f"Retry {retries} with revised query: {current_query}")
                else:
                    try:
                        # Try to parse the result as JSON to verify it's valid
                        if isinstance(result, str):
                            json.loads(result)
                        return result
                    except json.JSONDecodeError:
                        self.logger.error(f"Invalid JSON result: {result}")
                        error_response = {
                            "error": "Invalid query result format",
                            "details": {
                                "message": "The query executed but produced invalid results. Please try a different query."
                            }
                        }
                        return json.dumps(error_response, indent=2)
                        
            except Exception as e:
                self.logger.error(f"Database error executing query: {str(e)}")
                last_error = str(e)
                
                if retries == max_retries:
                    self.logger.error("Max retries reached after exceptions, preparing error response")
                    error_response = {
                        "error": "Query execution failed after multiple attempts",
                        "details": {
                            "last_error": last_error,
                            "attempted_queries": [query, current_query],
                            "message": "There was an error executing your query. Please try rephrasing your question or simplifying your request."
                        }
                    }
                    return json.dumps(error_response, indent=2)
                
                retries += 1
                self.logger.info(f"Retry {retries} after database error")
                
    def process_query(self, question: str):
        """Process a natural language query and return the result."""
        try:
            analysis = self._analyze_question(question)
            self.logger.info(f"Analysis: {analysis} \n-----------------\n")

            required_tables = self._extract_required_tables(analysis)
            self.logger.info(f"Required Tables: {required_tables} \n-----------------\n")

            schemas = self._load_table_schemas(required_tables)
            self.logger.info(f"Table Schemas Length: {len(schemas)} \n----------------\n")

            sql_query = self._generate_sql_query(
                question,
                analysis,
                schemas
            )
            self.logger.info(f"SQL Query: {sql_query} \n-----------------\n")

            executed_query_result = self._execute_query_safely(
            sql_query,
            strategy=analysis,  
            analysis=analysis,
            schemas=schemas
        )
            # Check if the result is an error response
            if isinstance(executed_query_result, str):
                try:
                    result_json = json.loads(executed_query_result)
                    if "error" in result_json:
                        return {
                            'sql_query': sql_query,
                            'result': executed_query_result,
                            'question': question,
                            'metadata': {
                                'tables_used': required_tables.tables,
                                'analysis': analysis,
                                'error': True
                            }
                        }
                except json.JSONDecodeError:
                    pass
            
            return {
                'sql_query': sql_query,
                'result': executed_query_result,
                'question': question,
                'metadata': {
                    'tables_used': required_tables.tables,
                    'analysis': analysis
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            error_response = {
            "error": "Query processing failed",
            "details": {
                "message": "There was an error processing your question. Please try rephrasing or simplifying your request.",
                "error_details": str(e)
                }
            }
            return {
                'sql_query': None,
                'result': json.dumps(error_response, indent=2),
                'question': question,
                'metadata': {
                    'error': True,
                    'error_message': str(e)
                }
            }

def test_module():
    """Test function for the TextToSQL engine."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to log file
    log_file = os.path.join(current_dir, '..', 'logs','text_to_sql_engine.log')
    
    engine = TextToSQLEngine(log_file)
    print("Enter your queries. Type 'q' to quit.")
    while True:
        query = input("Query: ")
        if query.lower() == 'q':
            break
        try:
            result = engine.process_query(query)
            result_data = json.loads(result['result'])
            print(json.dumps(result_data, indent=2))
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    test_module()