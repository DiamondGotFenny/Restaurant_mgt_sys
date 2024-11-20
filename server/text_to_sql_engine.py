import os
import json
from typing import List,Optional
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from fastapi import HTTPException
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from pydantic import BaseModel, Field
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,ChatPromptTemplate, MessagesPlaceholder
)
from langchain.schema import HumanMessage, AIMessage

from logger_config import setup_logger
from get_tables_info import get_tables_info

class RequiredTables(BaseModel):
    """Output from Step 2"""
    tables: List[str] = Field(description="Tables that are directly needed for the main query")

class TableSchema(BaseModel):
    """Schema information for a single table"""
    columns: dict = Field(description="Column definitions")
    primary_key: List[str] = Field(description="Primary key columns")
    foreign_keys: Optional[dict] = Field(description="Foreign key relationships")

class TextToSQLEngine:
    def __init__(self):
        """Initialize the TextToSQLEngine with necessary configurations and components."""
        # Load environment variables
        load_dotenv(find_dotenv())

        # Define the base directory and paths
        self.BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        self.LOG_FILE = os.path.join(self.BASE_DIR, 'logs', 'text_to_sql_engine.log')
        
        # Setup logger
        self.logger = setup_logger(self.LOG_FILE)
        
        # Load environment variables
        self._load_env_variables()
        
        # Initialize components
        self._initialize_components()
        
        # Setup prompts
        self._setup_prompts()

    def _load_env_variables(self):
        """Load environment variables."""
        self.logger.info("Loading environment variables")
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
        
        # Initialize database
        self.db = SQLDatabase.from_uri(self.db_uri)
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
        self.embeddings = AzureOpenAIEmbeddings(
                model=self.azure_openai_embedding_deployment,
                api_key=self.azure_openai_api_key,
                azure_endpoint=self.azure_openai_endpoint,
                deployment=self.azure_openai_embedding_deployment,
            )
        
        self.execute_query = QuerySQLDataBaseTool(db=self.db)
        self.logger.info("AI models initialized successfully")

    def _setup_prompts(self):
        """Setup all prompts for the multi-step process"""
        
         # Load dynamic examples from JSON
        try:
            with open('dynamic_examples.json', 'r', encoding='utf-8') as f:
                dynamic_examples = json.load(f)['cases']
            
        except Exception as e:
            self.logger.error(f"Error loading dynamic examples: {str(e)}")
            dynamic_examples = []
        
        # Setup for Analysis Prompt
        analysis_example_prompt = ChatPromptTemplate.from_messages([
        ("human", "Question: {question}"),
        ("assistant", """**Analysis**:
{analysis}

**Query Strategy**:
{query_strategy}""")
        ])


        # Prepare examples
        analysis_examples = [
        {
            "question": ex["question"],
            "analysis": ex["analysis"],
            "query_strategy": ex["query_strategy"]
        }
        for ex in dynamic_examples
    ]
        
        analysis_example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=analysis_examples,
        embeddings=self.embeddings,
        vectorstore_cls=Chroma(),
        k=2,
        input_keys=["question"]  
        )
        
        # Define answer prompt
        self.answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding PostgreSQL query, and PostgreSQL result, answer the user question.

    Question: {question}
    PostgreSQL Query: {query}
    PostgreSQL Result: {result}
    Answer: """
        )
        
       
        
        #load  table descriptions 
        try:
            df = pd.read_csv('database_table_descriptions.csv')
            # Convert to a formatted string with table name and description
            self.table_descriptions = "\n\n".join([
                f"Table: {row['Table']}\n"
                f"Description: {row['Description']}"
                for _, row in df.iterrows()
            ])
            self.logger.info("Table descriptions loaded successfully")
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
        
        here is some examples:"""),
        FewShotChatMessagePromptTemplate(
            example_selector=analysis_example_selector,
            example_prompt=analysis_example_prompt,
            input_variables=["question"]
        ),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

        

        # Step 2: Table Extraction Prompt
        table_extraction_template = """
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
        """
        self.table_extraction_prompt = PromptTemplate.from_template(table_extraction_template)

        # Step 3: Query Generation Prompt
        # Setup for Query Generation Prompt
        query_example_prompt = ChatPromptTemplate.from_messages([
        ("human", "Question: {question}"),
        ("assistant", "Query: {query}")
    ])

        query_examples = []
        for ex in dynamic_examples:
            query_examples.append({
                "question": str(ex["question"]),
                "query": str(ex.get("query", ""))
            })

        query_example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=query_examples,
            embeddings=self.embeddings,
            vectorstore_cls=Chroma,
            k=2
        )
        
        try:
            with open('tables_info.json', 'r') as f:
                table_info = json.load(f)
                # Convert the list of tables to a dictionary format for easier access
                self.all_table_schemas = {table['name']: table for table in table_info['tables']}
                
            self.query_generation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert PostgreSQL query generator. Generate a precise SQL query based on the provided strategy and table schemas.

                Requirements:
                1. Use only the necessary columns based on the strategy
                2. Include columns needed for joins, filters, and calculations
                3. Limit results to first 20 rows by default
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
                FewShotChatMessagePromptTemplate(
                example_selector=query_example_selector,
                example_prompt=query_example_prompt,
                input_variables=["question"]
            ),
                ("human", """Strategy: {strategy}
        Table Schemas: {table_schemas}""")
            ])

        except Exception as e:
            self.logger.error(f"Error in _setup_prompts step 3: {str(e)}")
            raise
        
       
    def clean_sql_query(self, query: str) -> str:
        """
        Clean SQL query by removing unnecessary SQL markers and fixing quote escaping.
        
        Args:
            query (str): The SQL query to clean
            
        Returns:
            str: The cleaned SQL query
        """
        # Remove SQL markdown indicators if present
        query = query.replace("```sql", "").replace("```", "")
        
        # Remove backticks
        query = query.replace("`", "")
        
        # Fix single quote escaping for PostgreSQL
        # PostgreSQL uses '' (double single quotes) to escape single quotes, not \'
        query = query.replace("\\'", "''")
        
        # Strip whitespace and ensure proper ending
        query = query.strip()
        
        # Ensure the query ends with a semicolon
        if not query.endswith(';'):
            query += ';'
        
        return query
    
    def _analyze_question(self, question: str) -> str:
        """Step 1: Analyze the question and generate strategy"""
        model = self.model_4o
        self.logger.info(f"-----------------_analyze_question question: {question} \n------------------\n" )
        self.logger.info(f"-----------------_analyze_question table_descriptions: {self.table_descriptions} \n------------------\n" )
        try:
            prompt_format =  self.analysis_prompt.format_messages(
                        table_descriptions=self.table_descriptions,
                        question=question
                    )
            self.logger.info(f"-----------------_analyze_question prompt_format: {prompt_format} \n------------------\n" )
        except Exception as e:
            self.logger.error(f"Error in _analyze_question formatting prompt: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to _analyze_question format prompt: {str(e)}"
            )
        try:
            result = model.invoke(
                prompt_format
            )
            self.logger.info(f"-----------------_analyze_question result: {result} \n------------------\n" )
            return result.content
        except Exception as e:
            self.logger.error(f"Error in _analyze_question analyzing question: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to _analyze_question analyze question: {str(e)}"
            )
    
    def _extract_required_tables(self, analysis: str) -> RequiredTables:
        """Step 2: Extract required tables"""
        model = self.model_4o_mini.with_structured_output(RequiredTables)
        result = model.invoke(
            self.table_extraction_prompt.format(
                analysis=analysis,
            )
        )
        self.logger.info(f"_extract_required_tables result: {result}" )
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
    
    def _generate_sql_query(self, query: str,strategy:str, 
                          schemas: dict) -> str:
        """Step 4: Generate SQL query from natural language query."""
        formatted_prompt = self.query_generation_prompt.format(
        strategy=strategy,
        table_schemas=json.dumps(schemas, indent=2)
    )
        model = self.model_4o
        response = model.invoke(formatted_prompt)
        sql_query = response.content
        
        cleaned_query = self.clean_sql_query(sql_query)
         # Add LIMIT clause if needed
        final_query = self.add_limit_to_query(cleaned_query)

        return final_query
    
    def add_limit_to_query(self, sql_query: str, default_limit: int = 20) -> str:
        """
        Add LIMIT clause to SQL query if not present, or keep existing LIMIT.
        
        Args:
            sql_query (str): The SQL query to process
            default_limit (int): Default limit to add if no LIMIT clause exists (default: 20)
            
        Returns:
            str: SQL query with LIMIT clause
        """
        self.logger.debug(f"Processing query for LIMIT clause: {sql_query}")
        
        # Remove any trailing semicolon and extra whitespace
        clean_query = sql_query.strip().rstrip(';')
        
        # Convert to uppercase for easier pattern matching
        upper_query = clean_query.upper()
        
        try:
            # Check if query already contains LIMIT
            if 'LIMIT' in upper_query:
                self.logger.debug("Query already contains LIMIT clause")
                return f"{clean_query};"
                
            # Check if query contains ORDER BY
            if 'ORDER BY' in upper_query:
                # Add LIMIT after ORDER BY clause
                return f"{clean_query} LIMIT {default_limit};"
            else:
                # Add LIMIT at the end of query
                return f"{clean_query} LIMIT {default_limit};"
                
        except Exception as e:
            self.logger.error(f"Error processing LIMIT clause: {str(e)}")
            # Return original query with semicolon if there's an error
            return f"{clean_query};"

    

    def rephrase_answer(self, question: str, query: str, result: str) -> str:
        """Process the answer using the model."""
        
        # Prepare the prompt
        prompt_input = {
            "question": question,
            "query": query,
            "result": result
        }
        
        # Get response from the model
        prompt_result = self.answer_prompt.format_prompt(**prompt_input)
        model_response = self.model_4o.invoke(prompt_result.to_string())
        final_result = StrOutputParser().parse(model_response.content)
        
        return final_result
    
    def _revise_sql_query(self, original_query: str, error_message: str) -> str:
        """
        Use LLM to revise the SQL query based on error message.
        """
        try:
            revision_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a PostgreSQL expert. Your task is to fix SQL queries that have errors.
                Rules:
                - Only output the corrected SQL query
                - No explanations or comments
                - Ensure proper type casting for mathematical functions
                - Maintain the original query's intent
                - The query must end with a semicolon
                - Keep LIMIT clause if present"""),
                ("human", """Original query:
    {original_query}

    Error message:
    {error_message}

    Please provide the corrected query:""")
            ])

            messages = revision_prompt.format_messages(
                original_query=original_query,
                error_message=error_message
            )
            
            response = self.model_4o_mini.invoke(messages)
            revised_query = self.clean_sql_query(response.content)
            
            self.logger.info(f"Revised query: ----{revised_query}----")
            return revised_query
            
        except Exception as e:
            self.logger.error(f"Error in query revision: {str(e)}")
            raise
    
    def _execute_query_safely(self, query: str, max_retries: int = 2) -> str:
        """
        Execute SQL query with error handling and automatic revision.
        """
        retries = 0
        current_query = query
        
        while retries <= max_retries:
            try:
                self.logger.info(f"------current_query in _execute_query_safely: {current_query}\n-----------------\n")
                result = self.execute_query.run(current_query)
                self.logger.info(f"------result in _execute_query_safely: {result}\n-----------------\n")
                
                # Check if result contains error message
                if isinstance(result, str) and "Error:" in result:
                    self.logger.warning(f"Query execution error: {result}")
                    
                    # Check if we've hit max retries
                    if retries == max_retries:
                        self.logger.error("Max retries reached, raising last error")
                        raise HTTPException(status_code=500, detail=result)
                    
                    # Try to revise the query
                    self.logger.info("Attempting to revise query...")
                    current_query = self._revise_sql_query(current_query, result)
                    retries += 1
                    self.logger.info(f"Retry {retries} with revised query: {current_query}")
                else:
                    # Query executed successfully
                    self.logger.info(f"Query executed successfully start---------\n{result}\n--------end")
                    return result
                    
            except Exception as e:
                self.logger.error(f"Database error executing query: {str(e)}")
                if retries == max_retries:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Database error: {str(e)}"
                    )
                retries += 1
                self.logger.info(f"Retry {retries} after database error")

    def process_query(self, question: str):
        """Process a natural language query and return the result."""
        try:
           # Step 1: Analyze question
            analysis = self._analyze_question(question)
            self.logger.info(f"Analysis: {analysis} \n-----------------\n")
            # Step 2: Extract required tables
            required_tables = self._extract_required_tables(analysis)
            self.logger.info(f"Required Tables: {required_tables} \n-----------------\n")

            # Step 3: Load schemas
            schemas = self._load_table_schemas(required_tables)
            self.logger.info(f"Table Schemas: {schemas} \n-----------------\n")

            # Step 4: Generate SQL query
            sql_query = self._generate_sql_query(
                question,
                analysis,
                schemas
            )
            self.logger.info(f"SQL Query: {sql_query} \n-----------------\n")

            # Execute query and generate response
            # Step 5: Execute query with automatic revision if needed
            executed_query_result = self._execute_query_safely(sql_query)
            self.logger.info(f"Executed Query Result: {executed_query_result} \n-----------------\n")
            # Return structured output with raw results and metadata
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
            raise HTTPException(status_code=500, detail=str(e))

def test_module():
    """Test function for the TextToSQL engine."""
    engine = TextToSQLEngine()
    print("Enter your queries. Type 'q' to quit.")
    while True:
        query = input("Query: ")
        if query.lower() == 'q':
            break
        try:
            result = engine.process_query(query)
            print("Result:", result['result'])
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    test_module()