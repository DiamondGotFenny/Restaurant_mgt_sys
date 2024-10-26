import os
import json
from typing import List
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from fastapi import HTTPException
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from pydantic import BaseModel, Field
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)

from logger_config import setup_logger
from get_tables_info import get_tables_info

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in PostgreSQL database.")

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
        
        self.execute_query = QuerySQLDataBaseTool(db=self.db)
        self.logger.info("AI models initialized successfully")

    def _setup_prompts(self):
        """Setup prompts and load examples."""
        self.logger.info("Setting up prompts")
        
        # Get database schema information
        self.table_info = get_tables_info()
        
        # Define answer prompt
        self.answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding PostgreSQL query, and PostgreSQL result, answer the user question.

    Question: {question}
    PostgreSQL Query: {query}
    PostgreSQL Result: {result}
    Answer: """
        )

        # Load examples
        with open('text_to_sql_examples.json', 'r') as f:
            self.examples = json.load(f)

        # Create example prompt
        self.example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}\nSQLQuery:"),
                ("ai", "{query}"),
            ]
        )

        # Create few-shot prompt with table_info
        self.few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=self.example_prompt,
            examples=self.examples,
            input_variables=["input", "top_k", "table_info"],  # Add table_info here
        )

        # Create final prompt with schema information
        self.final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a PostgreSQL expert. Your task is to create a syntactically correct PostgreSQL query based on the input question.

    Available Database Schema:
    {table_info}

    AVAILABLE DATABASE SCHEMA AND RELATIONSHIPS:

1. restaurant_inspections (ri):
   - Key columns: id, camis, dba, boro, cuisine_description, inspection_date, grade, score
   - Primary key: id
   - Note: This table must be joined with other tables using restaurant name (dba)

2. restaurant_menu (rm):
   - Key columns: id, restaurant_id, item, base_price, has_addons, max_price
   - Primary key: id
   - Foreign key: restaurant_id
   - Common usage: Menu items and pricing information

3. restaurant_reviews (rr):
   - Key columns: id, review_id, restaurant_id, title, text, rating, date
   - Primary key: id
   - Foreign key: restaurant_id -> restaurants_has_reviews.id
   - Common usage: Detailed review content and ratings

4. restaurants_has_menu (rhm):
   - Key columns: id, name, cuisine, address, restaurant_id
   - Primary key: id
   - Foreign key: restaurant_id -> restaurant_menu.restaurant_id
   - Common usage: Basic restaurant information with menu data

5. restaurants_has_reviews (rhr):
   - Key columns: id, name, city, price_interval, rating, type
   - Primary key: id
   - Relationships: 
     * id -> restaurant_reviews.restaurant_id
     * name can be used to join with other tables
   - Common usage: Basic restaurant information with aggregate review data

6. restaurants_week_2018_final (rwf):
   - Key columns: id, name, street_address, website, restaurant_type, average_review, 
     food_review, service_review, ambience_review, value_review, restaurant_id
   - Primary key: id
   - Note: Must be joined with other tables using restaurant name

7. trip_advisor_restaurants (tar):
   - Key columns: id, title, number_of_reviews, category, online_order, restaurant_id
   - Primary key: id
   - Note: Must be joined with other tables using restaurant name (title)

IMPORTANT RELATIONSHIP NOTES:
1. Key Cross-Table Relationships:
   - restaurants_has_reviews.id -> restaurant_reviews.restaurant_id (direct foreign key)
   - restaurants_has_menu.restaurant_id -> restaurant_menu.restaurant_id (direct foreign key)
   
2. Name-Based Joins:
   - Most cross-table relationships must use restaurant names for joining
   - Example: restaurants_week_2018_final.name = restaurants_has_reviews.name
   - Be careful with name matching as it must be exact

3. Review System Structure:
   - restaurant_reviews contains detailed reviews
   - restaurants_has_reviews contains aggregate review data
   - These tables are not guaranteed to have matching records
   - Always use appropriate LEFT/RIGHT joins when combining review data

QUERY GUIDELINES:
1. Table Joining:
   - Use exact name matching for joins between most tables
   - Only use id-based joins where explicitly specified in relationships
   - Always use LEFT/RIGHT joins when data presence is not guaranteed

2. Best Practices:
   - For name-based joins, consider case sensitivity
   - Use appropriate JOIN type based on data requirements
   - Include proper error handling for potential NULL values

Example Query Patterns:

1. Joining reviews with restaurant details:
```sql
SELECT rhr.name, rhr.rating, rwf.website
FROM restaurants_has_reviews rhr
LEFT JOIN restaurants_week_2018_final rwf ON rhr.name = rwf.name
WHERE rhr.name = 'Restaurant Name'; \

2.Getting detailed reviews for a restaurant:
SELECT rhr.name, rr.text, rr.rating, rr.date
FROM restaurants_has_reviews rhr
LEFT JOIN restaurant_reviews rr ON rhr.id = rr.restaurant_id
WHERE rhr.name = 'Restaurant Name'; \

3.Combining menu and restaurant information:
SELECT rhm.name, rhm.cuisine, rm.item, rm.base_price
FROM restaurants_has_menu rhm
LEFT JOIN restaurant_menu rm ON rhm.restaurant_id = rm.restaurant_id
LEFT JOIN restaurants_week_2018_final rwf ON rhm.name = rwf.name
WHERE rhm.name = 'Restaurant Name';
```
"""
                ),
                 self.few_shot_prompt,
                ("human", "{input}"),
            ]
        )
        self.logger.info("------Prompts setup completed-------")

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
        
        self.logger.info(f"---------Cleaned SQL query start-------")
        self.logger.info(f"{query}")
        self.logger.info(f"---------Cleaned SQL query end-------")
        return query

    def _get_relevant_tables(self, query: str) -> List[str]:
        """Get relevant tables for the query."""
        self.logger.info(f"Getting relevant tables for query: {query}")
        
        table_chain = create_extraction_chain_pydantic(
            Table, 
            self.model_4o_mini, 
            system_message=f"""Return the names of ALL the PostgreSQL tables that MIGHT be relevant to the user question. \
The tables are:

{self.table_info}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
        )
        
        result = table_chain.invoke(query)
        tables = [table.name for table in result]
        
        self.logger.info(f"Relevant tables: {tables}")
        return tables

    def _generate_sql_query(self, query: str, tables: List[str]) -> str:
        """Generate SQL query from natural language query."""
        self.logger.info(f"Generating SQL query for: {query}")
        
        query_chain = create_sql_query_chain(
            self.model_4o_mini, 
            self.db, 
            self.final_prompt
        )
        
        # Include table_info in the input dictionary
        sql_query = query_chain.invoke({
            "question": query,
            "input": query,
            "top_k": 5,
            "table_info": self.table_info
        })
        cleaned_query = self.clean_sql_query(sql_query)
        
        self.logger.info(f"Generated SQL query: {cleaned_query}")
        return cleaned_query

    def rephrase_answer(self, question: str, query: str, result: str) -> str:
        """Process the answer using the model."""
        self.logger.info(f"Rephrasing answer for question: {question}")
        
        # Prepare the prompt
        prompt_input = {
            "question": question,
            "query": query,
            "result": result
        }
        
        # Get response from the model
        prompt_result = self.answer_prompt.format_prompt(**prompt_input)
        self.logger.info(f"-----------Prompt Result start----------")
        self.logger.info(f"Prompt Result: {prompt_result}")
        self.logger.info(f"-----------Prompt Result end----------")
        model_response = self.model_4o.invoke(prompt_result.to_string())
        self.logger.info(f"Model response: {model_response.content}")
        final_result = StrOutputParser().parse(model_response.content)
        
        #self.logger.info(f"Generated answer: {final_result}")
        return final_result

    def process_query(self, query: str):
        """Process a natural language query and return the result."""
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Get relevant tables
            tables = self._get_relevant_tables(query)
            
            # Generate SQL query
            sql_query = self._generate_sql_query(query, tables)
            
            # Execute query
            query_result = self.execute_query.run(sql_query)
            
            # Generate final answer
            final_result = self.rephrase_answer(query, sql_query, query_result)
            
            self.logger.info(f"Query processed successfully")
            return {"result": final_result}
            
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