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
from pydantic import BaseModel,Field
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

_ = load_dotenv(find_dotenv())


db_uri = os.getenv("NEON_RESTARANT_DB_STR")

azure_openai_api_key = os.getenv("OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_deployment_mini = os.getenv("OPENAI_MODEL_4OMINI")
azure_openai_deployment = os.getenv("OPENAI_MODEL_4O")
azure_api_version = os.getenv("AZURE_API_VERSION")
from logger_config import setup_logger
# Initialize database connection
db = SQLDatabase.from_uri(db_uri)

# Initialize Azure OpenAI models
# openai-4o model is not able to gerneate the query for the given question, but openai-4o-mini is able to 
model_4o_mini =AzureChatOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_deployment_mini,
            api_version=azure_api_version,
            temperature=0,  # Low temperature for more deterministic outputs
            max_tokens=3000    # Adjust based on expected summary length
        )

model_4o = AzureChatOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_deployment,
            api_version=azure_api_version,
            temperature=0.2,  # Low temperature for more deterministic outputs
            max_tokens=3000    # Adjust based on expected summary length
        )

execute_query = QuerySQLDataBaseTool(db=db)
answer_prompt = PromptTemplate.from_template(
     """Given the following user question, corresponding PostgreSQL query, and PostgreSQL result, answer the user question.

 Question: {question}
 PostgreSQL Query: {query}
 PostgreSQL Result: {result}
 
 note: if the postgresql result is empty, please tell the user that you can not find relevant information in the database.
 
 Answer: """
 )

rephrase_answer = answer_prompt | model_4o | StrOutputParser()
def get_table_details():
    # Read the CSV file into a DataFrame
    table_description = pd.read_csv("database_table_descriptions.csv")
    table_docs = []

    # Iterate over the DataFrame rows to create Document objects
    table_details = ""
    for index, row in table_description.iterrows():
        table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['Description'] + "\n\n"

    return table_details


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in PostgreSQL database.")

table_details = get_table_details()

# Load example questions and SQL queries from JSON file
with open('text_to_sql_examples.json', 'r') as f:
    examples = json.load(f)

# Define Langchain prompts and chains
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    input_variables=["input", "top_k"],
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
           """You are a PostgreSQL expert. Your task is to create a syntactically correct PostgreSQL query based on the input question.

Important guidelines:
1. ONLY use tables that are explicitly listed in this table information:
{table_info}

2. ONLY use columns that are explicitly mentioned in the table information for each table.

3. Before writing any query:
   - Carefully check the available tables and their descriptions
   - Use ONLY the tables mentioned above, even if the query examples use different table names
   - Do not assume the existence of any tables not listed in the table information
   -DO NOT use any column name that are not explicitly mentioned in that specific table information

4. When writing queries:
   - Ensure all table names exactly match those in the table information
   - Use appropriate JOIN conditions based on the actual table structure
   - Include only columns that are likely to exist in these tables
   - Follow standard PostgreSQL syntax

Below are example questions and their corresponding SQL queries, but remember to adapt them to use ONLY the tables listed in the table information above.""",
        ),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

generate_query = create_sql_query_chain(model_4o_mini, db, final_prompt)

table_details_prompt = f"""Return the names of ALL the PostgreSQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_details}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

table_chain = create_extraction_chain_pydantic(Table, model_4o_mini, system_message=table_details_prompt)

def get_tables(tables: List[Table]) -> List[str]:
    #print all tables names
    print("Tables:", [table.name for table in tables])
    tables = [table.name for table in tables]
    return tables

select_table = (
    {"input": itemgetter("question")}
    | table_chain
    | get_tables
)

def clean_sql_query(query: str) -> str:
    """
    Clean SQL query by removing unnecessary SQL markers and backticks.
    
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

def create_sql_validator():
    """
    Create a function that cleans SQL queries.
    """
    def validate_and_clean_sql(query: str) -> str:
        cleaned_query = clean_sql_query(query)
        print(f"Cleaned SQL query: {cleaned_query}")
        return cleaned_query

    return validate_and_clean_sql

# Create the SQL validator function
sql_validator = create_sql_validator()

# Modify the generate_query chain to include LLM-based validation
generate_query_with_validation = generate_query | sql_validator

# Update the main chain
chain = (
    RunnablePassthrough.assign(table_names_to_use=select_table)
    | RunnablePassthrough.assign(query=generate_query_with_validation).assign(
        result=itemgetter("query") | execute_query
    )
    | rephrase_answer
)


# Function to process natural language queries
def process_query(query: str):
    """Processes a natural language query and returns the result."""
    try:
        result = chain.invoke({"question": query})
        return {"result": result}
    except Exception as e:
        print(f"process_query Error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Test function
def test_module():
    print("Enter your queries. Type 'q' to quit.")
    while True:
        query = input("Query: ")
        if query.lower() == 'q':
            break
        try:
            result = process_query(query)
            print("Result:", result['result'])
        except Exception as e:
            print("test_module Error:", e)

if __name__ == "__main__":
    test_module()