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
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

_ = load_dotenv(find_dotenv())

# Load environment variables
db_uri = os.getenv("NEON_RESTARANT_DB_STR")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("OPENAI_API_BASE")
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = os.getenv("OPENAI_MODEL_3")


# Initialize database connection
db = SQLDatabase.from_uri(db_uri)

# Initialize Azure OpenAI models
model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    temperature=0,
)
model_4o = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment="gpt4o",
    temperature=0,
)
execute_query = QuerySQLDataBaseTool(db=db)
answer_prompt = PromptTemplate.from_template(
     """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

 Question: {question}
 SQL Query: {query}
 SQL Result: {result}
 Answer: """
 )

rephrase_answer = answer_prompt | model | StrOutputParser()
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
            "You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run. Unless otherwise specificed.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
        ),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

generate_query = create_sql_query_chain(model, db, final_prompt)

table_details_prompt = f"""Return the names of ALL the PostgreSQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_details}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

table_chain = create_extraction_chain_pydantic(Table, model, system_message=table_details_prompt)

def get_tables(tables: List[Table]) -> List[str]:
    tables = [table.name for table in tables]
    return tables

select_table = (
    {"input": itemgetter("question")}
    | table_chain
    | get_tables
)

def create_sql_validator(llm):
    """
    Create a function that uses an LLM to validate and clean SQL queries.
    """
    validator_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert in PostgreSQL. Your task is to examine SQL queries, identify any issues, and provide a corrected version if necessary. Focus on:
        1. Syntax errors
        2. Unterminated quotes or identifiers
        3. Incorrect use of single vs double quotes
        4. Malformed clauses or statements
        5. Any other potential issues that could prevent query execution
        6. remove the 'sql' character that is used to indicate the start of the SQL query

        Always return a valid PostgreSQL query, even if significant changes are needed.
        Do not include any explanations or additional text in your response, just the corrected SQL query.""",
            ),
            ("human", "Please validate and clean the following SQL query:\n\n{query}"),
        ]
    )

    validator_chain = validator_prompt | llm | StrOutputParser()

    def validate_and_clean_sql(query: str) -> str:
        result = validator_chain.invoke({"query": query})
        cleaned_query = result.strip()
        if not isinstance(cleaned_query, str):
            print(f"Warning: SQL validator returned non-string: {cleaned_query}")
            return str(cleaned_query)  # Force conversion to string
        print(f"Cleaned SQL query: {cleaned_query}")
        return cleaned_query

    return validate_and_clean_sql

# Create the SQL validator function
sql_validator = create_sql_validator(model_4o)

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