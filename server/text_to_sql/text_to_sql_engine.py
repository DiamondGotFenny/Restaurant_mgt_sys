import os
from typing import TypedDict
from typing_extensions import Annotated
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from pydantic import BaseModel , Field
from pprint import pprint
# to do list: the query will fail if the query is complex, need to fix this issue to handle the error message and re-write the query
#maybe we can use the table_info instead of the get_table_info() result, as the the table_info hase more information about the table
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


class QueryOutput(BaseModel):
    """Generated SQL query."""
    query: Annotated[str, Field(..., description="Syntactically valid SQL query.")]


class TextToSQLEngine:
    def __init__(self):
      """Initializes the TextToSQLEngine with database connection details and LLM."""
      self.db_uri = os.getenv("NEON_RESTARANT_DB_STR")
      self.azure_openai_api_key = os.getenv("OPENAI_API_KEY")
      self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
      self.azure_openai_deployment_mini = os.getenv("OPENAI_MODEL_4OMINI")
      self.azure_api_version = os.getenv("AZURE_API_VERSION")
      
      # Check each environment variable and collect missing ones
      missing_vars = []
      if not self.db_uri:
          missing_vars.append("NEON_RESTARANT_DB_STR")
      if not self.azure_openai_api_key:
          missing_vars.append("OPENAI_API_KEY")
      if not self.azure_openai_endpoint:
          missing_vars.append("OPENAI_API_BASE")
      if not self.azure_openai_deployment_mini:
          missing_vars.append("OPENAI_MODEL_4OMINI")
      if not self.azure_api_version:
          missing_vars.append("AZURE_API_VERSION")
          
      if missing_vars:
          raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
          
      self.db = SQLDatabase.from_uri(self.db_uri)
      self.model_4o_mini = AzureChatOpenAI(
            api_key=self.azure_openai_api_key,
            azure_endpoint=self.azure_openai_endpoint,
            deployment_name=self.azure_openai_deployment_mini,
            api_version=self.azure_api_version,
            temperature=0,
            max_tokens=3000
        )

      template_string = """You are an agent designed to interact with a PostgreSQL database.
      Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
      Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
      You can order the results by a relevant column to return the most interesting examples in the database.
      Never query for all the columns from a specific table, only ask for the relevant columns given the question.
      You have access to tools for interacting with the database.
      Only use the below tools. Only use the information returned by the below tools to construct your final answer.
      You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

      DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

      To start you should ALWAYS look at the tables in the database to see what you can query.
      Do NOT skip this step.
      Then you should query the schema of the most relevant tables.
      Only use the following tables:
      {table_info}

      Question: {input}"""
      self.query_prompt_template = PromptTemplate(
          template=template_string,
          input_variables=["dialect", "top_k", "table_info", "input"]
      )
      print("TextToSQLEngine initialized successfully.")

    def write_query(self, state: State):
        """Generate SQL query to fetch information."""
        prompt = self.query_prompt_template.invoke(
            {
                "dialect": self.db.dialect,
                "top_k": 20,
                "table_info": self.db.get_table_info(),
                "input": state["question"],
            }
        )
        print(f"waiting for writing sql from model")
        structured_llm = self.model_4o_mini.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        pprint('write query success')
        return {"query": result.query}


    def execute_query(self, state: State):
        """Execute SQL query."""
        execute_query_tool = QuerySQLDataBaseTool(db=self.db)
        pprint(state["query"],)
        
        print(f"waiting for executing sql from model")
        exc_query_result=execute_query_tool.invoke(state["query"])
        pprint('execute query success')
        return {"result": exc_query_result}
    
    def generate_answer(self, state: State):
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {state["question"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["result"]}'
        )
        print(f"waiting for response from model")
        response = self.model_4o_mini.invoke(prompt)
        pprint('generate answer success')
        return {"answer": response.content}
    def process_query(self, question):
        """Process the user query to SQL"""
        state = {"question": question}
        state.update(self.write_query(state))
        state.update(self.execute_query(state))
        state.update(self.generate_answer(state))
        return state
    
if __name__ == "__main__":
  

    # Initialize TextToSQLEngine
    sql_engine = TextToSQLEngine()

    while True:
      # Get user input query
      user_query = input("Enter your question (or 'q' to quit): ")
      if user_query.lower() == 'q':
        break
      # Process query
      state = sql_engine.process_query(user_query)

      # Print the state
      print("\nState:")
      pprint(state)