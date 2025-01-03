import os
from typing import TypedDict
from typing_extensions import Annotated
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from pydantic import BaseModel , Field
from pprint import pprint
# to do list:
#maybe we can use the table_info instead of the get_table_info() result, as the the table_info hase more information about the table
#and need to add few shot dynamic example to the model to make it more accurate, this step will improve the engine accuracy
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
      self.table_info = self.db.get_table_info()
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
      
      self.rewrite_query_prompt_template = PromptTemplate(
          template="""You are a helpful agent interacting with a PostgreSQL database.

The previous attempt to answer the user's question resulted in the following PostgreSQL error: {error}

The user's original question was: {question}

To proceed, please generate a revised PostgreSQL query that addresses the encountered error and accurately answers the user's question.

When constructing the revised query, please refer to the information available in these tables:
{table_info}

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
check the tables info and the error message carefully to generate the correct query.

Ensure the generated query is valid PostgreSQL and avoids the previous error.
""",
          input_variables=["error", "question", "table_info"]
      )
      
      print("TextToSQLEngine initialized successfully.")

    def write_query(self, state: State):
        """Generate SQL query to fetch information."""
        prompt = self.query_prompt_template.invoke(
            {
                "dialect": self.db.dialect,
                "top_k": 20,
                "table_info": self.table_info,
                "input": state["question"],
            }
        )
        print(f"waiting for writing sql from model")
        try:
            structured_llm = self.model_4o_mini.with_structured_output(QueryOutput)
            result = structured_llm.invoke(prompt)
            pprint('write query success')
            return {"query": result.query}
        except Exception as e:
            print(f"Error during write_query: {e}")
            return {"query": "",
                    "answer": "Sorry, I could not generate an answer due to an error, please try again later."
                    }
    
    def rewrite_query(self, state: State, error: str):
        """Rewrite the SQL query based on the error."""
        prompt = self.rewrite_query_prompt_template.invoke(
            {
                "error": error,
                "question": state["question"],
                "table_info": self.table_info,
            }
        )
        print(f"waiting for rewriting sql from model due to error: {error}")
        try:
            structured_llm = self.model_4o_mini.with_structured_output(QueryOutput)
            result = structured_llm.invoke(prompt)
            pprint('rewrite query success')
            return {"query": result.query}
        except Exception as e:
            print(f"Error during rewrite_query: {e}")
            return {"query": ""}


    def execute_query(self, state: State, retry_count=0, max_retries=5):
        """Execute SQL query with retry logic."""
        execute_query_tool = QuerySQLDataBaseTool(db=self.db)
        pprint(state["query"],)

        print(f"waiting for executing sql from model (Attempt {retry_count + 1})")
        exc_query_result = execute_query_tool.invoke(state["query"])

        if "Error:" in exc_query_result or "psycopg2.errors" in exc_query_result:
            print(f"Error executing query: {exc_query_result}")
            if retry_count < max_retries:
                print("Attempting to rewrite the query...")
                rewritten_query_state = self.rewrite_query(state, exc_query_result)
                state.update(rewritten_query_state)
                return self.execute_query(state, retry_count=retry_count + 1, max_retries=max_retries)
            else:
                print(f"Failed to execute query after {max_retries} retries.")
                return {"result": "Sorry, I could not process this question. Please try another one."}
        else:
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
        try:
            response = self.model_4o_mini.invoke(prompt)
            pprint('generate answer success')
            return {"answer": response.content}
        except Exception as e:
            print(f"Error during generate_answer: {e}")
            return {"answer": "Sorry, I could not generate an answer due to an error."}
    
    def process_query(self, question):
        """Process the user query to SQL"""
        state = {"question": question}
        state.update(self.write_query(state))
        if not state["query"]: # If write_query failed, return
            return state
        
        execution_result = self.execute_query(state)
        state.update(execution_result)
        if "Sorry" in state.get("result", ""):  # If execution failed after retries, return
            return state

        # Generate Answer with error handling
        answer_result = self.generate_answer(state)
        state.update(answer_result)
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