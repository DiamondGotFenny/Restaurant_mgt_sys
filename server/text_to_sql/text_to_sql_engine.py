import os
from typing import TypedDict
from typing_extensions import Annotated
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from pydantic import BaseModel , Field
from pprint import pprint
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
import json

from ..logger_config import setup_logger


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(BaseModel):
    """Generated SQL query."""
    query: Annotated[str, Field(..., description="Syntactically valid SQL query.")]

class TextToSQLEngine():
    def __init__(self,log_file_path: str = "vector_store_agent.log"):
      """Initializes the TextToSQLEngine with database connection details and LLM."""
      self.db_uri = os.getenv("NEON_RESTARANT_DB_STR")
      self.azure_openai_api_key = os.getenv("OPENAI_API_KEY")
      self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
      self.azure_openai_deployment_mini = os.getenv("OPENAI_MODEL_4OMINI")
      self.azure_api_version = os.getenv("AZURE_API_VERSION")
      self.azure_openai_embedding_deployment = os.getenv("OPENAI_EMBEDDING_MODEL")
      self.logger = setup_logger(log_file_path)
      # Check each environment variable and collect missing ones
      missing_vars = []
      if not self.db_uri:
          missing_vars.append("NEON_RESTARANT_DB_STR")
      if not self.azure_openai_api_key:
          missing_vars.append("OPENAI_API_KEY")
      if not self.azure_openai_endpoint:
          missing_vars.append("AZURE_OPENAI_ENDPOINT")
      if not self.azure_openai_deployment_mini:
          missing_vars.append("OPENAI_MODEL_4OMINI")
      if not self.azure_api_version:
          missing_vars.append("AZURE_API_VERSION")
      if not self.azure_openai_embedding_deployment:
          missing_vars.append("OPENAI_EMBEDDING_MODEL")

      if missing_vars:
          self.logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
          raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

      self.db = SQLDatabase.from_uri(self.db_uri)
      self.model_4o_mini = AzureChatOpenAI(
            api_key=self.azure_openai_api_key,
            azure_endpoint=self.azure_openai_endpoint,
            api_version=self.azure_api_version,
            azure_deployment=self.azure_openai_deployment_mini,
            temperature=0,
            max_tokens=3000
        )
      self.table_info = self.db.get_table_info()

      # Get the current file's directory
      current_dir = os.path.dirname(os.path.abspath(__file__))
      examples_path = os.path.join(current_dir, 'dynamic_examples.json')

      # Load few-shot examples from JSON file
      try:
          with open(examples_path, 'r') as f:
              examples_data = json.load(f)
      except FileNotFoundError:
          self.logger.error(f"dynamic_examples.json file not found at: {examples_path}")
          raise ValueError(f"dynamic_examples.json file not found at: {examples_path}")
      except json.JSONDecodeError:
          self.logger.error(f"Error decoding JSON from: {examples_path}")
          raise ValueError(f"Error decoding JSON from: {examples_path}")

      # Extract relevant fields for few-shot examples
      examples = []
      for case in examples_data['cases']:
          examples.append({"input": case["question"], "query": case["query"]})

      # Initialize Embeddings
      self.embeddings = AzureOpenAIEmbeddings(
            api_key=self.azure_openai_api_key,
            azure_endpoint=self.azure_openai_endpoint,
            api_version=self.azure_api_version,
            azure_deployment=self.azure_openai_embedding_deployment,
            model=self.azure_openai_embedding_deployment,
        )


      # Create Example Selector
      self.example_selector = SemanticSimilarityExampleSelector.from_examples(
          examples,
          self.embeddings,
          Chroma,
          k=3  
      )

      # Define the main prompt template with few-shot examples
      template_string = """You are an agent designed to interact with a PostgreSQL database.
      Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
      Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
      You can order the results by a relevant column to return the most interesting examples in the database.
      Never query for all the columns from a specific table, only ask for the relevant columns given the question.
      You have access to tools for interacting with the database.
      Only use the below tools. Only use the information returned by the below tools to construct your final answer.
      You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

      DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

      Here are some examples of previous questions and their corresponding SQL queries:
      {few_shot_examples}

      To start you should ALWAYS look at the tables in the database to see what you can query.
      Do NOT skip this step.
      Then you should query the schema of the most relevant tables.
      Only use the following tables:
      {table_info}

      question: {input}"""
      self.query_prompt_template = PromptTemplate(
          template=template_string,
          input_variables=["dialect", "top_k", "table_info", "input", "few_shot_examples"]
      )

      self.rewrite_query_prompt_template = PromptTemplate(
          template="""You are a helpful agent interacting with a PostgreSQL database.

The previous attempt to answer the user's question resulted in the following PostgreSQL error: {error}

The user's original question was: {question}

Here are some examples of previous questions and their corresponding SQL queries to help you formulate a correct query:
{few_shot_examples}

To proceed, please generate a revised PostgreSQL query that addresses the encountered error and accurately answers the user's question.

When constructing the revised query, please refer to the information available in these tables:
{table_info}

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
check the tables info and the error message carefully to generate the correct query.

Ensure the generated query is valid PostgreSQL and avoids the previous error.
""",
          input_variables=["error", "question", "table_info", "few_shot_examples"]
      )

      self.logger.info("TextToSQLEngine initialized successfully.")

    def write_query(self, state: State):
        """Generate SQL query to fetch information."""
        # Select relevant examples based on the current question
        selected_examples = self.example_selector.select_examples({"input": state["question"]})
        # Format the selected examples for the prompt
        formatted_examples = "\n".join([f"Question: {ex['input']}\nSQL Query: {ex['query']}" for ex in selected_examples])
        self.logger.debug(f"\n------------selected_examples: {selected_examples}----------------\n")
        prompt = self.query_prompt_template.invoke(
            {
                "dialect": self.db.dialect,
                "top_k": 20,
                "table_info": self.table_info,
                "input": state["question"],
                "few_shot_examples": formatted_examples,
            }
        )
        self.logger.info("waiting for writing sql from model")
        try:
            structured_llm = self.model_4o_mini.with_structured_output(QueryOutput)
            result = structured_llm.invoke(prompt)
            self.logger.info('write query success')
            return {"query": result.query}
        except Exception as e:
            self.logger.error(f"Error during write_query: {e}")
            return {"query": "",
                    "answer": "Sorry, I could not generate an answer due to an error, please try again later."
                    }

    def rewrite_query(self, state: State, error: str):
        """Rewrite the SQL query based on the error."""
        # Select relevant examples based on the current question
        selected_examples = self.example_selector.select_examples({"input": state["question"]})
        # Format the selected examples for the prompt
        formatted_examples = "\n".join([f"Question: {ex['input']}\nSQL Query: {ex['query']}" for ex in selected_examples])

        prompt = self.rewrite_query_prompt_template.invoke(
            {
                "error": error,
                "question": state["question"],
                "table_info": self.table_info,
                "few_shot_examples": formatted_examples,
            }
        )
        self.logger.info(f"waiting for rewriting sql from model due to error: {error}")
        try:
            structured_llm = self.model_4o_mini.with_structured_output(QueryOutput)
            result = structured_llm.invoke(prompt)
            self.logger.info('rewrite query success')
            return {"query": result.query}
        except Exception as e:
            self.logger.error(f"Error during rewrite_query: {e}")
            return {"query": ""}

    def execute_query(self, state: State, retry_count=0, max_retries=5):
        """Execute SQL query with retry logic."""
        execute_query_tool = QuerySQLDataBaseTool(db=self.db)
        self.logger.debug(f"Query to execute: {state['query']}")

        self.logger.info(f"waiting for executing sql from model (Attempt {retry_count + 1})")
        exc_query_result = execute_query_tool.invoke(state["query"])

        if "Error:" in exc_query_result or "psycopg2.errors" in exc_query_result:
            self.logger.error(f"Error executing query: {exc_query_result}")
            if retry_count < max_retries:
                self.logger.info("Attempting to rewrite the query...")
                rewritten_query_state = self.rewrite_query(state, exc_query_result)
                state.update(rewritten_query_state)
                return self.execute_query(state, retry_count=retry_count + 1, max_retries=max_retries)
            else:
                self.logger.error(f"Failed to execute query after {max_retries} retries.")
                return {"result": "Sorry, I could not process this question. Please try another one."}
        else:
            self.logger.info('execute query success')
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
        self.logger.info("waiting for response from model")
        try:
            response = self.model_4o_mini.invoke(prompt)
            self.logger.info('generate answer success')
            return {"answer": response.content}
        except Exception as e:
            self.logger.error(f"Error during generate_answer: {e}")
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
        self.logger.info(f"Final state: {state}")
        return state

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the relative path to the logs directory
    logs_dir = os.path.join(current_dir, '..', 'logs')
    # Create the logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    # Construct the full path to the log file
    log_file_path = os.path.join(logs_dir, 'test_text_to_sql_engine.log')
    # Initialize TextToSQLEngine
    sql_engine = TextToSQLEngine(log_file_path)

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
