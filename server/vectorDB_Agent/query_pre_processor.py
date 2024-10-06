#query_pre_processor.py
"""
LLM Query Pre-Processor Module for Extracting Key Entities

This module uses Azure OpenAI's GPT-based models to extract relevant entities
from user queries, facilitating precise keyword-based searches using BM25.
It ensures that only critical entities are extracted, avoiding generic terms,
and formats the output as a structured JSON object for seamless integration
with the BM25RetrieverAgent.
"""

import os
from typing import List, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from logger_config import setup_logger
from dotenv import load_dotenv, find_dotenv
import re
import json

# Load environment variables from a .env file if present
_ = load_dotenv(find_dotenv())

class LLMQueryPreProcessor:
    def __init__(
        self,
        azure_openai_api_key: str,
        azure_openai_endpoint: str,
        azure_openai_deployment: str,
        azure_api_version: str,
        log_file: str,
    ):
        """
        Initializes the LLMQueryPreProcessor.
        
        Args:
            azure_openai_api_key (str): API key for Azure OpenAI.
            azure_openai_endpoint (str): Endpoint URL for Azure OpenAI.
            azure_openai_deployment (str): Deployment name for Azure OpenAI.
            azure_api_version (str): API version for Azure OpenAI.
            log_file (str): Path to the log file for recording processes and errors.
        """
        self.logger = setup_logger(log_file)
        
        # Initialize the AzureChatOpenAI model with specified parameters
        self.llm = AzureChatOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_deployment,
            api_version=azure_api_version,
            temperature=0.1,  # Low temperature for deterministic outputs
            max_tokens=500     # Adjust based on expected response length
        )
        
        # Define the prompt template with clear instructions and examples
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=(
               """Extract relevant entities from the following user query for keyword searching in a New York restaurant database.

**User Query:** "{query}"

**Instructions:**
1. **Include Only:**
   - **Restaurant Names:** Specific names of restaurants.
   - **Locations:** Neighborhoods, landmarks, or areas in New York (e.g., "Grand Central Terminal", "Upper West Side").
   - **Cuisine Types:** General types of cuisine using single words (e.g., "Indian", "Chinese").
   - **Specific Items:** Specific dishes or items (e.g., "curry chicken", "sushi","steak").

2. **Exclude:**
   - **Generic Terms:** Words like "restaurant", "food", "eatery", etc.
   - **Specific Items that user says that don't want** e.g., "no curry chicken", "I don't like oyster", then don't put 'curry chicken' or 'oyster' in the keyword list.
   - **Descriptive Phrases:** Phrases that describe preferences without adding unique information.

3. **Formatting:**
   - Provide the output as a JSON object with a single key `"entities"` mapping to a list of relevant entities.
   - Ensure the JSON is valid and **output only the JSON object** without any additional text.

**Example:**

- **Input Query:** "What are some highly rated Indian restaurants in the Upper West Side that serve curry chicken?"
  
- **Output:**
```
{{\n
  "entities": ["Indian", "Upper West Side","curry chicken"] \n
}}"""
            )
        )
        
        # Initialize the output parser to handle string outputs from LLM
        self.output_parser = StrOutputParser()
        
        # Chain the prompt and model with the output parser
        self.llm_chain = self.prompt | self.llm | self.output_parser

    def process_query(self, query: str) -> Dict[str, List[str]]:
        """
        Processes the user query to extract key entities.
        
        Args:
            query (str): The user input query.
        
        Returns:
            Dict[str, List[str]]: A dictionary with an 'entities' key containing a list of extracted entities.
        """
        self.logger.info(f"Processing query: {query}")
        try:
            # Invoke the LLM chain with the user query
            response = self.llm_chain.invoke({
                "query": query
            })

            self.logger.info(f"LLM response: {response}")

            # Use regex to extract JSON object from the LLM's response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("Failed to parse JSON from LLM response.")

            json_str = json_match.group(0)
            
            # Safely parse the JSON string into a Python dictionary
            extracted = json.loads(json_str)

            # Validate that 'entities' exists and is a list of strings
            entities = extracted.get('entities', [])
            if not isinstance(entities, list) or not all(isinstance(ent, str) for ent in entities):
                raise ValueError("'entities' must be a list of strings.")

            self.logger.info(f"Extracted entities: {entities}")

            return {
                'entities': entities
            }
        except json.JSONDecodeError as jde:
            self.logger.error(f"JSONDecodeError while parsing LLM response: {jde}")
            return {
                'entities': []
            }
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'entities': []
            }

def test_module():
    """
    A test method to interactively test the LLMQueryPreProcessor via the terminal.
    """
    # Set environment variables required for Azure OpenAI
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("OPENAI_API_BASE")
    os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")
    os.environ["AZURE_OPENAI_4O"] = os.getenv("OPENAI_MODEL_4o")

    # Retrieve environment variables
    AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_4O = os.environ.get("AZURE_OPENAI_4O")
    AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")
    
    # Initialize the LLMQueryPreProcessor
    processor = LLMQueryPreProcessor(
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_deployment=AZURE_OPENAI_4O,
        azure_api_version=AZURE_API_VERSION,
        log_file="llm_processor.log"
    )
    
    print("=== LLMQueryPreProcessor Test Module ===")
    print("Enter 'exit' or 'quit' to terminate the test.\n")
    
    while True:
        try:
            user_input = input("Enter your query: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting test module.")
            break

        if user_input.strip().lower() in ['exit', 'quit']:
            print("Exiting test module.")
            break

        if not user_input.strip():
            print("Empty query. Please enter a valid query.")
            continue

        # Process the user query to extract entities
        result = processor.process_query(user_input)
        print("\nExtracted Entities:")
        print(f"entities: {result['entities']}\n")

# Example usage:
if __name__ == "__main__":
    # Run the test module
    test_module()