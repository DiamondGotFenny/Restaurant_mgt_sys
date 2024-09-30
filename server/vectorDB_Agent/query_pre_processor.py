import os
from typing import List, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from logger_config import setup_logger
from dotenv import load_dotenv, find_dotenv
import re
_ = load_dotenv(find_dotenv())

class LLMProcessor:
    def __init__(
        self,
        azure_openai_api_key: str,
        azure_openai_endpoint: str,
        azure_openai_deployment: str,
        azure_api_version: str,
        log_file: str,
    ):
        """
        Initializes the LLMProcessor.

        Args:
            azure_openai_api_key (str): API key for Azure OpenAI.
            azure_openai_endpoint (str): Endpoint for Azure OpenAI.
            azure_openai_deployment (str): Deployment name for Azure OpenAI.
            azure_api_version (str): API version for Azure OpenAI.
            log_file (str): Path to the log file.
        """
        self.logger = setup_logger(log_file)
        self.llm = AzureChatOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_deployment,
            api_version=azure_api_version,
            temperature=0.1,  # Low temperature for more deterministic outputs
            max_tokens=500    # Adjust based on expected summary length
        )
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=(
               "Extract relevant terms and entities for keyword searching, from the following user query.\n\n"
                "User Query: \"{query}\"\n\n"
                "Provide the output as a JSON object with two fields: 'terms' and 'entities'.\n"
                "Ensure that 'terms' is a list of descriptive terms (e.g., 'spicy', 'quiet') and "
                "'entities' is a list of specific items or categories (e.g., 'Mediterranean', 'noodle').\n\n"
                "extracted 'terms' and 'entities' should be optimized for keyword searching for getting best result .\n\n"
                "try your best to identify the most relevant entities, base on user's intention, as entities will be put on higher priority.\n\n"
                "Output JSON only, without any additional text."
            )
        )
        self.output_parser = StrOutputParser()
        self.llm_chain = self.prompt | self.llm | self.output_parser
    def process_query(self, query: str) -> Dict[str, List[str]]:
        """
        Processes the user query to extract terms and entities.

        Args:
            query (str): The user input query.

        Returns:
            Dict[str, List[str]]: A dictionary with 'terms' and 'entities' lists.
        """
        self.logger.info(f"Processing query: {query}")
        try:
            response = self.llm_chain.invoke({
                "query": query
            })

            self.logger.info(f"LLM response: {response}")

            # Extract JSON from the response using regex to ensure proper parsing
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("Failed to parse JSON from LLM response.")

            json_str = json_match.group(0)
            extracted = eval(json_str)  # Caution: using eval; ensure safety in production

            # Validate extracted content
            terms = extracted.get('terms', [])
            entities = extracted.get('entities', [])

            self.logger.info(f"Extracted terms: {terms}")
            self.logger.info(f"Extracted entities: {entities}")

            return {
                'terms': terms,
                'entities': entities
            }
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'terms': [],
                'entities': []
            }

def test_module():
    """
    A test method to interactively test the LLMProcessor via the terminal.
    """
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("OPENAI_API_BASE")
    os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")
    os.environ["AZURE_OPENAI_4O"] = os.getenv("OPENAI_MODEL_4o")

    AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"] 
    AZURE_OPENAI_4O = os.environ["AZURE_OPENAI_4o"]
    AZURE_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
    processor = LLMProcessor(
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_deployment=AZURE_OPENAI_4O,
        azure_api_version=AZURE_API_VERSION,
        log_file="llm_processor.log"
    )
    print("=== LLMProcessor Test Module ===")
    print("Enter 'exit' or 'quit' to terminate the test.\n")
    while True:
        try:
            query = input("Enter your query: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting test module.")
            break
        if query.strip().lower() in ['exit', 'quit']:
            print("Exiting test module.")
            break
        if not query.strip():
            print("Empty query. Please enter a valid query.")
            continue
        
        result = processor.process_query(query)
        print("\nExtracted Keywords and Entities:")
        print(f"terms: {result['terms']}")
        print(f"Entities: {result['entities']}\n")

# Example usage:
if __name__ == "__main__":
    # Run the test module
    test_module()