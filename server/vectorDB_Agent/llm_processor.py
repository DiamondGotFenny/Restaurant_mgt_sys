# llm_processor.py

import os
from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from logger_config import setup_logger
from collections import namedtuple

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
            log_file (str): Path to the log file.
        """
        self.logger = setup_logger(log_file)
        self.llm = AzureChatOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_deployment,
            api_version=azure_api_version,
            temperature=0.1,  # Low temperature for more deterministic outputs
            max_tokens=3000    # Adjust based on expected summary length
        )
        self.prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template="""
You are an intelligent assistant. Based on the user query and the provided documents, extract the most relevant information that directly answers the query. \
just ignore the unrelated information, but return all related information.\
don't miss the key entities infomation, such as address, phone number, business hours etc.\ 
don't try to make up an answer don't generate hallucination. only extract the information from context or Retrieved Documents.

**User Query:**
{query}

**Retrieved Documents:**
{documents}

**Summary of Relevant Information:**
"""
        )
        self.output_parser = StrOutputParser()
        self.llm_chain = self.prompt|self.llm|self.output_parser

    def process_query_response(self, query: str, raw_results: List, min_length: int = 20) -> str:
        """
        Processes raw vector store results using an LLM to extract and summarize the most relevant information.

        Args:
            query (str): The user's query.
            raw_results (List): The raw documents retrieved from the vector store.
            min_length (int): Minimum number of characters required to perform summarization.

        Returns:
            str: A distilled and summarized response or an appropriate message.
        """
        if not raw_results:
            self.logger.info("No results found for the query. Skipping summarization.")
            return "No relevant information found to answer your query."

        # Calculate total length of retrieved documents
        total_length = sum(len(doc.page_content) for doc in raw_results)
        if total_length < min_length:
            self.logger.info("Retrieved results are too short. Skipping summarization.")
            return "Insufficient information found to answer your query."

        # Concatenate the content of all retrieved documents
        documents_content = "\n\n".join([doc.page_content for doc in raw_results])

        try:
            # Use the RunnableSequence to process the documents
            summary = self.llm_chain.invoke({
                "query": query,
                "documents": documents_content
            })
            self.logger.info(f" Raw Retrieved Documents: {documents_content} ")  
            self.logger.info("Summarization successful.")
            self.logger.info(f"Summarized Response: {summary.strip()}")
            return summary.strip()
        except Exception as e:
            self.logger.error(f"Error during summarization: {e}")
            return "An error occurred while processing your request."
        
def test_llm_processor():
    """
    Test function to interactively input query and content, and print the summarized result.
    """

    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("OPENAI_API_BASE")
    os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")
    os.environ["AZURE_OPENAI_4O"] = os.getenv("OPENAI_MODEL_4o")

    AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"] 
    AZURE_OPENAI_4O = os.environ["AZURE_OPENAI_4o"]
    AZURE_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]

    # Initialize LLMProcessor
    llm_processor = LLMProcessor(
         azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_deployment=AZURE_OPENAI_4O,
        azure_api_version=AZURE_API_VERSION,
        log_file="llm_processor_test.log"
    )

    print("=== LLM Processor Test Module ===")
    print("Enter 'exit' to quit.")
    while True:
        try:
            query = input("Enter your query: ")
            if query.strip().lower() in ['exit', 'quit']:
                print("Exiting test module.")
                break
            if not query.strip():
                print("Empty query. Please enter a valid query.")
                continue

            content = input("Enter the content from retrieved documents: ")
            if content.strip().lower() in ['exit', 'quit']:
                print("Exiting test module.")
                break
            if not content.strip():
                print("Empty content. Please enter valid content.")
                continue

            # Create a mock 'Document' list
            Document = namedtuple("Document", ["page_content"])
            raw_results = [Document(page_content=content)]

            # Process the query
            summary = llm_processor.process_query_response(query, raw_results)

            print(summary)
            print("\n--- End of Response ---\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting test module.")
            break

if __name__ == "__main__":
    test_llm_processor()