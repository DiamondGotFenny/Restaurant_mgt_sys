import os
import sys
from dotenv import load_dotenv, find_dotenv
from vector_store_agent import VectorStoreAgent
from llm_processor import LLMProcessor
from logger_config import setup_logger

def vectorDB_Engine():
    """
    Main function to initialize the agent and start the interactive test module.
    """
    # Configuration - Replace these with your actual paths and Azure OpenAI credentials
    _ = load_dotenv(find_dotenv())
    LOG_FILE = ".././logs/vector_store_agent.log" 
    
   
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("OPENAI_API_BASE")
    os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")
    os.environ["AZURE_OPENAI_EMBEDDING_MODEL"] = os.getenv("OPENAI_EMBEDDING_MODEL")
    os.environ["AZURE_OPENAI_4O"] = os.getenv("OPENAI_MODEL_4o")
    # Configuration - Replace these with your actual paths and Azure OpenAI credentials
    PDF_DIRECTORY = ".././data/Restaurants_data"  # e.g., "./pdfs"
    PERSIST_DIRECTORY = ".././data/vectorDB/chroma"
    AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"] 
    AZURE_OPENAI_EMBEDDING = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]
    AZURE_OPENAI_4O = os.environ["AZURE_OPENAI_4o"]
    AZURE_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]

    # Setup logger
    logger = setup_logger(LOG_FILE)

    # Validate directories
    if not os.path.isdir(PDF_DIRECTORY):
        logger.error(f"PDF directory does not exist: {PDF_DIRECTORY}")
        sys.exit(1)
    if not os.path.exists(os.path.dirname(PERSIST_DIRECTORY)):
        try:
            os.makedirs(os.path.dirname(PERSIST_DIRECTORY), exist_ok=True)
            logger.info(f"Created directory for vector store: {os.path.dirname(PERSIST_DIRECTORY)}")
        except Exception as e:
            logger.error(f"Failed to create directory {os.path.dirname(PERSIST_DIRECTORY)}: {e}")
            sys.exit(1)
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        try:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            logger.info(f"Created directory for logs: {os.path.dirname(LOG_FILE)}")
        except Exception as e:
            logger.error(f"Failed to create directory {os.path.dirname(LOG_FILE)}: {e}")
            sys.exit(1)

    # Initialize the VectorStoreAgent
    agent = VectorStoreAgent(
        pdf_directory=PDF_DIRECTORY,
        persist_directory=PERSIST_DIRECTORY,
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_deployment=AZURE_OPENAI_EMBEDDING,
        log_file=LOG_FILE
    )

    # Initialize the LLMProcessor
    llm_processor = LLMProcessor(
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_deployment=AZURE_OPENAI_4O,
        azure_api_version=AZURE_API_VERSION,
        log_file=LOG_FILE
    )

    def qa_chain(query):
        raw_results = agent.query(query)
        summary = llm_processor.process_query_response(query, raw_results)
        return summary

    def test_module():
        """
        A test method to interactively query the vector store via the terminal.
        Displays both raw and summarized results, handles edge cases, and logs interactions.
        """
        print("=== Vector Store Agent Test Module ===")
        print("Enter 'exit' to quit.")
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

            # Perform the query using the conversational chain
            try:
                response = qa_chain(query)
            except Exception as e:
                print(f"An error occurred during the query: {e}")
                continue

            # Display the summarized response
            print("\n--- Summarized Response ---")
            print(response)
            print("\n--- End of Response ---\n")

    # Start the test module
    test_module()

if __name__ == "__main__":
    vectorDB_Engine()