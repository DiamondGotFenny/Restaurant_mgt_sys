#vectorDB_Engine.py
import os
import sys
from dotenv import load_dotenv, find_dotenv
from vector_store_agent import VectorStoreAgent
from llm_post_processor import LLMProcessor
from logger_config import setup_logger
from combined_keyword_retriever import CombinedKeywordRetriever

def vectorDB_Engine():
    """
    Main function to initialize the agent and start the interactive test module.
    """
   
    _ = load_dotenv(find_dotenv())
    LOG_FILE = ".././logs/hybrid_search_agent.log" 
    
   
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("OPENAI_API_BASE")
    os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")
    os.environ["AZURE_OPENAI_EMBEDDING_MODEL"] = os.getenv("OPENAI_EMBEDDING_MODEL")
    os.environ["AZURE_OPENAI_4O"] = os.getenv("OPENAI_MODEL_4o")
  
    PDF_DIRECTORY = ".././data/Restaurants_data"  
    PERSIST_DIRECTORY = ".././data/vectorDB/chroma"
    AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"] 
    AZURE_OPENAI_EMBEDDING = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]
    AZURE_OPENAI_4O = os.environ["AZURE_OPENAI_4o"]
    AZURE_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
    WHOOSH_INDEX_DIR = "whoosh_index" 
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
    agent_vector_search = VectorStoreAgent(
        pdf_directory=PDF_DIRECTORY,
        persist_directory=PERSIST_DIRECTORY,
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_embedding_deployment=AZURE_OPENAI_EMBEDDING,
        log_file=LOG_FILE
    )
    # Initialize the CombinedKeywordRetriever
    agent_keyword_search = CombinedKeywordRetriever(
        pdf_directory=PDF_DIRECTORY,
        log_file_pre_processor="llm_processor.log",
        log_file_retriever="bm25_retriever_agent.log",
        chunk_size=2000,
        chunk_overlap=200,
        bm25_params={"k1": 0.5, "b": 0.75},
        whoosh_index_dir=WHOOSH_INDEX_DIR,
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_deployment=AZURE_OPENAI_4O,
        azure_api_version=AZURE_API_VERSION
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
        raw_vector_results = agent_vector_search.query(query)
        #log the full raw results
        logger.info(f"Raw raw_vector_results: {raw_vector_results}")
        raw_keyword_results= agent_keyword_search.retrieve_documents(query, top_k=5)
        logger.info(f"Raw raw_keyword_results: {raw_keyword_results}")
        #combined the two results
        combined_results = raw_vector_results + raw_keyword_results
        
        # Remove duplicates based on normalized page content
        seen_contents = set()
        unique_documents = []

        for doc in combined_results:
            # Normalize the page_content
            normalized_content = doc.page_content.strip().lower()
            
            if normalized_content not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(normalized_content)
                
        logger.info(f"Combined raw results: {unique_documents}")
        summary = llm_processor.process_query_response(query, unique_documents)
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