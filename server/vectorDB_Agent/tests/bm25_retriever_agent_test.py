import json
import sys
from dotenv import load_dotenv, find_dotenv
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bm25_retriever_agent import BM25RetrieverAgent
from query_pre_processor import LLMQueryPreProcessor
def test_module(agent: BM25RetrieverAgent):
    """
    A test method to interactively query the BM25 retriever via the terminal.
    """
    print("=== BM25 Retriever Agent Test Module ===")
    print("Enter 'exit' or 'quit' to terminate the test.\n")
    load_dotenv(find_dotenv())
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("OPENAI_API_BASE")
    os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")
    os.environ["AZURE_OPENAI_4O"] = os.getenv("OPENAI_MODEL_4o")

    AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"] 
    AZURE_OPENAI_4O = os.environ["AZURE_OPENAI_4o"]
    AZURE_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
    query_pre_processor = LLMQueryPreProcessor(
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_deployment=AZURE_OPENAI_4O,
        azure_api_version=AZURE_API_VERSION,
        log_file="llm_processor.log"
    )
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
            print("Empty query. Please enter a valid string.")
            continue

        try:
            # Process the query using the query pre-processor to extract entities
            query_json = query_pre_processor.process_query(user_input)
            if not isinstance(query_json, dict):
                raise ValueError("Input must be a JSON object with 'entities'.")

            # Validate presence of 'entities'
            if 'entities' not in query_json or not query_json['entities']:
                raise ValueError("JSON object must contain at least 'entities'.")

            # Query the BM25RetrieverAgent
            results = agent.query_documents(query=query_json, top_k=5)
            if not results:
                print("No results found or an error occurred.")
                continue

            print(f"\nTop {len(results)} Relevant Documents:")
            for idx, doc in enumerate(results, start=1):
                page = doc.metadata.get('page', 'N/A')
                source = doc.metadata.get('source', 'Unknown Source')
                # Log the metadata and content for auditing/debugging
                agent.logger.info(f"Document {idx}:")
                agent.logger.info(f"Page: {page}, Source: {source}")
                agent.logger.info(f"Content: {doc.page_content}")
                agent.logger.info("-" * 60)
                # Display a snippet or relevant information to the user
                snippet_length = 200
                content = doc.page_content.replace('\n', ' ').strip()
                snippet = (content[:snippet_length] + '...') if len(content) > snippet_length else content
                print(f"\nDocument {idx}:")
                print(f"Page: {page}, Source: {source}")
                print(f"Snippet: {snippet}")
                print("-" * 60)
        except json.JSONDecodeError as jde:
            print(f"Invalid JSON format: {jde}")
            continue
        except ValueError as ve:
            print(f"Input Error: {ve}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue


def main():
    """
    Main function to initialize the BM25RetrieverAgent and start the test module.
    """
    # Load environment variables from a .env file if present
    load_dotenv(find_dotenv())

    # Configuration
    PDF_DIRECTORY = "../.././data/Restaurants_data"
    LOG_FILE = "bm25_retriever_agent.log"  # Log file path
    WHOOSH_INDEX_DIR = "../whoosh_index"       # Whoosh index directory

    # Validate directories
    if not os.path.isdir(PDF_DIRECTORY):
        print(f"PDF directory does not exist: {PDF_DIRECTORY}")
        sys.exit(1)

    # Initialize the BM25RetrieverAgent
    agent = BM25RetrieverAgent(
        pdf_directory=PDF_DIRECTORY,
        log_file=LOG_FILE,
        chunk_size=2000,
        chunk_overlap=200,
        bm25_params={"k1": 0.5, "b": 0.75},
        whoosh_index_dir=WHOOSH_INDEX_DIR
    )

    if agent.documents:
        # Start the test module
        test_module(agent)
    else:
        print("No documents loaded. Exiting.")


if __name__ == "__main__":
    main()