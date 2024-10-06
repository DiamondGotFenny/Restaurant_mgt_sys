#vector_store_agent.py
import os
import glob
import sys
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
from document_processor import DocumentProcessor
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from logger_config import setup_logger
class VectorStoreAgent:
    def __init__(
        self,
        pdf_directory: str,
        persist_directory: str,
        azure_openai_api_key: str,
        azure_openai_endpoint: str,
        azure_openai_embedding_deployment: str,
        log_file: str = "vector_store_agent.log",
    ):
        """
        Initializes the VectorStoreAgent using the unified DocumentProcessor.
        """
        self.logger = setup_logger(log_file)
        self.persist_directory = persist_directory

        # Initialize DocumentProcessor
        self.document_processor = DocumentProcessor(
            pdf_directory=pdf_directory,
            logger_file=log_file
        )
        self.documents = self.document_processor.load_and_split_documents()

        if not self.documents:
            self.logger.error("No documents to process. Exiting.")
            sys.exit(1)

        # Initialize Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment=azure_openai_embedding_deployment,
        )

        # Initialize or load existing vector store
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.logger.info(f"Loading existing vector store from {self.persist_directory}...")
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.logger.info("Vector store not found. Creating a new one...")
            self.vector_store = Chroma.from_documents(
                self.documents,
                self.embeddings,
                persist_directory=self.persist_directory
            )
            self.vector_store.persist()
            self.logger.info(f"Vector store created and saved to {self.persist_directory}.")

    def query(self, query_text: str, top_k: int = 5) -> List:
        """
        Queries the vector store with the provided text.
        """
        self.logger.info(f"Querying vector store for: '{query_text}'")
        results = self.vector_store.max_marginal_relevance_search(query_text, k=3, fetch_k=top_k)
        self.logger.info(f"Found {len(results)} results.")
        return results
    
    
def test_module(agent: VectorStoreAgent):
    """
    A test method to interactively query the vector store via the terminal.
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
        results = agent.query(query)
        if not results:
            print("No results found.")
            continue
        print(f"Top {len(results)} results:")
        for idx, doc in enumerate(results, 1):
            # Display a snippet or page number if available
            if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                agent.logger.info(f"Page {doc.metadata['page']}")
            #log the metadata, source and the whole content
            agent.logger.info(f"page: {doc.metadata['page']}")
            agent.logger.info(f"Source: {doc.metadata['source']}")
            agent.logger.info(f"Content: {doc.page_content}")
            agent.logger.info("-" * 40)
def main():
    """
    Main function to initialize the agent and start the test module.
    """
    _ = load_dotenv(find_dotenv())
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("OPENAI_API_BASE")
    os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")
    os.environ["AZURE_OPENAI_EMBEDDING_MODEL"] = os.getenv("OPENAI_EMBEDDING_MODEL")
    # Configuration - Replace these with your actual paths and Azure OpenAI credentials
    PDF_DIRECTORY = ".././data/Restaurants_data"  # e.g., "./pdfs"
    PERSIST_DIRECTORY = ".././data/vectorDB/chroma"
    AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"] 
    AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]
    # Validate directories
    if not os.path.isdir(PDF_DIRECTORY):
        print(f"PDF directory does not exist: {PDF_DIRECTORY}")
        sys.exit(1)
    if not os.path.exists(os.path.dirname(PERSIST_DIRECTORY)):
        try:
            os.makedirs(os.path.dirname(PERSIST_DIRECTORY), exist_ok=True)
            print(f"Created directory for vector store: {os.path.dirname(PERSIST_DIRECTORY)}")
        except Exception as e:
            print(f"Failed to create directory {os.path.dirname(PERSIST_DIRECTORY)}: {e}")
            sys.exit(1)
    # Initialize the VectorStoreAgent
    agent = VectorStoreAgent(
        pdf_directory=PDF_DIRECTORY,
        persist_directory=PERSIST_DIRECTORY,
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_embedding_deployment=AZURE_OPENAI_DEPLOYMENT,
        log_file="vector_store_agent_test.log"
    )
    # Start the test module
    test_module(agent)
if __name__ == "__main__":
    main()