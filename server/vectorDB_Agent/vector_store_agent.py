import os
import glob
import sys
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
        azure_openai_deployment: str,
        log_file: str,
    ):
        """
        Initializes the VectorStoreAgent.

        Args:
            pdf_directory (str): Path to the directory containing PDF files.
            persist_directory (str): Directory to persist and load Chroma data.
            azure_openai_api_key (str): API key for Azure OpenAI.
            azure_openai_endpoint (str): Endpoint for Azure OpenAI.
            azure_openai_deployment (str): Deployment name for Azure OpenAI.
            log_file (str): Path to the log file.
        """
        self.logger = setup_logger(log_file)
        self.pdf_directory = pdf_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,      # Maximum characters per chunk
        chunk_overlap=400,    # Overlap to maintain context
        length_function=len
    )
        self.persist_directory = persist_directory
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment=azure_openai_deployment,
        )
        self.vector_store = self._load_or_create_vector_store()

    def _load_or_create_vector_store(self) -> Chroma:
        """
        Loads the vector store if it exists; otherwise, creates a new one.

        Returns:
            Chroma: The loaded or newly created vector store.
        """
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.logger.info(f"Loading existing vector store from {self.persist_directory}...")
            vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            return vector_store
        else:
            self.logger.info("Vector store not found. Creating a new one...")
            documents = self._load_pdfs()
            if not documents:
                self.logger.error("No documents found. Please add PDF files to the specified directory.")
                sys.exit(1)
            vector_store = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=self.persist_directory
            )
            vector_store.persist()  # Persist the Chroma vector store
            self.logger.info(f"Vector store created and saved to {self.persist_directory}.")
            return vector_store

    def _load_pdfs(self) -> List:
        """
        Loads and parses all PDF files from the specified directory,
        and splits them into chunks using RecursiveCharacterTextSplitter.

        Returns:
            List: A list of split documents extracted from PDFs.
        """
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))
        if not pdf_files:
            self.logger.warning(f"No PDF files found in directory: {self.pdf_directory}")
            return []
        
        documents = []
        for pdf_file in pdf_files:
            self.logger.info(f"Loading PDF file: {pdf_file}")
            loader = PyPDFLoader(pdf_file)
            raw_docs = loader.load()
            
            # Use the text splitter to split documents into smaller chunks
            split_docs = self.text_splitter.split_documents(raw_docs)
            for doc in split_docs:
                doc.metadata['source'] = pdf_file
            documents.extend(split_docs)
            self.logger.info(f"Loaded and split {len(split_docs)} documents from {pdf_file}.")
        
        self.logger.info(f"Total documents loaded and split: {len(documents)}")
        return documents

    def query(self, query_text: str, top_k: int = 5) -> List:
        """
        Queries the vector store with the provided text.

        Args:
            query_text (str): The query string.
            top_k (int): Number of top results to return.

        Returns:
            List: A list of the top_k matching documents.
        """
        self.logger.info(f"Querying vector store for: '{query_text}'")
        results = self.vector_store.max_marginal_relevance_search(query_text, k=3,fetch_k=top_k)
        #results = self.vector_store.similarity_search_with_score(query_text, top_k)
        #log the raw results
        self.logger.info(f"Raw result retrieved successfully!")
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
            print(f"\nResult {idx}:")
            # Display a snippet or page number if available
            if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                print(f"[Page {doc.metadata['page']}]")
            print(doc.page_content[:1000] + '...' if len(doc.page_content) > 1000 else doc.page_content)
            print("-" * 40)


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
        azure_openai_deployment=AZURE_OPENAI_DEPLOYMENT,
        log_file="vector_store_agent_test.log"
    )

    # Start the test module
    test_module(agent)


if __name__ == "__main__":
    main()