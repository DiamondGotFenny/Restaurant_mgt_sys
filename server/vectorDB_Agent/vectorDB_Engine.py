# vectorDB_Engine.py
import os
import sys
from dotenv import load_dotenv, find_dotenv
from  vectorDB_Agent.vector_store_agent import VectorStoreAgent
from  vectorDB_Agent.llm_post_processor import LLMProcessor
from  logger_config import setup_logger
from  vectorDB_Agent.combined_keyword_retriever import CombinedKeywordRetriever

class VectorDBEngine:
    def __init__(self):
        """
        Initializes the VectorDBEngine by setting up environment variables,
        directories, logger, and agents.
        """
        # Load environment variables
        load_dotenv(find_dotenv())

        # Define the base directory as the parent of the current file's directory
        self.BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Define paths using absolute paths
        self.LOG_FILE = os.path.join(self.BASE_DIR, 'logs', 'hybrid_search_agent.log')
        self.PDF_DIRECTORY = os.path.join(self.BASE_DIR, 'data', 'Restaurants_data')
        self.PERSIST_DIRECTORY = os.path.join(self.BASE_DIR, 'data', 'vectorDB', 'chroma')
        self.WHOOSH_INDEX_DIR = os.path.join(self.BASE_DIR, 'data', 'whoosh_index') 
      
        # Retrieve environment variables
        self.AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_EMBEDDING = os.getenv("OPENAI_EMBEDDING_MODEL")
        self.AZURE_OPENAI_4O = os.getenv("OPENAI_MODEL_4o")
        self.AZURE_OPENAI_4OMINI = os.getenv("OPENAI_MODEL_4OMINI")
        self.AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
        
        # Setup logger
        self.logger = setup_logger(self.LOG_FILE)
        
        # Log the paths for debugging
        self.logger.info(f"BASE_DIR: {self.BASE_DIR}")
        self.logger.info(f"LOG_FILE: {self.LOG_FILE}")
        self.logger.info(f"PDF_DIRECTORY: {self.PDF_DIRECTORY}")
        self.logger.info(f"PERSIST_DIRECTORY: {self.PERSIST_DIRECTORY}")
        self.logger.info(f"WHOOSH_INDEX_DIR: {self.WHOOSH_INDEX_DIR}")
    
        # Validate and create necessary directories
        self._validate_and_create_directories()
    
        # Initialize the VectorStoreAgent
        self.agent_vector_search = VectorStoreAgent(
            pdf_directory=self.PDF_DIRECTORY,
            persist_directory=self.PERSIST_DIRECTORY,
            azure_openai_api_key=self.AZURE_OPENAI_API_KEY,
            azure_openai_endpoint=self.AZURE_OPENAI_ENDPOINT,
            azure_openai_embedding_deployment=self.AZURE_OPENAI_EMBEDDING,
            log_file=self.LOG_FILE
        )
        
        # Initialize the CombinedKeywordRetriever
        self.agent_keyword_search = CombinedKeywordRetriever(
            pdf_directory=self.PDF_DIRECTORY,
            log_file_pre_processor=os.path.join(self.BASE_DIR, 'logs', 'llm_processor.log'),
            log_file_retriever=os.path.join(self.BASE_DIR, 'logs', 'bm25_retriever_agent.log'),
            chunk_size=2000,
            chunk_overlap=400,
            bm25_params={"k1": 0.5, "b": 0.75},
            whoosh_index_dir=self.WHOOSH_INDEX_DIR,
            azure_openai_api_key=self.AZURE_OPENAI_API_KEY,
            azure_openai_endpoint=self.AZURE_OPENAI_ENDPOINT,
            azure_openai_deployment=self.AZURE_OPENAI_4O,
            azure_api_version=self.AZURE_API_VERSION
        )
    
        # Initialize the LLMProcessor
        self.llm_processor = LLMProcessor(
            azure_openai_api_key=self.AZURE_OPENAI_API_KEY,
            azure_openai_endpoint=self.AZURE_OPENAI_ENDPOINT,
            azure_openai_deployment=self.AZURE_OPENAI_4OMINI,
            azure_api_version=self.AZURE_API_VERSION,
            log_file=self.LOG_FILE
        )
    
    def _validate_and_create_directories(self):
        """
        Validates the existence of required directories and creates them if necessary.
        """
        directories = {
            "PDF_DIRECTORY": self.PDF_DIRECTORY,
            "PERSIST_DIRECTORY": self.PERSIST_DIRECTORY,
            "LOG_FILE_DIR": os.path.dirname(self.LOG_FILE),
            "WHOOSH_INDEX_DIR": self.WHOOSH_INDEX_DIR
        }
        
        for name, path in directories.items():
            if name == "PDF_DIRECTORY":
                if not os.path.isdir(path):
                    self.logger.error(f"{name} does not exist: {path}")
                    sys.exit(1)
            else:
                if not os.path.exists(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                        self.logger.info(f"Created directory for {name}: {path}")
                    except Exception as e:
                        self.logger.error(f"Failed to create directory {path}: {e}")
                        sys.exit(1)
    
    def return_unique_documents(self, query):
        """
        Accepts a query and returns a list of unique documents retrieved 
        from both vector and keyword searches.
        
        Args:
            query (str): The search query.
        
        Returns:
            list: A list of unique document objects.
        """
        raw_vector_results = self.agent_vector_search.query(query)
        raw_keyword_results = self.agent_keyword_search.retrieve_documents(query, top_k=5)
        combined_results = raw_vector_results + raw_keyword_results
        seen_contents = set()
        unique_documents = []
        for doc in combined_results:
            normalized_content = doc.page_content.strip().lower()
            if normalized_content not in seen_contents:
                unique_documents.append(doc)
                seen_contents.add(normalized_content)
        return unique_documents
    
    def qa_chain(self, query):
        """
        Processes a query through vector and keyword searches, summarizes the results 
        using the LLMProcessor, and returns the summary.
        
        Args:
            query (str): The search query.
        
        Returns:
            str: The summarized response.
        """
        raw_vector_results = self.agent_vector_search.query(query)
        # Log the full raw results
        self.logger.info(f"Raw vector results: {raw_vector_results}")
        
        raw_keyword_results = self.agent_keyword_search.retrieve_documents(query, top_k=5)
        self.logger.info(f"Raw keyword results: {raw_keyword_results}")
        
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
                
        self.logger.info(f"Combined unique documents: {unique_documents}")
        summary = self.llm_processor.process_query_response(query, unique_documents)
        return summary
    
def test_module():
        """
        A test method to interactively query the vector store via the terminal.
        Displays both raw and summarized results, handles edge cases, and logs interactions.
        """
        print("=== Vector Store Agent Test Module ===")
        print("Enter 'exit' to quit.")
        os.environ.pop('OPENAI_API_BASE', None) 
        engine = VectorDBEngine()
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
                response = engine.qa_chain(query)
            except Exception as e:
                print(f"An error occurred during the query: {e}")
                continue

            # Display the summarized response
            print("\n--- Summarized Response ---")
            print(response)
            print("\n--- End of Response ---\n")

   

if __name__ == "__main__":
     # Start the test module
    test_module()