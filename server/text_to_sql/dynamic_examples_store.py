# dynamic_examples_store.py
import os
import json
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from server.logger_config import setup_logger
from langchain_core.documents import Document

class DynamicExamplesStore:
    def __init__(
        self,
        examples_file: str,
        persist_directory: str,
        azure_openai_api_key: str,
        azure_openai_endpoint: str,
        azure_openai_embedding_deployment: str,
        log_file: str = "server/logs/dynamic_examples_store.log",
        max_chunk_size: int = 500
    ):
        """
        Initialize the DynamicExamplesStore.
        
        Args:
            examples_file (str): Path to the JSON file containing examples
            persist_directory (str): Directory to persist the vector store
            azure_openai_api_key (str): Azure OpenAI API key
            azure_openai_endpoint (str): Azure OpenAI endpoint
            azure_openai_embedding_deployment (str): Azure OpenAI embedding model deployment name
            log_file (str): Path to log file
            max_chunk_size (int): Maximum chunk size for splitting examples
        """
        self.logger = setup_logger(log_file)
        self.persist_directory = persist_directory
        self.examples_file = examples_file
        
        # Load examples
        self.examples = self._load_examples()
        
        # Initialize splitter
        self.splitter = RecursiveJsonSplitter(
            max_chunk_size=max_chunk_size
        )
        
        # Process examples into chunks
        self.example_chunks = self._process_examples()
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model=azure_openai_embedding_deployment,
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment=azure_openai_embedding_deployment,
        )
        
        # Initialize or load vector store
        self._initialize_vector_store()

    def _load_examples(self) -> Dict[str, Any]:
        """Load examples from JSON file."""
        try:
            with open(self.examples_file, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            self.logger.info(f"Successfully loaded {len(examples.get('cases', []))} examples")
            return examples
        except Exception as e:
            self.logger.error(f"Error loading examples: {str(e)}")
            raise

    def _process_examples(self) -> List[Dict[str, Any]]:
        """Process examples into chunks suitable for embedding."""
        try:
            # Create documents from examples
            docs = self.splitter.create_documents(texts=[self.examples])
            
            # Convert documents to structured chunks
            processed_chunks = []
            for doc in docs:
                # Extract the relevant parts from the chunk
                try:
                    # Parse the page_content as JSON
                    chunk_data = json.loads(doc.page_content)
                    if 'cases' in chunk_data:
                        for case in chunk_data['cases']:
                            processed_chunks.append({
                                'question': case.get('question', ''),
                                'analysis': case.get('analysis', ''),
                                'query_strategy': case.get('query_strategy', ''),
                                'content': f"""Question: {case.get('question', '')}
    Analysis: {case.get('analysis', '')}
    Query Strategy: {case.get('query_strategy', '')}"""
                            })
                except json.JSONDecodeError:
                    self.logger.warning(f"Could not parse document content as JSON: {doc.page_content[:100]}...")
                    continue
                
            self.logger.info(f"Processed {len(processed_chunks)} example chunks")
            return processed_chunks
        except Exception as e:
            self.logger.error(f"Error processing examples: {str(e)}")
            raise

    def _initialize_vector_store(self):
        """Initialize or load existing vector store."""
        try:
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                self.logger.info(f"Loading existing vector store from {self.persist_directory}")
                self.vector_store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                self.logger.info("Creating new vector store")
                # Create documents for vector store
                documents = [
                Document(
                    page_content=chunk['content'],
                    metadata={
                        "question": chunk['question'],
                        "analysis": chunk['analysis'],
                        "query_strategy": chunk['query_strategy']
                    }
                )
                for chunk in self.example_chunks
            ]
                
                self.vector_store = Chroma.from_documents(
                    documents,
                    self.embeddings,
                    persist_directory=self.persist_directory
                )
                self.vector_store.persist()
                self.logger.info(f"Vector store created and saved to {self.persist_directory}")
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def get_relevant_examples(self, query: str, k: int = 2) -> List[Dict[str, Any]]:
        """
        Get relevant examples for a given query.
        
        Args:
            query (str): The query text
            k (int): Number of examples to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of relevant examples
        """
        try:
            self.logger.info(f"Searching for examples relevant to: {query}")
            results = self.vector_store.max_marginal_relevance_search(
                query, 
                k=k,
                fetch_k=k*2
            )
            
            relevant_examples = []
            for doc in results:
                example = {
                    "question": doc.metadata['question'],
                    "analysis": doc.metadata['analysis'],
                    "query_strategy": doc.metadata['query_strategy']
                }
                relevant_examples.append(example)
            
            self.logger.info(f"Found {len(relevant_examples)} relevant examples")
            return relevant_examples
        except Exception as e:
            self.logger.error(f"Error retrieving examples: {str(e)}")
            raise

def test_dynamic_examples_store():
    """
    Test function for the DynamicExamplesStore.
    This function allows interactive testing of the vector store search functionality.
    """
    import os
    from dotenv import load_dotenv, find_dotenv
    
    # Load environment variables
    load_dotenv(find_dotenv())
    
    # Get environment variables
    azure_openai_api_key = os.getenv("OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("OPENAI_API_BASE")
    azure_openai_embedding_deployment = os.getenv("OPENAI_EMBEDDING_MODEL")
    
    # Initialize the store
    try:
        # Get the absolute path of the current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Current directory: {current_dir}")
        # Construct the absolute path to dynamic_examples.json
        examples_file_path = os.path.join(current_dir, 'dynamic_examples.json')

        # Construct the absolute path to the dynamic_examples_vectorDB directory
        persist_dir_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'dynamic_examples_vectorDB'))
        
        store = DynamicExamplesStore(
            examples_file=examples_file_path,
            persist_directory=persist_dir_path, 
            azure_openai_api_key=azure_openai_api_key,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_embedding_deployment=azure_openai_embedding_deployment
        )

        print("\nDynamic Examples Store initialized successfully!")
        print("Enter your questions (type 'exit' to quit).\n")
        
        while True:
            # Get user input
            query = input("\nEnter your question: ").strip()
            
            # Check for exit command
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nExiting...\n")
                break
            
            if not query:
                print("Please enter a valid question.")
                continue
            
            # Get relevant examples
            print("\nSearching for relevant examples...")
            examples = store.get_relevant_examples(query, k=2)
            
            # Display results
            print("\nFound relevant examples:")
            print("-" * 80)
            
            for i, example in enumerate(examples, 1):
                print(f"\nExample {i}:")
                print("\nQuestion:")
                print(example['question'])
                print("\nAnalysis:")
                print(example['analysis'])
                print("\nQuery Strategy:")
                print(example['query_strategy'])
                print("-" * 80)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        return

if __name__ == "__main__":
    test_dynamic_examples_store()