"""
BM25 Retriever Agent Module with Structured JSON Input for Entities Only

This module provides functionality to perform BM25-based document retrieval
with support for structured JSON input containing only entities to optimize search
results. It allows loading and splitting PDF documents, annotating them with metadata,
initializing the BM25 retriever, setting up Whoosh indexing, and performing
queries to retrieve relevant documents based on specified entities.
"""

import os
import glob
import sys
import json
from typing import List, Optional, Tuple, Dict, Any
from logger_config import setup_logger
from dotenv import load_dotenv, find_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup
from whoosh.analysis import StemmingAnalyzer
from whoosh.scoring import BM25F
from query_pre_processor import LLMQueryPreProcessor
import nltk

# Ensure the necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class BM25RetrieverAgent:
    def __init__(
        self,
        pdf_directory: str,
        log_file: str = "bm25_retriever_agent.log",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        bm25_params: Optional[dict] = None,
        whoosh_index_dir: str = "whoosh_index"
    ):
        """
        Initializes the BM25RetrieverAgent with BM25 and Whoosh capabilities.

        Args:
            pdf_directory (str): Path to the directory containing PDF files.
            log_file (str, optional): Path to the log file. Defaults to "bm25_retriever_agent.log".
            chunk_size (int, optional): Maximum characters per chunk. Defaults to 1000.
            chunk_overlap (int, optional): Overlap to maintain context between chunks. Defaults to 200.
            bm25_params (Optional[dict], optional): Parameters for the BM25 algorithm (e.g., k1, b).
                Defaults to {"k1": 0.5, "b": 0.75}.
            whoosh_index_dir (str, optional): Directory to store Whoosh index. Defaults to "whoosh_index".
        """
        self.logger = setup_logger(log_file)
        self.pdf_directory = pdf_directory
        # Use a sentence-based splitter to preserve context and reduce duplication
        self.text_splitter = NLTKTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
            length_function=len
        )
        self.bm25_params = bm25_params if bm25_params else {"k1": 0.5, "b": 0.75}
        self.retriever = None
        self.documents = self._load_and_prepare_documents()

        if self.documents:
            self._initialize_bm25_retriever()
            self._initialize_whoosh_index(whoosh_index_dir)
            self.logger.info("BM25 Retriever and Whoosh Index initialized successfully.")
        else:
            self.logger.error("No documents loaded. BM25 Retriever and Whoosh Index not initialized.")

    def _load_and_prepare_documents(self) -> List[Document]:
        """
        Loads, splits, and annotates PDF documents from the specified directory.

        Returns:
            List[Document]: A list of annotated Document objects.
        """
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))
        if not pdf_files:
            self.logger.warning(f"No PDF files found in directory: {self.pdf_directory}")
            return []

        documents = []
        for pdf_file in pdf_files:
            self.logger.info(f"Loading PDF file: {pdf_file}")
            try:
                loader = PyPDFLoader(pdf_file)
                raw_docs = loader.load()
            except Exception as e:
                self.logger.error(f"Failed to load {pdf_file}: {e}")
                continue

            # Use the text splitter to split documents into smaller chunks
            split_docs = self.text_splitter.split_documents(raw_docs)
            self.logger.info(f"Loaded and split {len(split_docs)} documents from {pdf_file}.")

            # Add metadata to each chunk without altering the page_content
            for idx, doc in enumerate(split_docs, start=1):
                source = os.path.basename(pdf_file)
                metadata = {
                    "page": idx,
                    "source": source
                }
                annotated_doc = Document(page_content=doc.page_content, metadata=metadata)
                documents.append(annotated_doc)

        self.logger.info(f"Total documents loaded and split: {len(documents)}")
        return documents

    def _initialize_bm25_retriever(self):
        """
        Initializes the BM25 retriever from LangChain.
        """
        self.retriever = BM25Retriever.from_documents(
            self.documents,
            bm25_params=self.bm25_params
        )

    def _initialize_whoosh_index(self, whoosh_index_dir: str):
        """
        Initializes the Whoosh index for phrase matching with proximity searches.

        Args:
            whoosh_index_dir (str): Directory to store Whoosh index.
        """
        if not os.path.exists(whoosh_index_dir):
            os.mkdir(whoosh_index_dir)
            self.logger.info(f"Created Whoosh index directory at: {whoosh_index_dir}")

        # Define Whoosh schema with StemmingAnalyzer
        schema = Schema(
            id=ID(stored=True, unique=True),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            page=ID(stored=True),
            source=TEXT(stored=True)
        )

        # Create or open existing index
        if index.exists_in(whoosh_index_dir):
            self.index = index.open_dir(whoosh_index_dir)
            self.logger.info("Opened existing Whoosh index.")
        else:
            self.index = index.create_in(whoosh_index_dir, schema)
            self.logger.info("Created new Whoosh index.")

        # Index documents
        writer = self.index.writer()
        for doc_id, doc in enumerate(self.documents, start=1):
            writer.update_document(
                id=str(doc_id),
                content=doc.page_content,
                page=str(doc.metadata.get('page', 'N/A')),
                source=doc.metadata.get('source', 'Unknown Source')
            )
        writer.commit()
        self.logger.info("Whoosh indexing completed.")

    def _search_whoosh(self, query: str, top_k: int = 5) -> List[Tuple[float, Document]]:
        """
        Performs phrase matching using Whoosh with proximity search and returns relevant documents.

        Args:
            query (str): The search query with possible boosts and proximity operators.
            top_k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            List[Tuple[float, Document]]: List of tuples containing score and Document.
        """
        with self.index.searcher(weighting=BM25F()) as searcher:
            # Initialize QueryParser
            parser = QueryParser("content", schema=self.index.schema, group=OrGroup.factory(0.9))
            # No NearPlugin to be added

            try:
                parsed_query = parser.parse(query)
            except Exception as e:
                self.logger.error(f"Failed to parse query '{query}': {e}")
                return []

            results = searcher.search(parsed_query, limit=top_k)
            retrieved_docs = []
            for hit in results:
                try:
                    doc_id = int(hit['id']) - 1  # Adjusting index to match documents list
                    if 0 <= doc_id < len(self.documents):
                        retrieved_docs.append((hit.score, self.documents[doc_id]))
                except (ValueError, IndexError) as e:
                    self.logger.error(f"Error retrieving document ID {hit['id']}: {e}")
            self.logger.info(f"Whoosh search executed successfully. Retrieved {len(retrieved_docs)} results.")
        return retrieved_docs

    def _construct_bm25_query_from_json(self, query_dict: Dict[str, Any], proximity: int = 5) -> str:
        """
        Constructs a BM25 query string focusing solely on entities.
        Multi-word entities include both original and reversed word orders to handle variations.

        Args:
            query_dict (Dict[str, Any]): Dictionary with 'entities' list.
            proximity (int): The maximum number of words allowed between terms in a proximity search.

        Returns:
            str: Constructed BM25 query string.
        """
        if not isinstance(query_dict, dict):
            raise ValueError("Input must be a dictionary with 'entities'.")
        
        entities = query_dict.get("entities", [])
        if not isinstance(entities, list) or not all(isinstance(ent, str) for ent in entities):
            raise ValueError("'entities' must be a list of strings.")
        
        # Define boosting factor for entities
        ENTITY_BOOST = 3
        
        # Function to create proximity queries, including reversed word order for multi-word entities
        def create_proximity_queries(entity: str, boost: int) -> str:
            words = entity.strip().split()
            if len(words) > 1:
                # Original Order: "curry chicken"~5^3
                prox_query_1 = f'"{entity}"~{proximity}^{boost}'
                # Reversed Order: "chicken curry"~5^3
                reversed_entity = " ".join(reversed(words))
                prox_query_2 = f'"{reversed_entity}"~{proximity}^{boost}'
                # Combine both with OR
                prox_query = f'({prox_query_1} OR {prox_query_2})'
                return prox_query
            else:
                # Single word: "indian"^3
                return f'"{entity}"^{boost}'
        
        # Apply boosting and create queries for all entities
        boosted_entities = [create_proximity_queries(entity, ENTITY_BOOST) for entity in entities if entity]
        
        # Combine all boosted entities with OR (documents matching any entity)
        if boosted_entities:
            bm25_query = " OR ".join(boosted_entities)
        else:
            bm25_query = ""
        
        self.logger.debug(f"Boosted Entities: {boosted_entities}")
        self.logger.debug(f"Constructed BM25 Query: {bm25_query}")
        return bm25_query

    def query_documents(self, query: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        Performs a BM25 query using only entities to retrieve documents.

        Args:
            query (Dict[str, Any]): The search query as a JSON object with 'entities'.
                                      Example:
                                      {
                                          "entities": ["indian", "curry chicken"]
                                      }
            top_k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            List[Document]: A list of Document objects ranked by relevance.
        """
        if not self.retriever or not hasattr(self, 'index'):
            self.logger.error("BM25 Retriever or Whoosh Index is not initialized. Cannot perform queries.")
            return []

        if not isinstance(query, dict):
            self.logger.error("Query must be a dictionary with 'entities'.")
            return []

        try:
            # Construct the BM25 query string from the JSON object
            bm25_query = self._construct_bm25_query_from_json(query)
            if not bm25_query:
                self.logger.error("Constructed BM25 query is empty. Check your 'entities'.")
                return []

            self.logger.info(f"Constructed BM25 Query: {bm25_query}")

            # BM25-based retrieval using Whoosh
            bm25_results = self._search_whoosh(query=bm25_query, top_k=top_k)
            self.logger.info(f"BM25 query executed successfully. Retrieved {len(bm25_results)} results.")

            self.logger.info(f"Retrieved {len(bm25_results)} BM25 results.")

            return [doc for score, doc in bm25_results]

        except ValueError as ve:
            self.logger.error(f"ValueError during query_documents: {ve}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to execute query '{query}': {e}")
            return []


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
    PDF_DIRECTORY = ".././data/Restaurants_data"
    LOG_FILE = "bm25_retriever_agent.log"  # Log file path
    WHOOSH_INDEX_DIR = "whoosh_index"       # Whoosh index directory

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