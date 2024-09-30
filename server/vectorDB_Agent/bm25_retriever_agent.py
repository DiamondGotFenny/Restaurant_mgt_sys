"""
BM25 Retriever Agent Module with Phrase Matching via Whoosh

This module provides functionality to perform BM25-based document retrieval
and incorporate phrase matching using Whoosh. It allows loading and splitting
PDF documents, annotating them with metadata, initializing the BM25 retriever,
setting up Whoosh indexing, and performing queries to retrieve relevant documents.

Prerequisites:
    Install required packages:
        pip install langchain
        pip install rank_bm25
        pip install langchain-community
        pip install python-dotenv
        pip install pypdf
        pip install Whoosh
"""

import os
import glob
import sys
from logger_config import setup_logger
from typing import List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup
from whoosh.analysis import StemmingAnalyzer
from whoosh.query import And
from whoosh.scoring import BM25F
import nltk
nltk.download('punkt_tab')

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
        self.bm25_params = bm25_params if bm25_params else {"k1": 1.0, "b": 0.75}
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
        Initializes the Whoosh index for phrase matching.

        Args:
            whoosh_index_dir (str): Directory to store Whoosh index.
        """
        if not os.path.exists(whoosh_index_dir):
            os.mkdir(whoosh_index_dir)
            self.logger.info(f"Created Whoosh index directory at: {whoosh_index_dir}")

        # Define Whoosh schema
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
        Performs phrase matching using Whoosh and returns relevant documents.

        Args:
            query (str): The search query.
            top_k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            List[Tuple[float, Document]]: List of tuples containing score and Document.
        """
        with self.index.searcher(weighting=BM25F()) as searcher:
            # Use MultifieldParser to search in 'content' field
            parser = QueryParser("content", schema=self.index.schema, group=OrGroup)

            # Parse phrase query
            # To handle phrases, we use "quoted" query if applicable
            if '"' in query:
                parsed_query = parser.parse(query)
            else:
                # If no quotes, perform an AND search for all terms
                parsed_query = And([parser.parse(term) for term in query.split()])

            results = searcher.search(parsed_query, limit=top_k)
            retrieved_docs = []
            for hit in results:
                doc_id = int(hit['id']) - 1  # Adjusting index to match documents list
                if 0 <= doc_id < len(self.documents):
                    retrieved_docs.append((hit.score, self.documents[doc_id]))
            return retrieved_docs

    def query_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Performs a BM25 query and incorporates phrase matching to retrieve the top_k most relevant documents.

        Args:
            query (str): The search query.
            top_k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            List[Document]: A list of Document objects ranked by combined relevance.
        """
        if not self.retriever or not hasattr(self, 'index'):
            self.logger.error("BM25 Retriever or Whoosh Index is not initialized. Cannot perform queries.")
            return []

        try:
            # BM25 based retrieval
            bm25_results = self.retriever.invoke(query)[:top_k]
            self.logger.info(f"BM25 query '{query}' executed successfully. Retrieved {len(bm25_results)} results.")

            # Whoosh phrase matching
            whoosh_results = self._search_whoosh(query, top_k=top_k)
            self.logger.info(f"Whoosh phrase matching for query '{query}' retrieved {len(whoosh_results)} results.")

            # Combine results: Prefer Whoosh results for exact phrase matches
            combined_docs = []
            seen_sources = set()

            # Add Whoosh results first
            for score, doc in whoosh_results:
                key = (doc.metadata.get('source'), doc.metadata.get('page'))
                if key not in seen_sources:
                    combined_docs.append(doc)
                    seen_sources.add(key)
                if len(combined_docs) >= top_k:
                    break

            # Fill remaining slots with BM25 results
            for doc in bm25_results:
                key = (doc.metadata.get('source'), doc.metadata.get('page'))
                if key not in seen_sources:
                    combined_docs.append(doc)
                    seen_sources.add(key)
                if len(combined_docs) >= top_k:
                    break

            self.logger.info(f"Combined and de-duplicated results. Total returned: {len(combined_docs)}")
            return combined_docs

        except Exception as e:
            self.logger.error(f"Failed to execute query '{query}': {e}")
            return []


def test_module(agent: BM25RetrieverAgent):
    """
    A test method to interactively query the BM25 retriever via the terminal.
    """
    print("=== BM25 Retriever Agent Test Module ===")
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
        results = agent.query_documents(query=query, top_k=5)
        if not results:
            print("No results found.")
            continue
        print(f"\nTop {len(results)} Relevant Documents:")
        for idx, doc in enumerate(results, start=1):
            page = doc.metadata.get('page', 'N/A')
            source = doc.metadata.get('source', 'Unknown Source')
            # Log the metadata and content
            agent.logger.info(f"Document {idx}:")
            agent.logger.info(f"Page: {page}, Source: {source}")
            agent.logger.info(f"Content: {doc.page_content}")
            agent.logger.info("-" * 60)
            # Display a snippet or relevant information to the user
            snippet = (doc.page_content[:200] + '...') if len(doc.page_content) > 200 else doc.page_content
            print(f"\nDocument {idx}:")
            print(f"Page: {page}, Source: {source}")
            print(f"Snippet: {snippet}")
            print("-" * 60)


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

    # Optionally, clear existing Whoosh index to re-index
    # Uncomment the following lines if you need to rebuild the index
    # if os.path.exists(WHOOSH_INDEX_DIR):
    #     shutil.rmtree(WHOOSH_INDEX_DIR)
    #     print(f"Cleared existing Whoosh index at: {WHOOSH_INDEX_DIR}")

    # Initialize the BM25RetrieverAgent
    agent = BM25RetrieverAgent(
        pdf_directory=PDF_DIRECTORY,
        log_file=LOG_FILE,
        chunk_size=2000,
        chunk_overlap=200,
        bm25_params={"k1": 1.0, "b": 0.75},
        whoosh_index_dir=WHOOSH_INDEX_DIR
    )

    if agent.documents:
        # Start the test module
        test_module(agent)
    else:
        print("No documents loaded. Exiting.")


if __name__ == "__main__":
    main()