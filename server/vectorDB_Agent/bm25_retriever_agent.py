"""
BM25 Retriever Agent Module

This module provides functionality to perform BM25-based document retrieval.
It allows loading and splitting PDF documents, annotating them with metadata,
initializing the BM25 retriever, and performing queries to retrieve relevant documents.

Prerequisites:
    Install required packages:
        pip install langchain
        pip install rank_bm25
        pip install langchain-community
        pip install python-dotenv
        pip install pypdf
"""

import os
import glob
import sys
from logger_config import setup_logger
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class BM25RetrieverAgent:
    def __init__(
        self,
        pdf_directory: str,
        log_file: str = "bm25_retriever_agent.log",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        bm25_params: Optional[dict] = None
    ):
        """
        Initializes the BM25RetrieverAgent.

        Args:
            pdf_directory (str): Path to the directory containing PDF files.
            log_file (str, optional): Path to the log file. Defaults to "bm25_retriever_agent.log".
            chunk_size (int, optional): Maximum characters per chunk. Defaults to 1000.
            chunk_overlap (int, optional): Overlap to maintain context between chunks. Defaults to 200.
            bm25_params (Optional[dict], optional): Parameters for the BM25 algorithm (e.g., k1, b).
                Defaults to {"k1": 1.5, "b": 0.75}.
        """
        self.logger = setup_logger(log_file)
        self.pdf_directory = pdf_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.bm25_params = bm25_params if bm25_params else {"k1": 1.5, "b": 0.75}
        self.retriever = None
        self.documents = self._load_and_prepare_documents()

        if self.documents:
            self.retriever = BM25Retriever.from_documents(
                self.documents,
                bm25_params=self.bm25_params
            )
            self.logger.info("BM25 Retriever initialized successfully.")
        else:
            self.logger.error("No documents loaded. BM25 Retriever not initialized.")

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

            # Add metadata to each chunk
            for idx, doc in enumerate(split_docs, start=1):
                source = os.path.basename(pdf_file)
                metadata = {
                    "page": idx,
                    "source": source
                }
                annotated_content = f"Page: {idx}, Source: {source}\n{doc.page_content}"
                annotated_doc = Document(page_content=annotated_content, metadata=metadata)
                documents.append(annotated_doc)

        self.logger.info(f"Total documents loaded and split: {len(documents)}")
        return documents

    def query_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Performs a BM25 query to retrieve the top_k most relevant documents.

        Args:
            query (str): The search query.
            top_k (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            List[Document]: A list of Document objects ranked by relevance.
        """
        if not self.retriever:
            self.logger.error("BM25 Retriever is not initialized. Cannot perform queries.")
            return []

        try:
            results = self.retriever.invoke(query)
            self.logger.info(f"Query '{query}' executed successfully. Retrieved {len(results)} results.")
            return results[:top_k]
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
            #log the metadata, source and the whole content
            agent.logger.info(f"Document {idx}:")
            agent.logger.info(f"Page: {page}, Source: {source}")
            agent.logger.info(f"Content: {doc.page_content}")
            agent.logger.info("-" * 60)


def main():
    """
    Main function to initialize the BM25RetrieverAgent and start the test module.
    """
    # Load environment variables from a .env file if present
    load_dotenv(find_dotenv())

    # Configuration 
    PDF_DIRECTORY = ".././data/Restaurants_data"       
    LOG_FILE = "bm25_retriever_agent.log"  # Log file path

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
        bm25_params={"k1": 1.5, "b": 0.75}
    )

    if agent.documents:
        # Start the test module
        test_module(agent)
    else:
        print("No documents loaded. Exiting.")

if __name__ == "__main__":
    main()