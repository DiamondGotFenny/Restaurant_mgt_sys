#bm25_retriever_agent.py
"""
BM25 Retriever Agent Module with Structured JSON Input for Entities Only

This module provides functionality to perform BM25-based document retrieval
with support for structured JSON input containing only entities to optimize search
results. It allows loading and splitting PDF documents, annotating them with metadata,
initializing the BM25 retriever, setting up Whoosh indexing, and performing
queries to retrieve relevant documents based on specified entities.
"""

import os
import sys
from typing import List, Optional, Dict, Any
from server.logger_config import setup_logger
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from server.vectorDB_Agent.document_processor import DocumentProcessor
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup
from whoosh.scoring import BM25F



class BM25RetrieverAgent:
    def __init__(
        self,
        pdf_directory: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        bm25_params: Optional[dict] = None,
        whoosh_index_dir: str = "whoosh_index",
        log_file: str = "bm25_retriever_agent.log",
    ):
        """
        Initializes the BM25RetrieverAgent using the unified DocumentProcessor.
        """
        self.logger = setup_logger(log_file)
        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.bm25_params = bm25_params if bm25_params else {"k1": 0.5, "b": 0.75}
        self.whoosh_index_dir = whoosh_index_dir

        # Initialize DocumentProcessor
        self.document_processor = DocumentProcessor(
            pdf_directory=pdf_directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            logger_file=log_file
        )
        self.documents = self.document_processor.load_and_split_documents()

        if not self.documents:
            self.logger.error("No documents to process. Exiting.")
            sys.exit(1)

        # Initialize BM25 Retriever
        self.retriever = BM25Retriever.from_documents(
            self.documents,
            bm25_params=self.bm25_params
        )

        # Initialize Whoosh Index
        self._initialize_whoosh_index()

    def _initialize_whoosh_index(self):
        """
        Initializes the Whoosh index for BM25 retrieval.
        """
        if not os.path.exists(self.whoosh_index_dir):
            os.mkdir(self.whoosh_index_dir)
            self.logger.info(f"Created Whoosh index directory at: {self.whoosh_index_dir}")

        # Define Whoosh schema
        schema = Schema(
            id=ID(stored=True, unique=True),
            content=TEXT(stored=True),
            page=ID(stored=True),
            source=TEXT(stored=True)
        )

        # Create or open existing index
        if index.exists_in(self.whoosh_index_dir):
            self.index = index.open_dir(self.whoosh_index_dir)
            self.logger.info("Opened existing Whoosh index.")
        else:
            self.index = index.create_in(self.whoosh_index_dir, schema)
            self.logger.info("Created new Whoosh index.")

            # Index documents
            writer = self.index.writer()
            for doc_id, doc in enumerate(self.documents, start=1):
                writer.add_document(
                    id=str(doc_id),
                    content=doc.page_content,
                    page=str(doc.metadata.get('page', 'N/A')),
                    source=doc.metadata.get('source', 'Unknown Source')
                )
            writer.commit()
            self.logger.info("Whoosh indexing completed.")

    def query_documents(self, query: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        Performs a BM25 query using only entities to retrieve documents.
        """
        if not isinstance(query, dict):
            self.logger.error("Query must be a dictionary with 'entities'.")
            return []

        entities = query.get("entities", [])
        if not entities:
            self.logger.warning("No entities provided in the query.")
            return []

        # Construct BM25 query string
        bm25_query = " OR ".join([f'"{entity}"' for entity in entities])
        self.logger.info(f"Constructed BM25 Query: {bm25_query}")

        # Perform search
        with self.index.searcher(weighting=BM25F()) as searcher:
            parser = QueryParser("content", schema=self.index.schema, group=OrGroup.factory(0.9))
            try:
                parsed_query = parser.parse(bm25_query)
            except Exception as e:
                self.logger.error(f"Failed to parse query '{bm25_query}': {e}")
                return []

            results = searcher.search(parsed_query, limit=top_k)
            retrieved_docs = []
            for hit in results:
                doc_id = int(hit['id']) - 1  # Adjusting index
                if 0 <= doc_id < len(self.documents):
                    retrieved_docs.append(self.documents[doc_id])

        self.logger.info(f"Retrieved {len(retrieved_docs)} documents via BM25.")
        return retrieved_docs


