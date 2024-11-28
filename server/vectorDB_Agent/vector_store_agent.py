#vector_store_agent.py
import os
import glob
import sys
from typing import List
from  vectorDB_Agent.document_processor import DocumentProcessor
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from  logger_config import setup_logger
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
            model=azure_openai_embedding_deployment,
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
        results = self.vector_store.max_marginal_relevance_search(query_text, k=5, fetch_k=top_k)
        self.logger.info(f"Found {len(results)} results.")
        return results
    
    
