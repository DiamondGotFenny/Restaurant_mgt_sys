# combined_keyword_retriever.py

from typing import List,Optional
from server.vectorDB_Agent.query_pre_processor import LLMQueryPreProcessor
from server.vectorDB_Agent.bm25_retriever_agent import BM25RetrieverAgent
from langchain.schema import Document

class CombinedKeywordRetriever:
    def __init__(
        self,
        pdf_directory: str,
        log_file_pre_processor: str = "llm_processor.log",
        log_file_retriever: str = "bm25_retriever_agent.log",
        chunk_size: int = 2000,
        chunk_overlap: int = 400,
        bm25_params: Optional[dict] = None,
        whoosh_index_dir: str = "whoosh_index",
        azure_openai_api_key: str='' ,
        azure_openai_endpoint: str='' ,
        azure_openai_deployment: str='',
        azure_api_version: str =''
    ):
        """
        Initializes the CombinedKeywordRetriever with both LLMQueryPreProcessor and BM25RetrieverAgent.
        """
        self.llm_pre_processor = LLMQueryPreProcessor(
            azure_openai_api_key=azure_openai_api_key,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_deployment=azure_openai_deployment,
            azure_api_version=azure_api_version,
            log_file=log_file_pre_processor
        )
        
        self.bm25_retriever = BM25RetrieverAgent(
            pdf_directory=pdf_directory,
            log_file=log_file_retriever,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            bm25_params=bm25_params,
            whoosh_index_dir=whoosh_index_dir
        )
    
    def retrieve_documents(self, user_query: str, top_k: int = 5) -> List[Document]:
        """
        Processes the user query to extract entities and retrieves relevant documents.
        
        Args:
            user_query (str): The user's search query.
            top_k (int, optional): Number of top results to return. Defaults to 5.
        
        Returns:
            List[Document]: A list of retrieved Document objects.
        """
        # Extract entities using LLMQueryPreProcessor
        entities = self.llm_pre_processor.process_query(user_query).get('entities', [])
        
        if not entities:
            # Handle cases where no entities are extracted
            self.llm_pre_processor.logger.warning("No entities extracted from the query.")
            return []
        
        # Retrieve documents using BM25RetrieverAgent
        documents = self.bm25_retriever.query_documents({'entities': entities}, top_k=top_k)
        
        return documents