# llm_post_processor.py

import os
from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from logger_config import setup_logger
from collections import namedtuple

_ = load_dotenv(find_dotenv())

class LLMProcessor:
    def __init__(
        self,
        azure_openai_api_key: str,
        azure_openai_endpoint: str,
        azure_openai_deployment: str,
        azure_api_version: str,
        log_file: str,
    ):
        """
        Initializes the LLMProcessor.

        Args:
            azure_openai_api_key (str): API key for Azure OpenAI.
            azure_openai_endpoint (str): Endpoint for Azure OpenAI.
            azure_openai_deployment (str): Deployment name for Azure OpenAI.
            log_file (str): Path to the log file.
        """
        self.logger = setup_logger(log_file)
        self.llm = AzureChatOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_deployment,
            api_version=azure_api_version,
            temperature=0.1,  # Low temperature for more deterministic outputs
            max_tokens=3000    # Adjust based on expected summary length
        )
        self.prompt = PromptTemplate(
            input_variables=["query", "documents"],
            template="""
You are a Data Processing Assistant. Your task is to extract and organize all relevant information from the provided documents based on the user's query. The extracted information will be passed to the next data processor, so it is crucial to preserve all details accurately. Lossless information retention is essential at this stage. Adhere to the following guidelines:

**Guidelines**
1. Understand the Query:
Carefully read and comprehend the user’s query to determine the specific information being requested.
Identify keywords and key concepts, such as locations, types of entities (e.g., restaurants, events, products), and any specific requirements or criteria.
2. Extract Relevant Information:
Analyze the provided documents to identify and extract all information directly related to the user’s query.
Retrieve full relevant information from each pertinent section without summarizing or abstracting.
Include key entities and details relevant to the query, such as:
    -Name/Title
    -Address/Location
    -Phone Number/Contact Information
    -Category/Type
    -Business Hours/Operational Details
    -Website or Email Address (if available)
    -Additional Details (e.g., specialties, ambiance, ratings)
    
3. Exclude Unrelated Information:
Disregard any data that does not directly pertain to the subject of the user’s query.
Ignore mentions of unrelated topics, entities outside the scope of the query, or general information not pertinent.
4. Maintain Accuracy:
Do not fabricate, infer, or assume information not explicitly present in the documents.
Ensure all extracted details are accurate and verifiable against the source material.
Avoid any form of content generation that isn't supported by the source documents to prevent inaccuracies.
5. Organize the Output:
Present the information in a structured and clear format without using tables.
Use clearly labeled sections and bullet points for each entity to enhance readability.
Include the source document name and page number for each entry to allow the user to verify credibility.
6. Re-Ranking Criteria:
Prioritize entities based on criteria relevant to the query, such as:
Quality/Rating: Indicators like grades, reviews, or specific comments on quality.
Relevance to Requirements: Preference for entities that meet specified needs (e.g., no reservations needed, specific amenities).
Other Relevant Criteria: Depending on the query, such as distance, price range, popularity, etc.
List all relevant entities in each category or section, sorted first by the highest priority criteria, then by secondary criteria.
7. Quality Assurance:
Double-check all extracted information for completeness and accuracy.
Ensure no key entities or details related to the query are omitted.

**User Query:**
"{query}"

**Retrieved Documents:**
"{documents}"

**Relevant Information:**
"""
        )
        self.output_parser = StrOutputParser()
        self.llm_chain = self.prompt|self.llm|self.output_parser

    def process_query_response(self, query: str, raw_results: List, min_length: int = 20) -> str:
        """
        Processes raw vector store results using an LLM to extract and summarize the most relevant information.

        Args:
            query (str): The user's query.
            raw_results (List): The raw documents retrieved from the vector store.
            min_length (int): Minimum number of characters required to perform summarization.

        Returns:
            str: A distilled and summarized response or an appropriate message.
        """
        if not raw_results:
            self.logger.info("No results found for the query. Skipping summarization.")
            return "No relevant information found to answer your query."

        # Calculate total length of retrieved documents
        total_length = sum(len(doc.page_content) for doc in raw_results)
        if total_length < min_length:
            self.logger.info("Retrieved results are too short. Skipping summarization.")
            return "Insufficient information found to answer your query."

        # Concatenate the content of all retrieved documents
        #add source and page number to the content
        documents_content = "\n\n".join([f"Source: {doc.metadata['source']}\nPage: {doc.metadata['page']}\n{doc.page_content}" for doc in raw_results])

        try:
            # Use the RunnableSequence to process the documents
            summary = self.llm_chain.invoke({
                "query": query,
                "documents": documents_content
            })
            self.logger.info(f" Raw Retrieved Documents: {documents_content} ")  
            self.logger.info("Summarization successful.")
            self.logger.info(f"Summarized Response: {summary.strip()}")
            return summary.strip()
        except Exception as e:
            self.logger.error(f"Error during summarization: {e}")
            return "An error occurred while processing your request."
        
