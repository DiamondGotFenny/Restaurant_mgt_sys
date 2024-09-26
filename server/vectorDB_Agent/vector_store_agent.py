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
from langchain_openai.chat_models import AzureChatOpenAI # Import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import json

class VectorStoreAgent:
    def __init__(
        self,
        pdf_directory: str,
        persist_directory: str,
        azure_openai_api_key: str,
        azure_openai_endpoint: str,
        azure_openai_embedding_deployment: str,
        azure_openai_llm_deployment: str,  # Added LLM deployment
        azure_api_version: str,  # Added API version
        log_file: str,
        documents_list: str,  # Added documents list
    ):
        """
        Initializes the VectorStoreAgent.

        Args:
            pdf_directory (str): Path to the directory containing PDF files.
            persist_directory (str): Directory to persist and load Chroma data.
            azure_openai_api_key (str): API key for Azure OpenAI.
            azure_openai_endpoint (str): Endpoint for Azure OpenAI.
            azure_openai_deployment (str): Deployment name for Azure OpenAI.
            azure_api_version (str): API version for Azure OpenAI.
            log_file (str): Path to the log file.
            documents_list (str): A detailed list of documents with summaries.
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
            deployment=azure_openai_embedding_deployment,
        )
        self.vector_store = self._load_or_create_vector_store()
        
        # Initialize the LLM
        self.llm = AzureChatOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_llm_deployment,
            api_version=azure_api_version,
            temperature=0.1,  # Low temperature for more deterministic outputs
            max_tokens=3000    # Adjust based on expected summary length
        )
        
        # Store the documents list
        self.documents_list = documents_list

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
                doc.metadata['source'] = os.path.basename(pdf_file)  # Store only the filename
            documents.extend(split_docs)
            self.logger.info(f"Loaded and split {len(split_docs)} documents from {pdf_file}.")
        
        self.logger.info(f"Total documents loaded and split: {len(documents)}")
        return documents

    def _determine_sources(self, query_text: str) -> Optional[List[str]]:
        """
        Uses the LLM to determine which document sources to query based on the user query.

        Args:
            query_text (str): The user's query.

        Returns:
            Optional[List[str]]: A list of source filenames to filter the search, or None to search all.
        """
        self.logger.info(f"Determining relevant sources for the query: '{query_text}'")
        
        # Define the prompt
        prompt = self._create_prompt(query_text)
        
        # Get the response from the LLM
        try:
            response = self.llm(prompt)
            self.logger.info(f"LLM response: {response}")
        except Exception as e:
            self.logger.error(f"LLM query failed: {e}")
            return None  # Fallback to searching all documents
        
        # Parse the response to extract document names
        # Assume the LLM returns a JSON list of filenames or a directive to search all
        # Example responses:
        # ["Local New Yorker's Secret List_by Cherie Chen.pdf"]
        # "ALL"
        response = response.content.strip()
        if response.upper() == "ALL":
            return None
        try:
            # Attempt to parse as a list
            sources = json.loads(response)
            if isinstance(sources, list):
                # Map filenames to full path strings
                sources_with_path = [f".././data/Restaurants_data\\{source}" for source in sources]
                self.logger.info(f"Filtered sources based on LLM: {sources_with_path}")
                return sources_with_path
            else:
                self.logger.warning("LLM response was not a list. Searching all documents.")
                return None
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse LLM response as JSON. Searching all documents.")
            return
            #return None

    def _create_prompt(self, query_text: str) -> str:
        """
        Creates the prompt for the LLM based on the user query and documents list.

        Args:
            query_text (str): The user's query.

        Returns:
            str: The formatted prompt.
        """
        prompt_template = f"""
You are an intelligent assistant that helps determine the most relevant documents to answer user queries.

Here is a list of available documents with their summaries:

{self.documents_list}

When a user asks a question, decide which document(s) the answer can be found in. If the question pertains specifically to one document, return a list containing only that document's filename. If it pertains to multiple documents, include all relevant filenames in the list. If the query is general and not specific to any document, reply with "ALL" to indicate that all documents should be searched. 

IMPORTANT! JUST RETURN "ALL" if there is NO specific NAME of documents mentioned in the query.

User Query: "{query_text}"

"ALL" or Respond with a list of filenames. For example: "ALL" or ["Document1.pdf", "Document2.pdf"]. 
"""
        return prompt_template

    def query(self, query_text: str, top_k: int = 5) -> List:
        """
        Queries the vector store with the provided text, filtering based on LLM's instruction.

        Args:
            query_text (str): The query string.
            top_k (int): Number of top results to return.

        Returns:
            List: A list of the top_k matching documents.
        """
        self.logger.info(f"Received query: '{query_text}'")
        
        # Determine which sources to query using the LLM
        selected_sources = self._determine_sources(query_text)
        
        # Set the filter based on selected_sources
        if selected_sources:
            filter_param = {"source": {"$in": selected_sources}}
            self.logger.info(f"Applying filter: {filter_param}")
        else:
            filter_param = None
            self.logger.info("No filter applied. Querying all documents.")
        
        self.logger.info(f"Querying vector store for: '{query_text}' with top_k={top_k}")
        try:
            results = self.vector_store.similarity_search(
                query_text,
                k=top_k,
                filter=filter_param
            )

        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []
        
        self.logger.info(f"Similarity search completed. Found {len(results)} results.")
        for idx, doc in enumerate(results, 1):
            self.logger.info(f"Result {idx}: {doc.metadata['source']}")
            self.logger.info(f"Page Content: {doc.page_content}")
            self.logger.info(f"Page Content Length: {len(doc.page_content)}")
            self.logger.info("-" * 40)
       
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
        results = agent.query(query, top_k=5)
        if not results:
            print("No results found.")
            continue
        print(f"Top {len(results)} results:")
        for idx, doc in enumerate(results, 1):
            print(f"{idx}. Source: {doc.metadata['source']}")
            print(f"   Content Snippet: {doc.page_content[:200]}...")
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
    os.environ["AZURE_OPENAI_4O"] = os.getenv("OPENAI_MODEL_4o")
    # Configuration - Replace these with your actual paths and Azure OpenAI credentials
    PDF_DIRECTORY = ".././data/Restaurants_data"  # e.g., "./pdfs"
    PERSIST_DIRECTORY = ".././data/vectorDB/chroma"
    AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"] 
    AZURE_OPENAI_DEPLOYMENT_SMALL_EMBEDDING = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
    AZURE_OPENAI_4O =os.environ["AZURE_OPENAI_4O"]
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")  # Ensure this is set
    # List of documents with summaries
    DOCUMENTS_LIST = """1. Top 10 Must-Reserve Restaurants in New York City.pdf: This document provides a curated list of ten highly-rated New York City restaurants, categorized by cuisine type, and notes that reservations are required. It's useful for users seeking recommendations for upscale or specific types of dining in NYC.

2. Restaurant_Guide_to_NYC.pdf: This is a personal review of numerous New York City restaurants across various cuisines and price points. Each review includes a letter grade (A+ to D), neighborhood, date of visit, and comments on food quality, service, and value. The guide focuses on mid-priced establishments. It's best for users seeking detailed, subjective opinions on a wide range of NYC restaurants.

3. New York Gui kitchen Diary.pdf: This is a personal food blog entry detailing a visit to Gui Kitchen, a Guangxi cuisine restaurant in NYC's Lower East Side. The author provides personal ratings and descriptions of several dishes, along with comments on the restaurant's ambiance and service. This is useful for users specifically interested in Gui Kitchen or Guangxi cuisine.

4. My First Time at Gui Kitchen.pdf: This document is a more detailed review of Gui Kitchen, focusing on the restaurant's specialization in Guangxi cuisine and highlighting several signature dishes with descriptions of their preparation and flavors. It's suitable for users researching Gui Kitchen or seeking information about Guangxi cuisine.

5. Gui Kitchen reviews.pdf: This document compiles various online reviews of Gui Kitchen restaurants, including ratings and comments on food quality, service, price, and ambiance. Reviews cover multiple locations. It's useful for users wanting a broader perspective on customer experiences at Gui Kitchen.

6. Local New Yorker's Secret List_by Cherie Chen.pdf: This document offers a local's perspective on six restaurants and shops in New York City's Lower East Side. It includes descriptions of the ambiance, food/products, and personal recommendations. It's helpful for users looking for trendy or unique dining and shopping experiences in the Lower East Side.

7. Friedman's - Chelsea Market Restaurant.pdf: This is a menu for Friedman's restaurant in Chelsea Market, New York City. It lists food and drink options with prices, including breakfast, lunch, and dinner items, as well as specifying gluten-free options. This is useful for users planning to dine at Friedman's or looking for menu information.

8. big_bear_Rice_Noodles.pdf: This document describes Big Bear Rice Noodles, a Liuzhou-style rice noodle restaurant in Flushing, NY. It details the restaurant's history, cooking methods (particularly the separate broth and marinade), ingredients, and ambiance. It's suitable for users seeking information about Big Bear Rice Noodles or Liuzhou cuisine.

9. A Guide to New York City_Bumble and bumble University Edition.pdf: This is a guide to New York City attractions, restaurants, bars, and shops, specifically curated for attendees of Bumble and bumble University. It includes location information, price ranges, and brief descriptions. This document is useful for users planning a trip to New York City and interested in the locations recommended in this specific guide.

10. 2024-National-Meeting-Restaurant-Guide.pdf: This is a restaurant guide for a national meeting held at the New York Hilton Midtown. It lists restaurants within walking distance of the hotel, categorized by cuisine type and price range, including those with private dining spaces and low-noise options. It also includes nearby quick lunch options and coffee shops. This is useful for users attending this specific national meeting or looking for dining options near the New York Hilton Midtown.
"""
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
        azure_openai_embedding_deployment=AZURE_OPENAI_DEPLOYMENT_SMALL_EMBEDDING,
        azure_openai_llm_deployment=AZURE_OPENAI_4O,
        azure_api_version=AZURE_API_VERSION,
        log_file="vector_store_agent_test.log",
        documents_list=DOCUMENTS_LIST
    )

    # Start the test module
    test_module(agent)


if __name__ == "__main__":
    main()