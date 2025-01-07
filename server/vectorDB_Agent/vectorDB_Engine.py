from document_processor import DocumentProcessor
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_config import setup_logger
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))



class VectorDBEngine:
    def __init__(self, log_path):
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.chain = None
        self.llm = None
        self.embeddings = None
        self.documents = None
        self.processor = None
        self.logger = setup_logger(log_path)
        self.initialize_engine()

    def initialize_engine(self):
        """Initializes the vector database engine."""
         # Set the PDF directory using relative path
        pdf_directory = os.path.join(current_dir, '..', 'data', 'Restaurants_data')
        self.processor = DocumentProcessor(pdf_directory)

        # Load and split documents
        try:
            self.documents = self.processor.load_and_split_documents()
        except Exception as e:
            self.logger.error(f"Failed to load and split documents: {e}")
            self.documents = []

        if not self.documents:
            self.logger.error("No documents available for processing.")
            return
        
        # Initialize embeddings and llm
        self._initialize_embeddings_llm()

        # Set up Chroma vector store
        self._setup_vector_store()

        # Set up BM25Retriever
        self._setup_bm25_retriever()

        # Combine retrievers for hybrid search
        self._setup_ensemble_retriever()

        # Create RAG chain
        self._create_rag_chain()

    def _initialize_embeddings_llm(self):
        """Initializes Azure OpenAI embeddings and language model."""
        AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_OPENAI_EMBEDDING = os.getenv("OPENAI_EMBEDDING_MODEL")
        AZURE_OPENAI_4OMINI = os.getenv("OPENAI_MODEL_4OMINI")
        AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
        
        #check if the environment variables are set, and show which ones are missing
        if not AZURE_OPENAI_API_KEY:
            self.logger.error("Azure OpenAI API key is not set.")
            return
        if not AZURE_OPENAI_ENDPOINT:
            self.logger.error("Azure OpenAI endpoint is not set.")
            return
        if not AZURE_OPENAI_EMBEDDING:
            self.logger.error("Azure OpenAI embedding model is not set.")
            return
        if not AZURE_OPENAI_4OMINI:
            self.logger.error("Azure OpenAI model 4OMINI is not set.")
            return
        if not AZURE_API_VERSION:
            self.logger.error("Azure API version is not set.")
            return

        self.embeddings = AzureOpenAIEmbeddings(
            model=AZURE_OPENAI_EMBEDDING,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            deployment=AZURE_OPENAI_EMBEDDING,
        )
        self.llm = AzureChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name=AZURE_OPENAI_4OMINI,
            api_version=AZURE_API_VERSION,
            temperature=0,
            max_tokens=3000
        )

    def _setup_vector_store(self):
        """Sets up the Chroma vector store."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        persist_directory = os.path.join(current_dir, '..', 'data', 'vectorDB', 'chroma')

        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            self.logger.info("Vector store not found. Creating a new one...")
            self.vector_store = Chroma.from_documents(
                self.documents,
                self.embeddings,
                persist_directory=persist_directory
            )
            self.logger.info("Chroma vector store initialized.")
        else:
            try:
                self.vector_store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory
                )
                self.logger.info("Chroma vector store loaded from existing directory.")
            except Exception as e:
                self.logger.error(f"Failed to load Chroma vector store: {e}")
                self.vector_store = None

    def _setup_bm25_retriever(self):
        """Sets up the BM25 retriever."""
        try:
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = 5
            self.logger.info("BM25 retriever initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize BM25 retriever: {e}")
            self.bm25_retriever = None

    def _setup_ensemble_retriever(self):
        """Sets up the ensemble retriever."""
        if self.vector_store is None or self.bm25_retriever is None:
            self.logger.error("Vector store or BM25 retriever is not initialized.")
            self.ensemble_retriever = None
            return
        
        try:
            vector_store_retriever = self.vector_store.as_retriever()
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_store_retriever, self.bm25_retriever],
                weights=[0.5, 0.5]
            )
            self.logger.info("Ensemble retriever initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize ensemble retriever: {e}")
            self.ensemble_retriever = None

    def _create_rag_chain(self):
        """Creates the RAG chain."""
        if self.ensemble_retriever is None:
            self.logger.error("Ensemble retriever is not initialized.")
            self.chain = None
            return
        
        try:
            template = """You are a Data Processing Assistant. Your task is to extract and organize all relevant information from the provided documents based on the user's query. The extracted information will be passed to the next data processor, so it is crucial to preserve all details accurately. Lossless information retention is essential at this stage. Adhere to the following guidelines:

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
{question}

**Retrieved Documents:**
            {context}
            
**Relevant Information:**
            """
            prompt = ChatPromptTemplate.from_template(template)

            self.chain = (
                {"context": self.ensemble_retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            self.logger.info("RAG chain created.")
        except Exception as e:
            self.logger.error(f"Failed to create RAG chain: {e}")
            self.chain = None

    def run_query(self, query):
        """Runs a query against the RAG chain."""
        if self.chain:
            try:
                result = self.chain.invoke(query)
                self.logger.info("Result:----------------")
                self.logger.info(result)
                self.logger.info(f"--------------------\n")
                return result
            except Exception as e:
                self.logger.error(f"Failed to process query: {e}")
                print("An error occurred while processing your query.")
                return None
        else:
            print("Chain not initialized. Unable to process query.")
            return None

def main():
    # Set up logging
    path=os.path.join(current_dir,'..', 'tests',"vectorDB_Engine_test.log")
    engine = VectorDBEngine(path)

    while True:
        query = input("Enter your query (or 'q' to quit): ")
        if query.lower() == 'q':
            break
        result=engine.run_query(query)
        print(f"Result: {result}")

if __name__ == "__main__":
    main()