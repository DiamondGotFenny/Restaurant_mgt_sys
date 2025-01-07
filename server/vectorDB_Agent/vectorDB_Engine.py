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

# Set up logging
logger = setup_logger("vector_store_agent.log")

# Define the main function
def main():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the PDF directory using relative path
    pdf_directory = os.path.join(current_dir, '..', 'data', 'Restaurants_data')
    
    # Initialize DocumentProcessor
    processor = DocumentProcessor(pdf_directory)
    
    # Load and split documents
    try:
        documents = processor.load_and_split_documents()
    except Exception as e:
        logger.error(f"Failed to load and split documents: {e}")
        documents = []
    
    if not documents:
        logger.error("No documents available for processing.")
        return
    
    # Initialize embeddings
    AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_EMBEDDING = os.getenv("OPENAI_EMBEDDING_MODEL")
    AZURE_OPENAI_4OMINI = os.getenv("OPENAI_MODEL_4OMINI")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
    embeddings = AzureOpenAIEmbeddings(
        model=AZURE_OPENAI_EMBEDDING,
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_API_VERSION,
                deployment=AZURE_OPENAI_EMBEDDING,
    )
    llm = AzureChatOpenAI(
         api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name=AZURE_OPENAI_4OMINI,
            api_version=AZURE_API_VERSION,
            temperature=0,
            max_tokens=3000
    )
    
    # Set up Chroma vector store with relative path
    persist_directory = os.path.join(current_dir, '..', 'data', 'vectorDB', 'chroma')
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
    else:
        try:
            logger.info("Vector store not found. Creating a new one...")
            vector_store = Chroma.from_documents(
                documents,
                embeddings,
                persist_directory=persist_directory
            )
            vector_store.persist()
            logger.info("Chroma vector store initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma vector store: {e}")
            vector_store = None
    
    # Set up BM25Retriever
    try:
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5  # Retrieve top 5 results
        logger.info("BM25 retriever initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize BM25 retriever: {e}")
        bm25_retriever = None
    
    # Combine retrievers for hybrid search
    try:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_store, bm25_retriever],
            weights=[0.5, 0.5]  # Adjust weights as needed
        )
        logger.info("Ensemble retriever initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize ensemble retriever: {e}")
        ensemble_retriever = None
    
    # Create RAG chain
    try:
        # Define prompt template
        template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        
        # Create RAG chain
        chain = (
            {"context": ensemble_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("RAG chain created.")
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {e}")
        chain = None
    
    # Input loop for user queries
    while True:
        query = input("Enter your query (or 'q' to quit): ")
        if query.lower() == 'q':
            break
        if chain:
            try:
                result = chain.invoke(query)
                print("Result:")
                print(result)
            except Exception as e:
                logger.error(f"Failed to process query: {e}")
                print("An error occurred while processing your query.")
        else:
            print("Chain not initialized. Unable to process query.")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()