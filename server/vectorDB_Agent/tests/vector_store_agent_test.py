
import sys
from dotenv import load_dotenv, find_dotenv
import os
import json
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import AzureOpenAIEmbeddings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store_agent import VectorStoreAgent

class VectorSearchEvaluator:
    def __init__(self, azure_openai_api_key: str, azure_openai_endpoint: str, azure_openai_embedding_deployment: str):
        """
        Initialize the evaluator with Azure OpenAI Embeddings.
        
        Args:
            azure_openai_api_key (str): Azure OpenAI API key.
            azure_openai_endpoint (str): Azure OpenAI endpoint.
            azure_openai_embedding_deployment (str): Azure OpenAI embedding deployment name.
        """
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment=azure_openai_embedding_deployment,
        )
    
    def compute_similarity(self, prediction: str, reference: str) -> float:
        """
        Compute the cosine similarity between prediction and reference texts.
        
        Args:
            prediction (str): The prediction string from the vector search.
            reference (str): The reference answer string from the GS dataset.
        
        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        if not prediction or not reference:
            return 0.0
        try:
            # Generate embeddings using Azure OpenAI Embeddings
            pred_embedding = self.embeddings.embed_documents([prediction])[0]
            ref_embedding = self.embeddings.embed_documents([reference])[0]
            # Compute cosine similarity
            similarity = cosine_similarity([pred_embedding], [ref_embedding])[0][0]
            return similarity
        except Exception as e:
            # Log the error and return 0 similarity
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def compare_attributes(self, prediction_metadata: Dict, reference_metadata: Dict) -> Dict[str, bool]:
        """
        Compare 'source' and 'page' attributes between prediction and reference.
        
        Args:
            prediction_metadata (Dict): Metadata from the prediction (search result).
            reference_metadata (Dict): Metadata from the reference (GS dataset).
        
        Returns:
            Dict[str, bool]: Dictionary indicating if 'source' and 'page' match.
        """
        source_match = prediction_metadata.get('source', '').strip().lower() == reference_metadata.get('source', '').strip().lower()
        page_match = prediction_metadata.get('page') == reference_metadata.get('page')
        return {
            'source_match': source_match,
            'page_match': page_match
        }

def load_gold_standard(gs_filepath: str) -> Dict[str, List[Dict]]:
    """
    Load the Gold Standard dataset from a JSON file.
    
    Args:
        gs_filepath (str): Path to the GS JSON file.
    
    Returns:
        dict: A dictionary mapping questions to a list of their answers and metadata.
    """
    try:
        with open(gs_filepath, 'r', encoding='utf-8') as file:
            gs_data = json.load(file)
        gs_dict = {
            entry['question'].strip().lower(): entry['answers'] for entry in gs_data
        }
        return gs_dict
    except Exception as e:
        print(f"Failed to load Gold Standard dataset: {e}")
        sys.exit(1)

def test_module(agent: VectorStoreAgent, gs_dict: Dict[str, List[Dict]], evaluator: VectorSearchEvaluator):
    """
    A test method to interactively query the vector store via the terminal and evaluate results.
    
    Args:
        agent (VectorStoreAgent): The vector store agent instance.
        gs_dict (Dict[str, List[Dict]]): Gold Standard dataset mapping questions to answers and metadata.
        evaluator (VectorSearchEvaluator): The evaluator instance for similarity scoring.
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
        
        # Perform the vector search
        results = agent.query(query)
        if not results:
            print("No results found.")
            continue
        
        print(f"Top {len(results)} results:")
        for idx, doc in enumerate(results, 1):
            # Log the metadata and content
            if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                agent.logger.info(f"Page: {doc.metadata['page']}")
            agent.logger.info(f"Source: {doc.metadata.get('source', 'Unknown Source')}")
            agent.logger.info(f"Content: {doc.page_content}")
            agent.logger.info("-" * 40)
        
        # Evaluation against Gold Standard
        # Assuming the user's query directly maps to a question in GS
        query_key = query.strip().lower()
        if query_key in gs_dict:
            expected_answers = gs_dict[query_key]  # List of expected answers
            agent.logger.info("=== Evaluation ===")
            agent.logger.info(f"Input Query: {query}")
            
            # Iterate over each search result
            match_count = 0
            for idx, doc in enumerate(results, 1):
                prediction = doc.page_content
                prediction_metadata = doc.metadata
                # Compare with all expected answers
                similarities = []
                attribute_matches = []
                for expected in expected_answers:
                    similarity = evaluator.compute_similarity(prediction, expected['answer'])
                    comparisons = evaluator.compare_attributes(doc.metadata, {
                        'source': expected['source'],
                        'page': expected['page']
                    })
                    similarities.append(similarity)
                    attribute_matches.append(comparisons)
                
                # Determine the best match
                if similarities:
                    max_similarity = max(similarities)
                    best_match_index = similarities.index(max_similarity)
                    best_comparison = attribute_matches[best_match_index]
                    
                    agent.logger.info(f"Result {idx} Similarity Score: {max_similarity:.4f}")
                    agent.logger.info(f"Result {idx} Source Match: {best_comparison['source_match']}")
                    agent.logger.info(f"Result {idx} Page Match: {best_comparison['page_match']}")
                    
                    # Determine pass/fail based on similarity and attribute matches
                    similarity_threshold = 0.5  # Example threshold
                    if (max_similarity >= similarity_threshold and 
                        best_comparison['source_match'] and 
                        best_comparison['page_match']):
                        pass_fail = "PASS"
                        match_count += 1
                    else:
                        pass_fail = "FAIL"
                    agent.logger.info(f"Result {idx} Evaluation: {pass_fail}")
                    agent.logger.info("-" * 20)
            
            agent.logger.info(f"Total Matches: {match_count} out of {len(expected_answers)}")
            agent.logger.info("=" * 40)
            
            print(f"Total Matches: {match_count} out of {len(expected_answers)}")
        else:
            print("No Gold Standard answer found for this query.")
            agent.logger.info("No Gold Standard answer found for this query.")

def main():
    """
    Main function to initialize the agent, load GS dataset, initialize evaluator, and start the test module.
    """
    # Load environment variables
    load_dotenv(find_dotenv())
    
    # Set Azure OpenAI environment variables
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("OPENAI_API_BASE")
    os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_API_VERSION")
    os.environ["AZURE_OPENAI_EMBEDDING_MODEL"] = os.getenv("OPENAI_EMBEDDING_MODEL")
    
    # Configuration - Replace these with your actual paths and Azure OpenAI credentials
    PDF_DIRECTORY = "../.././data/Restaurants_data"  # e.g., "./pdfs"
    PERSIST_DIRECTORY = "../.././data/vectorDB/chroma"
    GS_FILEPATH = "golden_standard_Raw_Chunks.json"  # Path to your GS JSON file
    
    AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT") 
    AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL")
    
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
    if not os.path.isfile(GS_FILEPATH):
        print(f"Gold Standard file does not exist: {GS_FILEPATH}")
        sys.exit(1)
    
    # Load the Gold Standard dataset
    gs_dict = load_gold_standard(GS_FILEPATH)
    print(f"Loaded Gold Standard dataset with {len(gs_dict)} questions.")
    
    # Initialize the VectorStoreAgent
    agent = VectorStoreAgent(
        pdf_directory=PDF_DIRECTORY,
        persist_directory=PERSIST_DIRECTORY,
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_embedding_deployment=AZURE_OPENAI_DEPLOYMENT,
        log_file="vector_store_agent_test.log"
    )
    
    # Initialize the Evaluator
    evaluator = VectorSearchEvaluator(
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_embedding_deployment=AZURE_OPENAI_DEPLOYMENT
    )
    
    # Start the test module
    test_module(agent, gs_dict, evaluator)

if __name__ == "__main__":
    main()