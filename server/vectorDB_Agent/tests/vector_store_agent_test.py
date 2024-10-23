import sys
from dotenv import load_dotenv, find_dotenv
import os
import json
import logging
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import AzureOpenAIEmbeddings

# Add parent directory to path to import VectorStoreAgent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_store_agent import VectorStoreAgent
from logger_config import setup_logger

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
                model=azure_openai_embedding_deployment,
                api_key=azure_openai_api_key,
                azure_endpoint=azure_openai_endpoint,
                deployment=azure_openai_embedding_deployment,
            )

    def compute_similarity(self, prediction: str, reference: str,logger:logging.Logger) -> float:
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
            logger.error(f"Error computing similarity: {e}")
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

def load_gold_standard(gs_filepath: str,logger:logging.Logger) -> Dict[str, List[Dict]]:
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
        # Assuming gs_data is a list of qa_pairs
        gs_dict = {
            entry['question'].strip().lower(): entry['answers'] for entry in gs_data
        }
        return gs_dict
    except Exception as e:
        logger.error(f"Failed to load Gold Standard dataset: {e}")
        sys.exit(1)


def test_module(agent: VectorStoreAgent, gs_dict: Dict[str, List[Dict]], evaluator: VectorSearchEvaluator,logger:logging.Logger):
    """
    Automated test method to iterate through all questions in the Gold Standard dataset,
    perform vector searches, compare results with expected answers, and log the outcomes.

    Args:
        agent (VectorStoreAgent): The vector store agent instance.
        gs_dict (Dict[str, List[Dict]]): Gold Standard dataset mapping questions to answers and metadata.
        evaluator (VectorSearchEvaluator): The evaluator instance for similarity scoring.
    """
    logger.info("=== Starting Vector Store Agent Test Module ===\n")

    total_questions = len(gs_dict)
    passed_questions = []
    failed_questions = []

    for idx, (question, answers) in enumerate(gs_dict.items(), 1):
        q_id = idx
        logger.info(f"--- Testing Question {idx}/{total_questions} (ID: {q_id}) ---")

        # Perform the vector search
        try:
            results = agent.query(question)
        except Exception as e:
            logger.error(f"Vector search failed for Question ID {q_id}: {e}")
            failed_questions.append(q_id)
            continue

        if not results:
            logger.warning(f"No results found for Question ID {q_id}.")
            failed_questions.append(q_id)
            continue

        # Initialize match flag for this question
        match_found = False

        # Iterate over each expected answer
        for expected in answers:
            reference_answer = expected.get('answer', '').strip()
            reference_source = expected.get('source', '').strip()
            reference_page = expected.get('page', '')

            prediction_id=0
            # Iterate over search results to find a match
            for doc in results:
                prediction = doc.page_content.strip()
                prediction_metadata = doc.metadata
                prediction_id += 1
                # Compute similarity
                similarity = evaluator.compute_similarity(prediction, reference_answer,logger)
                # Compare attributes
                attribute_matches = evaluator.compare_attributes(prediction_metadata, {
                    'source': reference_source,
                    'page': reference_page
                })

                logger.info(f"Comparing with Vector search output ID {prediction_id}:")
                logger.info(f"Vector search output content: {prediction}")
                logger.info(f"Vector search output Source: {prediction_metadata.get('source', '')}")
                logger.info(f"Vector search output Page: {prediction_metadata.get('page', '')}")
                logger.info(f"---------------------------------------\n")
                logger.info(f"Reference from golden standard dataset: {reference_answer}")
                logger.info(f"Reference from golden standard dataset Source: {reference_source}")
                logger.info(f"Reference from golden standard dataset Page: {reference_page}")
                logger.info(f"---------------------------------------\n")
                logger.info(f"Similarity Score: {similarity:.4f}")
                logger.info(f"Source Match: {attribute_matches['source_match']}")
                logger.info(f"Page Match: {attribute_matches['page_match']}")
               

                # Define thresholds
                similarity_threshold = 0.7  # Adjust as needed

                # Determine pass/fail for this answer
                if (similarity >= similarity_threshold and
                    attribute_matches['source_match'] and
                    attribute_matches['page_match']):
                    logger.info("Evaluation Result: PASS\n")
                    match_found = True
                    break  # Stop searching if a match is found

                else:
                    logger.info("Evaluation Result: FAIL\n")
                logger.info(f"---------Answer {prediction_id} test end----------------\n")

            if match_found:
                break  # No need to check other answers if one has already passed

        if match_found:
            passed_questions.append(q_id)
            logger.info(f"Question ID {q_id} Status: PASS\n")
        else:
            failed_questions.append(q_id)
            logger.info(f"Question ID {q_id} Status: FAIL\n")

    # Summary of results
    logger.info("=== Test Summary ===")
    logger.info(f"Total Questions Tested: {total_questions}")
    logger.info(f"Total Passes: {len(passed_questions)}")
    logger.info(f"Total Failures: {len(failed_questions)}")
    success_rate = (len(passed_questions) / total_questions) * 100 if total_questions else 0
    logger.info(f"Overall Success Rate: {success_rate:.2f}%\n")

    # Log Passed Questions
    if passed_questions:
        logger.info(f"Passed Questions (Total: {len(passed_questions)}):")
        for pid in passed_questions:
            logger.info(f" - Question ID: {pid}")
    else:
        logger.info("No questions passed the threshold.")

    # Log Failed Questions
    if failed_questions:
        logger.info(f"\nFailed Questions (Total: {len(failed_questions)}):")
        for fid in failed_questions:
            logger.info(f" - Question ID: {fid}")
    else:
        logger.info("No questions failed the threshold.")

    logger.info("=== End of Vector Store Agent Test Module ===")

def main(logger:logging.Logger,test_log_filepath:str):
    """
    Main function to initialize the agent, load the Gold Standard dataset, initialize the evaluator,
    and start the automated test module.
    """
    # Load environment variables
    load_dotenv(find_dotenv())


    # Configuration - Replace these with your actual paths and Azure OpenAI credentials
    PDF_DIRECTORY = "../.././data/Restaurants_data"  # e.g., "./pdfs"
    PERSIST_DIRECTORY = "../.././data/vectorDB/chroma"
    GS_FILEPATH = "golden_standard_Raw_Chunks.json"  # Path to your GS JSON file

    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("OPENAI_API_BASE")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("OPENAI_EMBEDDING_MODEL")

    # Validate directories
    if not os.path.isdir(PDF_DIRECTORY):
        print(f"PDF directory does not exist: {PDF_DIRECTORY}")
        sys.exit(1)
    if not os.path.exists(PERSIST_DIRECTORY):
        try:
            os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
            print(f"Created directory for vector store: {PERSIST_DIRECTORY}")
        except Exception as e:
            print(f"Failed to create directory {PERSIST_DIRECTORY}: {e}")
            sys.exit(1)
    if not os.path.isfile(GS_FILEPATH):
        print(f"Gold Standard file does not exist: {GS_FILEPATH}")
        sys.exit(1)

    # Load the Gold Standard dataset
    gs_dict = load_gold_standard(GS_FILEPATH,logger)
    logging.info(f"Loaded Gold Standard dataset with {len(gs_dict)} questions.\n")

    # Initialize the VectorStoreAgent
    agent = VectorStoreAgent(
        pdf_directory=PDF_DIRECTORY,
        persist_directory=PERSIST_DIRECTORY,
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_embedding_deployment=AZURE_OPENAI_DEPLOYMENT,
        log_file=test_log_filepath
    )

    # Initialize the Evaluator
    evaluator = VectorSearchEvaluator(
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_embedding_deployment=AZURE_OPENAI_DEPLOYMENT
    )

    # Start the automated test module
    test_module(agent, gs_dict, evaluator,logger)

if __name__ == "__main__":
    test_log_filepath = "vector_store_agent_test.log"
    logger=setup_logger(test_log_filepath)
    main(logger,test_log_filepath)