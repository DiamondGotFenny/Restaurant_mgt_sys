import os
import sys
import json
import logging
from typing import List, Dict, Tuple
from dotenv import load_dotenv, find_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_pre_processor import LLMQueryPreProcessor 
from logger_config import setup_logger


def load_qa_keywords(filepath: str,logger:logging.Logger) -> List[Dict]:
    """
    Load the QA Keywords from a JSON file.
    
    Args:
        filepath (str): Path to the 'qa_keywords.json' file.
    
    Returns:
        List[Dict]: List of QA pairs with questions and keywords.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            qa_data = json.load(file)['qa_pairs']
        return qa_data
    except Exception as e:
        logger.error(f"Failed to load QA Keywords dataset: {e}")
        sys.exit(1)

def compute_similarity_metrics(extracted: List[str], expected: List[str]) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F1-Score between extracted entities and expected keywords.
    
    Args:
        extracted (List[str]): Entities extracted by the processor.
        expected (List[str]): Expected keywords from the dataset.
    
    Returns:
        Tuple[float, float, float]: Precision, Recall, F1-Score
    """
    extracted_set = set([e.strip().lower() for e in extracted])
    expected_set = set([k.strip().lower() for k in expected])
    
    true_positives = extracted_set.intersection(expected_set)
    precision = len(true_positives) / len(extracted_set) if extracted_set else 0.0
    recall = len(true_positives) / len(expected_set) if expected_set else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score

def test_module(logger:logging.Logger,test_log_filepath:str):
    """
    Test Module to automatically evaluate the LLMQueryPreProcessor against the QA Keywords dataset.
    """
    # Load environment variables
    load_dotenv(find_dotenv())

    # Retrieve environment variables
    AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_4O = os.getenv("OPENAI_MODEL_4o")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
    
      # Validate environment variables
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_4O, AZURE_API_VERSION]):
        logger.error("One or more Azure OpenAI environment variables are missing.")
        sys.exit(1)
        
    # Initialize the LLMQueryPreProcessor
    processor = LLMQueryPreProcessor(
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_deployment=AZURE_OPENAI_4O,
        azure_api_version=AZURE_API_VERSION,
        log_file=test_log_filepath
    )

    # Load QA Keywords dataset
    qa_filepath = "qa_keywords.json"  # Path to your 'qa_keywords.json'
    qa_pairs = load_qa_keywords(qa_filepath,logger)
    logger.info(f"Loaded {len(qa_pairs)} QA pairs for testing.")

    logger.info("=== Starting LLMQueryPreProcessor Test Module ===\n")

    # Define similarity threshold for passing the test
    f1_threshold = 0.7  # Adjust based on desired strictness

    total_questions = len(qa_pairs)
    total_passes = 0
    # Initialize lists to store passed and failed question IDs
    passed_questions = []
    failed_questions = []
    for qa in qa_pairs:
        qa_id = qa.get('id', 'Unknown ID')
        question = qa.get('question', '').strip()
        expected_keywords = qa.get('question_keywords', [])
        
        logger.info(f"--- Testing Question ID: {qa_id} ---")
        logger.info(f"Question: {question}")
        logger.info(f"Expected Keywords: {expected_keywords}")
        
        # Process the query to extract entities
        try:
            result = processor.process_query(question)
            extracted_entities = result.get('entities', [])
            #already logged extracted entities in the process_query function, so here we just need to log the Expected entities
            logger.info(f"Expected Keywords: {expected_keywords}")
        except Exception as e:
            logger.error(f"Error processing query ID {qa_id}: {e}")
            extracted_entities = []
        
        # Compute similarity metrics
        precision, recall, f1_score = compute_similarity_metrics(extracted_entities, expected_keywords)
        logger.info(f"Similarity Metrics - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}")
        
        # Determine pass/fail
        if f1_score >= f1_threshold:
            test_result = "PASS"
            passed_questions.append(qa_id)
            total_passes += 1
        else:
            test_result = "FAIL"
            failed_questions.append(qa_id)
        
        logger.info(f"Test Result: {test_result}\n")

    # Summary of results
    logger.info("=== Test Summary ===")
    logger.info(f"Total Questions Tested: {total_questions}")
    logger.info(f"Total Passes: {total_passes}")
    logger.info(f"Total Failures: {total_questions - total_passes}")
    success_rate = (total_passes / total_questions) * 100 if total_questions else 0
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

    logger.info("=== End of Test Module ===")

if __name__ == "__main__":
    test_log_filepath = "query_pre_processor_test.log"
    logger=setup_logger(test_log_filepath)
    test_module(logger,test_log_filepath)