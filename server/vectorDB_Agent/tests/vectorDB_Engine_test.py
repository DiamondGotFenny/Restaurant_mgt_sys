import os
import sys
import json
import logging
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorDB_Engine import VectorDBEngine
from langchain_openai import AzureOpenAIEmbeddings
from logger_config import setup_logger

def load_qa_pairs(filepath: str, logger: logging.Logger) -> List[Dict]:
    """
    Load the QA pairs from a JSON file.

    Args:
        filepath (str): Path to the 'qa_keywords.json' file.

    Returns:
        List[Dict]: List of QA pairs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            qa_data = json.load(file)['qa_pairs']
        return qa_data
    except Exception as e:
        logger.error(f"Failed to load QA pairs dataset: {e}")
        sys.exit(1)

class EmbeddingSimilarityEvaluator:
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

    def compute_similarity(self, prediction: str, reference: str, logger: logging.Logger) -> float:
        """
        Compute the cosine similarity between prediction and reference texts using embeddings.

        Args:
            prediction (str): The predicted answer from the agent.
            reference (str): The reference answer string from the QA dataset.

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

def test_vectorDB_engine(logger: logging.Logger):
    """
    Test function to evaluate the vectorDB_Engine using QA pairs and compute similarity with expected answers.
    """
    # Load environment variables
    load_dotenv(find_dotenv())

    # Retrieve environment variables
    AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT =os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

    # Validate environment variables
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_MODEL, AZURE_API_VERSION]):
        logger.error("One or more Azure OpenAI environment variables are missing.")
        sys.exit(1)

    engine=VectorDBEngine()
    
    # Initialize EmbeddingSimilarityEvaluator
    evaluator = EmbeddingSimilarityEvaluator(
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_embedding_deployment=AZURE_OPENAI_EMBEDDING_MODEL
    )

    # Load QA pairs
    qa_filepath = "qa_keywords.json"  # Path to your 'qa_keywords.json'
    qa_pairs = load_qa_pairs(qa_filepath, logger)
    logger.info(f"Loaded {len(qa_pairs)} QA pairs for testing.")

    total_questions = len(qa_pairs)
    passed_questions = []
    failed_questions = []

    # Store unique_documents for output as JSON
    all_unique_documents = {}

    logger.info("=== Starting vectorDB_Engine Test Module ===\n")

    for qa in qa_pairs:
        qa_id = qa.get('id', 'Unknown ID')
        question = qa.get('question', '').strip()
        expected_answers = list(qa.get('answers', {}).values())

        logger.info(f"--- Testing Question ID: {qa_id} ---")
        logger.info(f"Question: {question}")

        # Perform the query using the qa_chain function from vectorDB_Engine
        try:
            response = engine.qa_chain(question)
        except Exception as e:
            logger.error(f"Error processing query for Question ID {qa_id}: {e}")
            failed_questions.append(qa_id)
            continue

        # Access unique_documents from agent_vector_search
        # Note: Adjust this part based on how unique_documents are stored or returned
        # For this example, we'll assume unique_documents are accessible
        unique_documents = engine.return_unique_documents(question)  # Adjust as needed
        #store all doc in all_unique_documents, not just page content
        all_unique_documents[qa_id] = "\n\n".join([f"Source: {doc.metadata['source']}\nPage: {doc.metadata['page']}\n{doc.page_content}" for doc in unique_documents])

        # Compare the response with expected answers using embeddings
        highest_similarity = 0.0
        for expected_answer in expected_answers:
            similarity = evaluator.compute_similarity(response, expected_answer, logger)
            if similarity > highest_similarity:
                highest_similarity = similarity
            logger.info("compare response with expected answer start-----------------")
            logger.info(f"Expected Answer:\n{expected_answer}")
            logger.info(f"Agent Response:\n{response}")
            logger.info("compare response with expected answer end-----------------")

        logger.info(f"Highest Similarity Score: {highest_similarity:.4f}")

        # Define threshold
        similarity_threshold = 0.7  # Adjust as needed

        # Determine pass/fail
        if highest_similarity >= similarity_threshold:
            logger.info("Evaluation Result: PASS\n")
            passed_questions.append(qa_id)
        else:
            logger.info("Evaluation Result: FAIL\n")
            failed_questions.append(qa_id)
        
        logger.info(f"--- Testing Question ID: {qa_id} End---")

    # Output unique_documents as a JSON file
    with open('unique_documents_output.json', 'w', encoding='utf-8') as f:
        json.dump(all_unique_documents, f, ensure_ascii=False, indent=4)
    logger.info("Unique documents have been saved to 'unique_documents_output.json'.")

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

    logger.info("=== End of vectorDB_Engine Test Module ===")

if __name__ == "__main__":
    test_log_filepath = "vectorDB_Engine_test.log"
    # Initialize logger
    logger = setup_logger(test_log_filepath)
    test_vectorDB_engine(logger)




