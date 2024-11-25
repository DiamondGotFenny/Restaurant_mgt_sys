import os
import sys
import json
import logging
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from server.vectorDB_Agent.bm25_retriever_agent import BM25RetrieverAgent
from server.logger_config import setup_logger
from langchain_openai import AzureOpenAIEmbeddings
current_dir = os.path.dirname(os.path.abspath(__file__))

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
            prediction (str): The prediction string from the BM25 retriever.
            reference (str): The reference answer string from the Gold Standard dataset.

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

def load_qa_keywords(filepath: str, logger: logging.Logger) -> List[Dict]:
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

def load_gold_standard(gs_filepath: str, logger: logging.Logger) -> Dict[str, List[Dict]]:
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
        # Assuming gs_data is a list of entries with 'question' and 'answers'
        gs_dict = {
            entry['question'].strip().lower(): entry['answers'] for entry in gs_data
        }
        return gs_dict
    except Exception as e:
        logger.error(f"Failed to load Gold Standard dataset: {e}")
        sys.exit(1)

def compare_attributes(prediction_metadata: Dict, reference_metadata: Dict) -> Dict[str, bool]:
    """
    Compare 'source' and 'page' attributes between prediction and reference.

    Args:
        prediction_metadata (Dict): Metadata from the prediction (search result).
        reference_metadata (Dict): Metadata from the reference (Gold Standard dataset).

    Returns:
        Dict[str, bool]: Dictionary indicating if 'source' and 'page' match.
    """
    source_match = prediction_metadata.get('source', '').strip().lower() == reference_metadata.get('source', '').strip().lower()
    page_match = prediction_metadata.get('page') == reference_metadata.get('page')
    return {
        'source_match': source_match,
        'page_match': page_match
    }

def test_module(agent: BM25RetrieverAgent, evaluator: EmbeddingSimilarityEvaluator, qa_pairs: List[Dict], gs_dict: Dict[str, List[Dict]], logger: logging.Logger):
    """
    Test module to evaluate the BM25RetrieverAgent against the QA Keywords and Gold Standard datasets.

    Args:
        agent (BM25RetrieverAgent): The BM25 retriever agent instance.
        evaluator (EmbeddingSimilarityEvaluator): Evaluator instance for similarity scoring.
        qa_pairs (List[Dict]): List of QA pairs from the qa_keywords.json.
        gs_dict (Dict[str, List[Dict]]): Gold Standard dataset mapping questions to answers and metadata.
    """
    logger.info("=== Starting BM25 Retriever Agent Test Module ===\n")

    total_questions = len(qa_pairs)
    passed_questions = []
    failed_questions = []

    for qa in qa_pairs:
        qa_id = qa.get('id', 'Unknown ID')
        question = qa.get('question', '').strip().lower()
        question_keywords = qa.get('question_keywords', [])

        logger.info(f"--- Testing Question ID: {qa_id} ---")
        logger.info(f"Question: {question}")
        logger.info(f"Question Keywords: {question_keywords}")

        if not question_keywords:
            logger.warning(f"No question keywords found for Question ID {qa_id}. Skipping.")
            failed_questions.append(qa_id)
            continue

        # Construct the query as a dictionary with 'entities'
        query = {'entities': question_keywords}

        # Query the BM25RetrieverAgent with the question keywords
        try:
            results = agent.query_documents(query=query, top_k=5)
        except Exception as e:
            logger.error(f"BM25 query failed for Question ID {qa_id}: {e}")
            failed_questions.append(qa_id)
            continue

        if not results:
            logger.warning(f"No results found for Question ID {qa_id}.")
            failed_questions.append(qa_id)
            continue

        # Get Reference Answers from Gold Standard
        reference_answers = gs_dict.get(question, [])
        if not reference_answers:
            logger.warning(f"No reference answers found in Gold Standard for Question ID {qa_id}.")
            failed_questions.append(qa_id)
            continue

        # Initialize match flag for this question
        match_found = False

        # Iterate over each expected answer
        for expected in reference_answers:
            reference_answer = expected.get('answer', '').strip()
            reference_source = expected.get('source', '').strip()
            reference_page = expected.get('page', '')

            prediction_id = 0
            # Iterate over search results to find a match
            for doc in results:
                prediction = doc.page_content.strip()
                prediction_metadata = doc.metadata
                prediction_id += 1

                # Compute similarity using embeddings
                similarity = evaluator.compute_similarity(prediction, reference_answer, logger)

                # Compare attributes
                attribute_matches = compare_attributes(prediction_metadata, {
                    'source': reference_source,
                    'page': reference_page
                })

                logger.info(f"Comparing with BM25 search output ID {prediction_id}:")
                logger.info(f"BM25 search output content: {prediction}")
                logger.info(f"BM25 search output Source: {prediction_metadata.get('source', '')}")
                logger.info(f"BM25 search output Page: {prediction_metadata.get('page', '')}")
                logger.info(f"---------------------------------------\n")
                logger.info(f"Reference from golden standard dataset: {reference_answer}")
                logger.info(f"Reference Source: {reference_source}")
                logger.info(f"Reference Page: {reference_page}")
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
                break  # No need to check other reference answers if one has already passed

        if match_found:
            passed_questions.append(qa_id)
            logger.info(f"Question ID {qa_id} Status: PASS\n")
        else:
            failed_questions.append(qa_id)
            logger.info(f"Question ID {qa_id} Status: FAIL\n")

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

    logger.info("=== End of BM25 Retriever Agent Test Module ===")

def main(logger: logging.Logger, test_log_filepath: str):
    """
    Main function to initialize the BM25RetrieverAgent and start the test module.
    """
    # Load environment variables
    load_dotenv(find_dotenv())

    # Retrieve environment variables
    AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

    # Validate environment variables
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_MODEL, AZURE_API_VERSION]):
        logger.error("One or more Azure OpenAI environment variables are missing.")
        sys.exit(1)

    # Configuration
    PDF_DIRECTORY=os.path.join(current_dir,"..","data","Restaurants_data") 
    LOG_FILE=os.path.join(current_dir,"..","logs","bm25_retriever_agent.log")
    WHOOSH_INDEX_DIR=os.path.join(current_dir,"..","data","whoosh_index")
    QA_KEYWORDS_FILEPATH=os.path.join(current_dir,"qa_keywords.json")
    GOLDEN_STANDARD_FILEPATH=os.path.join(current_dir,"golden_standard_Raw_Chunks.json")

    # Validate directories and files
    if not os.path.isdir(PDF_DIRECTORY):
        logger.error(f"PDF directory does not exist: {PDF_DIRECTORY}")
        sys.exit(1)
    if not os.path.exists(WHOOSH_INDEX_DIR):
        try:
            os.makedirs(WHOOSH_INDEX_DIR, exist_ok=True)
            logger.info(f"Created directory for Whoosh index: {WHOOSH_INDEX_DIR}")
        except Exception as e:
            logger.error(f"Failed to create directory {WHOOSH_INDEX_DIR}: {e}")
            sys.exit(1)
    if not os.path.isfile(QA_KEYWORDS_FILEPATH):
        logger.error(f"QA Keywords file does not exist: {QA_KEYWORDS_FILEPATH}")
        sys.exit(1)
    if not os.path.isfile(GOLDEN_STANDARD_FILEPATH):
        logger.error(f"Gold Standard file does not exist: {GOLDEN_STANDARD_FILEPATH}")
        sys.exit(1)

    # Load QA Keywords and Gold Standard datasets
    qa_pairs = load_qa_keywords(QA_KEYWORDS_FILEPATH, logger)
    logger.info(f"Loaded QA Keywords dataset with {len(qa_pairs)} questions.\n")

    gs_dict = load_gold_standard(GOLDEN_STANDARD_FILEPATH, logger)
    logger.info(f"Loaded Gold Standard dataset with {len(gs_dict)} questions.\n")

    # Initialize the BM25RetrieverAgent
    agent = BM25RetrieverAgent(
        pdf_directory=PDF_DIRECTORY,
        log_file=LOG_FILE,
        chunk_size=2000,
        chunk_overlap=200,
        bm25_params={"k1": 0.5, "b": 0.75},
        whoosh_index_dir=WHOOSH_INDEX_DIR
    )

    # Initialize the EmbeddingSimilarityEvaluator
    evaluator = EmbeddingSimilarityEvaluator(
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_embedding_deployment=AZURE_OPENAI_EMBEDDING_MODEL
    )

    # Start the test module
    test_module(agent, evaluator, qa_pairs, gs_dict, logger)

if __name__ == "__main__":
    test_log_filepath = os.path.join(current_dir,"..", "logs", "bm25_retriever_agent_test.log")
    logger = setup_logger(test_log_filepath)
    main(logger, test_log_filepath)