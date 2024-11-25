# llm_post_processor_test.py

import os
import sys
import json
import logging
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from collections import namedtuple  
from server.vectorDB_Agent.llm_post_processor import LLMProcessor
from server.logger_config import setup_logger
from langchain_openai import AzureOpenAIEmbeddings
current_dir = os.path.dirname(os.path.abspath(__file__))

class SimilarityEvaluator:
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


    def compute_similarity(self, text1: str, text2: str, logger: logging.Logger) -> float:
        """
        Compute the cosine similarity between two texts using embeddings.

        Args:
            text1 (str): First text string.
            text2 (str): Second text string.

        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        if not text1 or not text2:
            return 0.0
        try:
            # Generate embeddings using Azure OpenAI Embeddings
            embedding1 = self.embeddings.embed_documents([text1])[0]
            embedding2 = self.embeddings.embed_documents([text2])[0]
            # Compute cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return similarity
        except Exception as e:
            # Log the error and return 0 similarity
            logger.error(f"Error computing similarity: {e}")
            return 0.0

def load_unique_documents(filepath: str, logger: logging.Logger) -> Dict[str, str]:
    """
    Load unique documents from a JSON file.

    Args:
        filepath (str): Path to the unique_documents_output.json file.

    Returns:
        Dict[str, str]: Dictionary mapping document IDs to their content.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            unique_docs = json.load(file)
        return unique_docs
    except Exception as e:
        logger.error(f"Failed to load unique documents: {e}")
        sys.exit(1)

def load_golden_standard(filepath: str, logger: logging.Logger) -> List[Dict]:
    """
    Load the Golden Standard dataset from a JSON file.

    Args:
        filepath (str): Path to the golden_standard_extracted_chunks.json file.

    Returns:
        List[Dict]: List of QA pairs with questions and expected answers.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            golden_data = json.load(file)
        return golden_data
    except Exception as e:
        logger.error(f"Failed to load Golden Standard dataset: {e}")
        sys.exit(1)

def test_llm_processor(
    llm_processor: LLMProcessor,
    evaluator: SimilarityEvaluator,
    unique_docs: Dict[str, str],
    golden_data: List[Dict],
    logger: logging.Logger,
    similarity_threshold: float = 0.8
):
    """
    Automated test method to iterate through all questions in the Golden Standard dataset,
    process the unique documents, compare the summaries with expected answers, and log the outcomes.

    Args:
        llm_processor (LLMProcessor): The LLM processor instance.
        evaluator (SimilarityEvaluator): The evaluator instance for similarity scoring.
        unique_docs (Dict[str, str]): Dictionary of unique documents.
        golden_data (List[Dict]): Golden Standard dataset with questions and expected answers.
        logger (logging.Logger): Logger instance.
        similarity_threshold (float): Threshold for similarity to consider as pass.
    """
    logger.info("=== Starting LLM Post Processor Test Module ===\n")

    total_questions = len(golden_data)
    passed_questions = []
    failed_questions = []

    for entry in golden_data:
        qa_id = entry.get('id', 'Unknown ID')
        question = entry.get('question', '').strip()
        expected_answer = entry.get('answers', '').strip()

        logger.info(f"--- Testing QA ID: {qa_id} ---")
        logger.info(f"Question: {question}")

        # Retrieve corresponding unique documents
        unique_doc_content = unique_docs.get(str(qa_id))
        if not unique_doc_content:
            logger.warning(f"No unique documents found for QA ID {qa_id}.")
            failed_questions.append(qa_id)
            continue

        # Create a mock 'Document' list
        # Ensure that 'metadata' is provided if required by the LLMProcessor
        Document = namedtuple("Document", ["page_content", "metadata"])
        # Extract metadata from the unique_doc_content if available
        # For this example, we'll parse 'Source' and 'Page' from the document content
        # This is a simplistic approach; adjust as needed based on actual data structure

        def extract_metadata(content: str) -> Dict:
            metadata = {}
            lines = content.split('\n')
            for line in lines:
                if line.startswith("Source:"):
                    metadata['source'] = line.replace("Source:", "").strip()
                elif line.startswith("Page:"):
                    metadata['page'] = line.replace("Page:", "").strip()
            return metadata

        metadata = extract_metadata(unique_doc_content)
        raw_results = [Document(page_content=unique_doc_content, metadata=metadata)]

        # Process the query
        summary = llm_processor.process_query_response(question, raw_results)

        logger.info(f"Expected Answer: {expected_answer}")
        logger.info(f"Generated Summary: {summary}")

        # Compute similarity between expected answer and generated summary
        similarity = evaluator.compute_similarity(expected_answer, summary, logger)
        logger.info(f"Similarity Score: {similarity:.4f}")

        if similarity >= similarity_threshold:
            logger.info("Evaluation Result: PASS\n")
            passed_questions.append(qa_id)
        else:
            logger.info("Evaluation Result: FAIL\n")
            failed_questions.append(qa_id)

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
            logger.info(f" - QA ID: {pid}")
    else:
        logger.info("No questions passed the similarity threshold.")

    # Log Failed Questions
    if failed_questions:
        logger.info(f"\nFailed Questions (Total: {len(failed_questions)}):")
        for fid in failed_questions:
            logger.info(f" - QA ID: {fid}")
    else:
        logger.info("No questions failed the similarity threshold.")

    logger.info("=== End of LLM Post Processor Test Module ===")

def main(logger: logging.Logger, test_log_filepath: str):
    """
    Main function to initialize the LLMProcessor, load datasets, initialize the evaluator,
    and start the automated test module.
    """
    # Load environment variables
    load_dotenv(find_dotenv())

    AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_4O = os.getenv("OPENAI_MODEL_4o")
    AZURE_OPENAI_4OMINI = os.getenv("OPENAI_MODEL_4OMINI")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
    AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

    # Validate environment variables
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_MODEL, AZURE_API_VERSION]):
        logger.error("One or more Azure OpenAI environment variables are missing.")
        sys.exit(1)

    # Configuration
    UNIQUE_DOCS_FILEPATH = os.path.join(current_dir,"unique_documents_output.json") 
    
    GOLDEN_STANDARD_FILEPATH = os.path.join(current_dir, "golden_standard_extracted_chunks.json")

    # Validate files
    if not os.path.isfile(UNIQUE_DOCS_FILEPATH):
        logger.error(f"Unique Documents file does not exist: {UNIQUE_DOCS_FILEPATH}")
        sys.exit(1)
    if not os.path.isfile(GOLDEN_STANDARD_FILEPATH):
        logger.error(f"Golden Standard file does not exist: {GOLDEN_STANDARD_FILEPATH}")
        sys.exit(1)

    # Load datasets
    unique_docs = load_unique_documents(UNIQUE_DOCS_FILEPATH, logger)
    logger.info(f"Loaded Unique Documents with {len(unique_docs)} entries.\n")

    golden_data = load_golden_standard(GOLDEN_STANDARD_FILEPATH, logger)
    logger.info(f"Loaded Golden Standard dataset with {len(golden_data)} QA pairs.\n")

    # Initialize the LLMProcessor
    llm_processor = LLMProcessor(
         azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_deployment=AZURE_OPENAI_4OMINI,
        azure_api_version=AZURE_API_VERSION,
        log_file=os.path.join(current_dir,"..", "logs", "llm_post_processor_test.log")
    )

    # Initialize the SimilarityEvaluator
    evaluator = SimilarityEvaluator(
        azure_openai_api_key=AZURE_OPENAI_API_KEY,
        azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_openai_embedding_deployment=AZURE_OPENAI_EMBEDDING_MODEL
    )

    # Start the test module
    test_llm_processor(llm_processor, evaluator, unique_docs, golden_data, logger)

if __name__ == "__main__":
    test_log_filepath = os.path.join(current_dir,"..", "logs", "llm_post_processor_test.log")
    logger = setup_logger(test_log_filepath)
    main(logger, test_log_filepath)