#test_text_to_sql_engine.py
import json
from typing import Dict, List, Any
from datetime import datetime
import difflib
from dotenv import load_dotenv, find_dotenv
from text_to_sql_engine import TextToSQLEngine
from logger_config import setup_logger
import os
import sys
import ast
from langchain_openai  import AzureOpenAIEmbeddings,AzureChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
   

class QueryEvaluator:
    def __init__(self, golden_standard_file: str,azure_openai_api_key: str, azure_openai_endpoint: str, azure_openai_embedding_deployment: str,azure_openai_deployment_mini,azure_api_version):
        """Initialize the QueryEvaluator with golden standard data."""
        # Setup logger
        self.LOG_FILE = os.path.join('logs', 'test_text_to_sql_engine.log')
        self.logger = setup_logger(self.LOG_FILE)
        
        self.logger.info("Initializing QueryEvaluator")
        self.engine = TextToSQLEngine()
        
        try:
            with open(golden_standard_file, 'r',encoding='utf-8') as f:
                self.golden_data = json.load(f)
            self.logger.info(f"Successfully loaded golden standard data from {golden_standard_file}")
        except Exception as e:
            self.logger.error(f"Error loading golden standard file: {str(e)}")
            raise
            
        self.results = []
        self.embeddings = AzureOpenAIEmbeddings(
                model=azure_openai_embedding_deployment,
                api_key=azure_openai_api_key,
                azure_endpoint=azure_openai_endpoint,
                deployment=azure_openai_embedding_deployment,
            )
        
        self.model_4o_mini=AzureChatOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            deployment_name=azure_openai_deployment_mini,
            api_version=azure_api_version,
            temperature=0,
            max_tokens=3000
        )
        
    def normalize_query(self, query: str) -> str:
        """Normalize SQL query for comparison."""
        self.logger.debug(f"Normalizing query: {query}")
        # Remove extra whitespace and convert to lowercase
        query = ' '.join(query.lower().split())
        # Remove semicolon at the end if present
        return query.rstrip(';')
    
    def calculate_query_similarity(self, generated_query: str, golden_query: str) -> float:
        """Calculate similarity between two SQL queries using difflib."""
        self.logger.debug("Calculating query similarity")
        normalized_generated = self.normalize_query(generated_query)
        normalized_golden = self.normalize_query(golden_query)
        
        similarity = difflib.SequenceMatcher(None, 
                                           normalized_generated, 
                                           normalized_golden).ratio()
        self.logger.debug(f"------Normalized generated query: {normalized_generated}-------")
        self.logger.debug(f"------Normalized golden query: {normalized_golden}-------")
        self.logger.debug(f"Query similarity score: {similarity}")
        return similarity

    def parse_query_result(self, result_str: str) -> List[Dict]:
            """Convert query result string to list of dictionaries."""
            try:
                if not result_str:
                    return []
                    
                # If result is already a list of dictionaries
                if isinstance(result_str, list) and all(isinstance(x, dict) for x in result_str):
                    return result_str

                # If result is a string representation of a tuple list
                if isinstance(result_str, str):
                    # Convert string to Python literal structure
                    result_tuple_list = ast.literal_eval(result_str)
                    
                    # Get column names from the first query in golden standard
                    columns = self.extract_columns_from_query(self.golden_data['results'][0]['query'])
                    
                    # Convert tuples to dictionaries
                    result_dicts = []
                    for row in result_tuple_list:
                        result_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                result_dict[columns[i]] = value
                            else:
                                result_dict[f'column_{i}'] = value
                        result_dicts.append(result_dict)
                    return result_dicts
                    
            except Exception as e:
                self.logger.error(f"Error parsing query result: {str(e)}")
                return []

    def extract_columns_from_query(self, query: str) -> List[str]:
        """Extract column names from SELECT statement."""
        try:
            # Get the part between SELECT and FROM
            select_part = query.upper().split('FROM')[0].replace('SELECT', '').strip()
            
            # Split by comma and clean up each column name
            columns = []
            for col in select_part.split(','):
                col = col.strip()
                # Handle cases with table alias
                if '.' in col:
                    col = col.split('.')[-1]
                # Handle cases with AS
                if ' AS ' in col.upper():
                    col = col.split(' AS ')[-1]
                columns.append(col.strip().lower().replace('"', '').replace("'", ""))
            
            return columns
        except Exception as e:
            self.logger.error(f"Error extracting columns: {str(e)}")
            return ['column_0', 'column_1', 'column_2', 'column_3']  # fallback column names

    def compare_results(self, generated_result: List[Dict], golden_result: List[Dict]) -> float:
        """Compare query results using vector embeddings."""
        self.logger.debug("Comparing query results using embeddings")
        
        if not generated_result or not golden_result:
            similarity = 1.0 if generated_result == golden_result else 0.0
            self.logger.debug(f"Empty result comparison, similarity: {similarity}")
            return similarity
            
        try:
            # Convert results to string representation
            generated_str = json.dumps(generated_result, sort_keys=True)
            golden_str = json.dumps(golden_result, sort_keys=True)
            
            # Compare using vector embeddings
            similarity = self.compute_vector_similarity(generated_str, golden_str)
            
            self.logger.debug(f"Vector similarity score: {similarity}")
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error comparing results: {str(e)}")
            self.logger.error("Full error details: ", exc_info=True)
            return 0.0

    def compute_vector_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using embeddings."""
        try:
            # Generate embeddings
            embedding1 = self.embeddings.embed_documents([text1])[0]
            embedding2 = self.embeddings.embed_documents([text2])[0]
            # Compute cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error computing vector similarity: {str(e)}")
            return 0.0
    
    def rephrase_answer(self, question: str, query: str, result: str) -> str:
        """Process the answer using the model."""
        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding PostgreSQL query, and PostgreSQL result, format the data to be a json object for better processing .
    PostgreSQL Query: {query}
    PostgreSQL Result: {result}
    Answer: """
        )
        prompt_input = {
            "query": query,
            "result": result
        }
    
        prompt_result = answer_prompt.format_prompt(**prompt_input)
        model_response = self.model_4o_mini.invoke(prompt_result.to_string())
        return {
            "sql_query": query,
            "result": StrOutputParser().parse(model_response.content)
        }
    
    
    
    def evaluate_single_query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single query test case."""
        self.logger.info(f"Evaluating test case {test_case['id']}")
        input_query = test_case['input']
        
        try:
            output = self.engine.process_query(input_query)
            generated_output=self.rephrase_answer(input_query,output['sql_query'],output['result'])
            # Log the actual content for debugging
            self.logger.info("Generated output structure:")
            self.logger.info(f"Query Result Type: {type(generated_output['result'])}")
            
            self.logger.info(f"user input: {test_case['input']}")
            self.logger.info(f"--------Generated query start--------")
            self.logger.info(f"Generated query: {generated_output['sql_query']}")
            self.logger.info(f"--------Generated query end--------")
            self.logger.info(f"--------Golden query start--------")
            self.logger.info(f"Golden query: {test_case['query']}")
            self.logger.info(f"--------Golden query end--------")
            self.logger.info(f"---------------------------------------- \n")
            self.logger.info(f"--------Generated query result start--------")
            self.logger.info(f"Generated query result: {generated_output['result']}")
            self.logger.info(f"--------Generated query result end--------\n")
            self.logger.info(f"--------Golden query result start--------")
            self.logger.info(f"Golden query result: {test_case['result']}")
            self.logger.info(f"--------Golden query end--------")
            
            
            # Calculate metrics
            query_similarity = self.calculate_query_similarity(
                generated_output['sql_query'],
                test_case['query']
            )
            
            result_similarity = self.compare_results(
                generated_output['result'],
                test_case['result']
            )
            
            # Define pass/fail criteria
            QUERY_SIMILARITY_THRESHOLD = 0.7
            RESULT_SIMILARITY_THRESHOLD = 0.7
            
            query_passed = query_similarity >= QUERY_SIMILARITY_THRESHOLD
            result_passed = result_similarity >= RESULT_SIMILARITY_THRESHOLD
            overall_passed = query_passed or result_passed
            
            # Log pass/fail status
            self.logger.info(f"Test case {test_case['id']} - Query Similarity: {query_similarity:.2f} - {'PASSED' if query_passed else 'FAILED'}")
            self.logger.info(f"Test case {test_case['id']} - Result Similarity: {result_similarity:.2f} - {'PASSED' if result_passed else 'FAILED'}")
            self.logger.info(f"Test case {test_case['id']} - Overall Status: {'PASSED' if overall_passed else 'FAILED'}")
            
            evaluation_result = {
                'test_id': test_case['id'],
                'input_query': input_query,
                'generated_query': generated_output['sql_query'],
                'golden_query': test_case['query'],
                'query_similarity': query_similarity,
                'generated_result': generated_output['result'],
                'golden_result': test_case['result'],
                'result_similarity': result_similarity,
                'generated_answer': generated_output['result'],
                'timestamp': datetime.now().isoformat(),
                'status': {
                    'query_passed': query_passed,
                    'result_passed': result_passed,
                    'overall_passed': overall_passed
                }
            }
            
            self.logger.info(f"Successfully evaluated test case {test_case['id']}")
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error evaluating test case {test_case['id']}: {str(e)}")
            self.logger.error(f"Full error details: ", exc_info=True)
            raise
    
    
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all test cases and generate summary."""
        self.logger.info("Starting evaluation of all test cases")
        total_cases = len(self.golden_data['results'])
        evaluation_results = []
        
        # Track pass/fail statistics
        passed_cases = []
        failed_cases = []
        
        for test_case in self.golden_data['results']:
            try:
                result = self.evaluate_single_query(test_case)
                evaluation_results.append(result)
                
                # Track pass/fail status based on the new status structure
                if result['status']['overall_passed']:
                    passed_cases.append(test_case['id'])
                else:
                    failed_cases.append(test_case['id'])
                    
            except Exception as e:
                self.logger.error(f"Error evaluating test case {test_case['id']}: {str(e)}")
                self.logger.error("Full error details:", exc_info=True)
                failed_cases.append(test_case['id'])
                # Create a failed result entry
                evaluation_results.append({
                    'test_id': test_case['id'],
                    'input_query': test_case['input'],
                    'status': {
                        'query_passed': False,
                        'result_passed': False,
                        'overall_passed': False
                    },
                    'error': str(e)
                })
        
        # Calculate statistics
        total_passed = len(passed_cases)
        total_failed = len(failed_cases)
        pass_rate = (total_passed / total_cases) * 100
        fail_rate = (total_failed / total_cases) * 100
        
        # Calculate average similarities for successful evaluations only
        successful_results = [r for r in evaluation_results if 'query_similarity' in r]
        avg_query_similarity = sum(r['query_similarity'] for r in successful_results) / len(successful_results) if successful_results else 0
        avg_result_similarity = sum(r['result_similarity'] for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Generate detailed summary
        summary = {
            'metadata': {
                'total_cases': total_cases,
                'execution_date': datetime.now().isoformat(),
                'average_query_similarity': avg_query_similarity,
                'average_result_similarity': avg_result_similarity,
                'pass_rate': pass_rate,
                'fail_rate': fail_rate
            },
            'statistics': {
                'total_passed': total_passed,
                'total_failed': total_failed,
                'passed_cases': passed_cases,
                'failed_cases': failed_cases
            },
            'results': evaluation_results
        }
        
        # Log detailed summary
        self.logger.info("\n=== Evaluation Summary ===")
        self.logger.info(f"Total test cases: {total_cases}")
        self.logger.info(f"Passed cases: {total_passed} ({pass_rate:.2f}%)")
        self.logger.info(f"Failed cases: {total_failed} ({fail_rate:.2f}%)")
        self.logger.info("\nPassed test cases: " + ", ".join(map(str, passed_cases)) if passed_cases else "None")
        self.logger.info("Failed test cases: " + ", ".join(map(str, failed_cases)) if failed_cases else "None")
        self.logger.info(f"\nAverage query similarity: {avg_query_similarity:.2f}")
        self.logger.info(f"Average result similarity: {avg_result_similarity:.2f}")
        self.logger.info("========================")
        
        return summary

def main():
    """Main function to run the evaluation."""
    # Setup logger
    LOG_FILE = os.path.join('logs', 'test_text_to_sql_engine.log')
    logger = setup_logger(LOG_FILE)
      # Load environment variables
    load_dotenv(find_dotenv())

    # Retrieve environment variables
    AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT =os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_MINI=os.getenv("OPENAI_MODEL_4OMINI")
    logger.info("Starting evaluation process")
    try:
          # Validate environment variables
        if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_EMBEDDING_MODEL, AZURE_API_VERSION]):
            logger.error("One or more Azure OpenAI environment variables are missing.")
            sys.exit(1)
        model_config = {
            "azure_openai_api_key": AZURE_OPENAI_API_KEY,
            "azure_openai_endpoint": AZURE_OPENAI_ENDPOINT,
            "azure_openai_embedding_deployment": AZURE_OPENAI_EMBEDDING_MODEL,
            "azure_api_version": AZURE_API_VERSION,
            "azure_openai_deployment_mini": AZURE_OPENAI_DEPLOYMENT_MINI
        }
        evaluator = QueryEvaluator('query_results_test_set.json', **model_config)
        results = evaluator.evaluate_all()
        
        
        logger.info("\nEvaluation Summary:")
        logger.info(f"Total test cases: {results['metadata']['total_cases']}")
        logger.info(f"Average query similarity: {results['metadata']['average_query_similarity']:.2f}")
        logger.info(f"Average result similarity: {results['metadata']['average_result_similarity']:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main evaluation process: {str(e)}")
        raise

if __name__ == "__main__":
    main()