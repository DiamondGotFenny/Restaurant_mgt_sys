import json
import os
import sys
from dotenv import load_dotenv, find_dotenv
from  vectorDB_Agent.bm25_retriever_agent import BM25RetrieverAgent
current_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    """
    Main function to execute the BM25 searching based on 'qa_keywords.json' and output the results to 'golden_standard_Raw_Chunks.json'.
    """
    # Load environment variables from a .env file if present
    load_dotenv(find_dotenv())

    # Configuration
    PDF_DIRECTORY = os.path.join(current_dir,"..","data","Restaurants_data") 
    LOG_FILE = os.path.join(current_dir,"..","logs","bm25_retriever_agent.log")
    WHOOSH_INDEX_DIR = os.path.join(current_dir,"..","data","whoosh_index")

    # Validate directories
    if not os.path.isdir(PDF_DIRECTORY):
        print(f"PDF directory does not exist: {PDF_DIRECTORY}")
        sys.exit(1)

    # Initialize the BM25RetrieverAgent with specified configurations
    agent = BM25RetrieverAgent(
        pdf_directory=PDF_DIRECTORY,
        log_file=LOG_FILE,
        chunk_size=1000,
        chunk_overlap=200,
        bm25_params={"k1": 0.5, "b": 0.75},
        whoosh_index_dir=WHOOSH_INDEX_DIR
    )

    if not agent.documents:
        print("No documents loaded. Exiting.")
        sys.exit(1)

    # Load the 'qa_keywords.json' file
    QA_KEYWORDS_FILE = os.path.join(current_dir, "qa_keywords.json")
    if not os.path.isfile(QA_KEYWORDS_FILE):
        print(f"The file '{QA_KEYWORDS_FILE}' does not exist in the current directory.")
        sys.exit(1)

    with open(QA_KEYWORDS_FILE, 'r', encoding='utf-8') as f:
        try:
            qa_data = json.load(f)
        except json.JSONDecodeError as jde:
            print(f"Error decoding JSON from '{QA_KEYWORDS_FILE}': {jde}")
            sys.exit(1)

    if 'qa_pairs' not in qa_data:
        print("The JSON structure is invalid. Expected 'qa_pairs' key.")
        sys.exit(1)

    output = []

    for qa_pair in qa_data['qa_pairs']:
        question_text = qa_pair.get('question', '').strip()
        if not question_text:
            print("A Q&A pair is missing the 'question' field or it's empty. Skipping this pair.")
            continue

        restaurants = qa_pair.get('restaurants', {})
        if not restaurants:
            print(f"No restaurants found for the question: '{question_text}'. Skipping this pair.")
            continue

        qa_output = {
            "question": question_text,
            "answers": []
        }

        for restaurant_name in restaurants.keys():
            # Perform BM25 query for each restaurant name
            query_json = {"entities": [restaurant_name]}
            try:
                results = agent.query_documents(query=query_json, top_k=1)
            except Exception as e:
                print(f"Error querying BM25 for restaurant '{restaurant_name}': {e}")
                continue

            if not results:
                print(f"No results found for restaurant '{restaurant_name}'.")
                continue

            top_result = results[0]
            answer_content = top_result.page_content.strip().replace('\n', '      ')
            source = top_result.metadata.get('source', 'Unknown Source')
            page = top_result.metadata.get('page', 'N/A')

            answer_entry = {
                "answer": answer_content,
                "source": source,
                "page": page
            }

            qa_output["answers"].append(answer_entry)

        output.append(qa_output)

    # Define the output JSON file name
    OUTPUT_FILE = "golden_standard_Raw_Chunks.json"

    # Write the output to the JSON file
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            json.dump(output, outfile, indent=2, ensure_ascii=False)
        print(f"Successfully wrote the query results to '{OUTPUT_FILE}'.")
    except Exception as e:
        print(f"Error writing to '{OUTPUT_FILE}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()