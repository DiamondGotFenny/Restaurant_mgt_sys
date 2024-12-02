from openai import AzureOpenAI
import os
import json
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
_ = load_dotenv(find_dotenv())
class QueryRewriter:
    def __init__(self, client: AzureOpenAI):
        self.client = client
        self.max_history_rounds = 4

    def _create_context_from_history(self, chat_history: List[Dict]) -> str:
        """Create a condensed context from recent relevant chat history."""
        # Filter out system messages and convert to a format easier for the model to process
        relevant_history = []
        for msg in chat_history:
            if msg['sender'] not in ['system']:
                relevant_history.append(f"{msg['sender']}: {msg['text']}")
        
        # Join the last few messages to provide context
        return "\n".join(relevant_history[-4:])  # Use last 4 messages for context

    def rewrite_query(self, current_query: str, chat_history: List[Dict]) -> Dict:
        """
        Rewrite the query using chat history context when necessary.
        Returns a standardized JSON response for all cases.
        """
        context = self._create_context_from_history(chat_history)
        
        system_prompt = """You are a query rewriting assistant for a NYC restaurant information system. You should carefully analyze the conversation context to understand user's intentions, especially for follow-up questions about specific restaurants or their attributes. 

Analyze queries in the context of recent conversation history and respond with a JSON object in the following format:

{
    "status": string,        // One of: "needs_clarification", "rewritten", "unchanged"
    "query": string,         // The rewritten query or original query
    "confidence": float,     // Confidence score between 0 and 1
    "reasoning": string,     // Brief explanation of the decision
    "suggested_clarification": string | null  // Required if status is "needs_clarification", null otherwise
}

Guidelines for each status:

1. "needs_clarification" (confidence < 0.7):
   - ONLY when truly ambiguous with multiple possible interpretations
   - When NO clear reference can be found in recent context
   - When critical context is completely missing
   Example: {
     "status": "needs_clarification",
     "query": "Which one is cheaper?",
     "confidence": 0.4,
     "reasoning": "Multiple restaurants mentioned (Adda, Tamarind, Indian Accent) with no clear reference to any specific one",
     "suggested_clarification": "Could you specify which restaurants you'd like to compare prices for?"
   }

2. "rewritten" (confidence >= 0.7):
   - When the restaurant name or topic is clearly referenced in recent context
   - When a single entity is being discussed, even if mentioned briefly
   - When follow-up questions relate to previously mentioned restaurants
   - When query can be improved or extended for better understanding
   Example: {
     "status": "rewritten",
     "query": "What is Sobaya's price range and popular dishes?",
     "confidence": 0.9,
     "reasoning": "User is clearly asking about Sobaya, which was mentioned in recent conversation",
     "suggested_clarification": null
   }

3. "unchanged" (confidence = 1.0):
   - When query is already clear and self-contained
   Example: {
     "status": "unchanged",
     "query": "What are the best Italian restaurants in Manhattan?",
     "confidence": 1.0,
     "reasoning": "Query is already explicit and self-contained",
     "suggested_clarification": null
   }

Important:
- For short queries mentioning a restaurant name from recent context (e.g., "Sobaya"), assume user wants general information
- For follow-up questions about attributes (price, menu, etc.), connect them to the most recently discussed restaurant
- If a restaurant was just mentioned or discussed, prefer rewriting over asking for clarification
- Only ask for clarification when truly necessary, not just because the query is brief

Return ONLY the JSON object, no additional text or explanations."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Recent Conversation Context:
{context}

Current Query: {current_query}

Analyze this query and provide the appropriate JSON response based on the guidelines."""}
        ]

        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_4o"),
                messages=messages,
                temperature=0.1,
                response_format={ "type": "json_object" }  # Ensure JSON response
            )
            
            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"Error in query rewriting: {e}")
            # Return a basic unchanged response in case of error
            return {
                "status": "unchanged",
                "query": current_query,
                "confidence": 1.0,
                "reasoning": "Error in processing, returning original query",
                "suggested_clarification": None
            }

class QueryRewriterTester:
    def __init__(self, test_data_path: str):
        self.test_data_path = test_data_path
        self.load_test_data()
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Initialize query rewriter
        self.rewriter = QueryRewriter(
            self.client, 
        )

    def load_test_data(self):
        """Load test cases from JSON file"""
        with open(self.test_data_path, 'r') as f:
            self.test_data = json.load(f)

    def run_single_test_case(self, test_case: Dict):
        """Run a single test case"""
        print(f"\nRunning test case: {test_case['id']}")
        print(f"Description: {test_case['description']}")
        print("-" * 50)

        chat_history = []
        
        for idx, interaction in enumerate(test_case['conversation_flow']):
            # Current query
            current_query = interaction['text']
            expected_result = interaction['expected_result']
            
            print(f"\nTurn {idx + 1}:")
            print(f"Original query: {current_query}")
            
            # Get rewritten query
            result = self.rewriter.rewrite_query(current_query, chat_history)
            print(f"Result:")
            print(json.dumps(result, indent=2))
            
            # Compare with expected result
            print("\nValidation:")
            
            # Check status
            status_match = result['status'] == expected_result['status']
            print(f"Status: {'✓' if status_match else '✗'} "
                f"(Got: {result['status']}, Expected: {expected_result['status']})")
            
            # Check query
            query_match = result['query'] == expected_result['query']
            print(f"Query: {'✓' if query_match else '✗'} "
                f"(Got: {result['query']}, Expected: {expected_result['query']})")
            
            # Check confidence (with small tolerance for floating point comparison)
            confidence_match = abs(result['confidence'] - expected_result['confidence']) < 0.1
            print(f"Confidence: {'✓' if confidence_match else '✗'} "
                f"(Got: {result['confidence']}, Expected: {expected_result['confidence']})")
            
            # Check if clarification is needed and matches
            if expected_result['status'] == 'needs_clarification':
                clarification_needed = (result['suggested_clarification'] is not None and 
                                    expected_result['suggested_clarification'] is not None)
                print(f"Clarification needed: {'✓' if clarification_needed else '✗'}")
                if clarification_needed:
                    print(f"Suggested clarification:\n  Got: {result['suggested_clarification']}\n  "
                        f"Expected: {expected_result['suggested_clarification']}")
            
            # Update chat history
            chat_history.extend([
                {
                    "id": interaction['id'],
                    "text": current_query,
                    "sender": "user",
                    "timestamp": datetime.now()
                },
                {
                    "id": f"{interaction['id']}_response",
                    "text": interaction['assistant_response'],
                    "sender": "assistant",
                    "timestamp": datetime.now()
                }
            ])
            
            # Overall test result
            all_passed = status_match and query_match and confidence_match
            print(f"\nOverall test result: {'✓ Passed' if all_passed else '✗ Failed'}")

    def run_all_tests(self):
        """Run all test cases"""
        print("Starting Query Rewriter Tests")
        print("============================")
        
        for test_case in self.test_data['test_cases']:
            self.run_single_test_case(test_case)

    def add_test_case(self, test_case: Dict):
        """Add a new test case to the dataset"""
        self.test_data['test_cases'].append(test_case)
        self.save_test_data()

    def save_test_data(self):
        """Save test cases back to JSON file"""
        with open(self.test_data_path, 'w') as f:
            json.dump(self.test_data, f, indent=4)
def main():
    # Initialize and run tests
    test_data_path = "test_conversations.json"
    tester = QueryRewriterTester(test_data_path)
    
    tester.run_all_tests()
if __name__ == "__main__":
    main()