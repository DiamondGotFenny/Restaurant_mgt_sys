from typing import Dict, Any
from openai import AzureOpenAI
import os
import json
from logger_config import setup_logger

class RelevanceChecker:
    def __init__(self, log_file_path: str):
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.logger = setup_logger(log_file_path)

    def check_relevance(self, query: str) -> Dict[str, Any]:
        """Check if the query is relevant to NYC dining."""
        self.logger.info("Checking relevance of the query.")
        prompt = """Determine if this query is relevant to NYC dining.

        RELEVANT TOPICS INCLUDE:
        - NYC restaurants and dining establishments
        - Restaurant reviews, ratings, or recommendations in NYC
        - Food type or cuisine or specific dishes that users show interest in
        - Menu items, prices, or cuisine types in NYC restaurants
        - Restaurant locations, neighborhoods, or accessibility in NYC
        - Restaurant safety, inspections, or ratings in NYC
        - Restaurant business hours, reservations, or delivery options in NYC
        - Specific NYC restaurants or dining experiences
        - If the query doesn't tell any city or location, you can assume it's in NYC

        Return ONLY a JSON with these fields:
        {
            "is_relevant": boolean,
            "reasoning": string
        }"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_4o"),
                messages=messages,
                response_format={ "type": "json_object" },
                temperature=0.1
            )
            self.logger.info("Relevance check completed.")
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Error in relevance check: {e}")
            return {"is_relevant": False, "reasoning": "Error in analysis"}