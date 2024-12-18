from typing import List, Dict
import json
import random
import os
from pydantic import BaseModel
from logger_config import setup_logger

class Promotion(BaseModel):
    id: str
    type: str
    content: str
    title: str
    category: str
    restaurant_name: str
    cuisine_type: str
    neighborhood: str

class PromotionsProvider:
    def __init__(self, promotions_file_path: str = None, log_file_path: str = None):
        self.logger = setup_logger(log_file_path or "logs/promotions.log")
        self.promotions_file_path = promotions_file_path or os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 
            "data", 
            "promotions.json"
        )
        self.promotions: List[Promotion] = self._load_promotions()

    def _load_promotions(self) -> List[Promotion]:
        """Load promotions from file"""
        try:
            with open(self.promotions_file_path, 'r') as f:
                data = json.load(f)
                return [Promotion(**item) for item in data]
        except Exception as e:
            self.logger.error(f"Error loading promotions: {str(e)}")
            return []

    def get_promotions(self, limit: int = 5) -> List[Dict]:
        """Get random promotions with specified limit"""
        try:
            selected_promotions = random.sample(
                self.promotions, 
                min(limit, len(self.promotions))
            )
            return [
                promo.model_dump(exclude_none=True) 
                for promo in selected_promotions
            ]
        except Exception as e:
            self.logger.error(f"Error getting promotions: {str(e)}")
            return []