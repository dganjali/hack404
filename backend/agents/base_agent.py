from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.schema import BaseMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import re

class BaseAgent(ABC):
    """Base class for all AI agents in ProcessUnravel Pro"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model = ChatOpenAI(model_name=model_name, temperature=0.1)
        self.prompt_template = None
        
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return structured output"""
        pass
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        try:
            # Find JSON pattern in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except json.JSONDecodeError:
            return {}
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate agent output structure"""
        return isinstance(output, dict) and len(output) > 0 