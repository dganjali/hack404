from .base_agent import BaseAgent
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from models import StepType
import re

class StepClassifierAgent(BaseAgent):
    """Agent that classifies chunks into specific step types"""
    
    def __init__(self):
        super().__init__()
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert at classifying government process steps into specific categories.
        
        Given a content chunk from a government process, classify it into one of these step types:
        
        - ACTION: A step that requires the user to do something (fill form, visit office, etc.)
        - DOCUMENT: A step that requires providing or obtaining documents
        - FEE: A step that involves payment or fees
        - WAIT: A step that involves waiting periods or processing time
        - DECISION: A step that involves making a choice or decision point
        - INFO: Informational content that doesn't require action
        
        Content to classify:
        {content}
        
        Return a JSON object with:
        - step_type: the classified type (ACTION, DOCUMENT, FEE, WAIT, DECISION, INFO)
        - title: a clear, concise title for this step
        - description: a brief description of what this step involves
        - cost: any mentioned fees or costs (null if none)
        - duration: any mentioned time requirements (null if none)
        - required_documents: list of documents mentioned (empty array if none)
        - conditions: any conditions or requirements mentioned (empty array if none)
        
        Return only valid JSON:
        """)
    
    def process(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a single chunk into a step"""
        try:
            content = chunk.get('content', '')
            title = chunk.get('title', '')
            
            # Combine title and content for analysis
            full_content = f"Title: {title}\n\nContent: {content}"
            
            # Create prompt
            prompt = self.prompt_template.format(content=full_content)
            
            # Get response
            response = self.model.predict(prompt)
            
            # Extract JSON
            result = self.extract_json_from_response(response)
            
            if self.validate_output(result):
                # Ensure step_type is valid
                step_type = result.get('step_type', 'INFO')
                if step_type not in [e.value for e in StepType]:
                    step_type = 'INFO'
                
                return {
                    'id': chunk.get('id', 'unknown'),
                    'type': step_type,
                    'title': result.get('title', title),
                    'description': result.get('description', ''),
                    'cost': result.get('cost'),
                    'duration': result.get('duration'),
                    'required_documents': result.get('required_documents', []),
                    'conditions': result.get('conditions', []),
                    'depends_on': [],
                    'outputs': [],
                    'metadata': {
                        'original_chunk': chunk,
                        'section_type': chunk.get('section_type', 'unknown')
                    }
                }
            else:
                return self._fallback_classification(chunk)
                
        except Exception as e:
            print(f"Error in StepClassifierAgent: {e}")
            return self._fallback_classification(chunk)
    
    def _fallback_classification(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback classification using rule-based approach"""
        content = chunk.get('content', '').lower()
        title = chunk.get('title', '').lower()
        full_text = f"{title} {content}"
        
        # Rule-based classification
        step_type = 'INFO'  # default
        
        # Check for actions
        action_keywords = ['apply', 'submit', 'register', 'file', 'complete', 'fill out']
        if any(keyword in full_text for keyword in action_keywords):
            step_type = 'ACTION'
        
        # Check for documents
        doc_keywords = ['document', 'certificate', 'license', 'permit', 'form', 'proof']
        if any(keyword in full_text for keyword in doc_keywords):
            step_type = 'DOCUMENT'
        
        # Check for fees
        fee_keywords = ['fee', 'cost', 'payment', '$', 'dollars', 'charge']
        if any(keyword in full_text for keyword in fee_keywords):
            step_type = 'FEE'
        
        # Check for waiting
        wait_keywords = ['wait', 'time', 'days', 'weeks', 'processing', 'review']
        if any(keyword in full_text for keyword in wait_keywords):
            step_type = 'WAIT'
        
        # Check for decisions
        decision_keywords = ['choose', 'select', 'option', 'either', 'or', 'decision']
        if any(keyword in full_text for keyword in decision_keywords):
            step_type = 'DECISION'
        
        # Extract cost information
        cost = None
        cost_match = re.search(r'\$[\d,]+(?:\.\d{2})?', content)
        if cost_match:
            cost = cost_match.group()
        
        # Extract duration information
        duration = None
        duration_match = re.search(r'(\d+)\s*(days?|weeks?|months?)', content)
        if duration_match:
            duration = f"{duration_match.group(1)} {duration_match.group(2)}"
        
        # Extract documents
        documents = []
        doc_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:certificate|license|permit|form))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+document)',
        ]
        for pattern in doc_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            documents.extend(matches)
        
        return {
            'id': chunk.get('id', 'unknown'),
            'type': step_type,
            'title': chunk.get('title', 'Untitled'),
            'description': content[:200] + '...' if len(content) > 200 else content,
            'cost': cost,
            'duration': duration,
            'required_documents': list(set(documents)),
            'conditions': [],
            'depends_on': [],
            'outputs': [],
            'metadata': {
                'original_chunk': chunk,
                'section_type': chunk.get('section_type', 'unknown'),
                'classification_method': 'rule_based'
            }
        } 