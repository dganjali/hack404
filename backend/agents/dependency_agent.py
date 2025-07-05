from .base_agent import BaseAgent
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
import re

class DependencyAgent(BaseAgent):
    """Agent that identifies dependencies between process steps"""
    
    def __init__(self):
        super().__init__()
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert at analyzing government processes and identifying dependencies between steps.
        
        Given a list of process steps, identify which steps depend on other steps and what outputs from one step become inputs to another.
        
        Steps to analyze:
        {steps}
        
        For each step, identify:
        1. What other steps it depends on (prerequisites)
        2. What outputs it produces that other steps might need
        3. Any conditional dependencies (e.g., "if food business, then...")
        
        Return a JSON object where each step ID maps to:
        - depends_on: list of step IDs this step depends on
        - outputs: list of outputs this step produces
        - conditions: any conditional logic for dependencies
        
        Return only valid JSON:
        """)
    
    def process(self, steps: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze dependencies between steps"""
        try:
            # Format steps for analysis
            steps_text = self._format_steps_for_analysis(steps)
            
            # Create prompt
            prompt = self.prompt_template.format(steps=steps_text)
            
            # Get response
            response = self.model.predict(prompt)
            
            # Extract JSON
            dependencies = self.extract_json_from_response(response)
            
            if self.validate_output(dependencies):
                return self._validate_dependencies(dependencies, steps)
            else:
                return self._fallback_dependency_analysis(steps)
                
        except Exception as e:
            print(f"Error in DependencyAgent: {e}")
            return self._fallback_dependency_analysis(steps)
    
    def _format_steps_for_analysis(self, steps: List[Dict[str, Any]]) -> str:
        """Format steps into readable text for analysis"""
        formatted = []
        for step in steps:
            step_text = f"Step {step['id']}: {step['title']}\n"
            step_text += f"Type: {step['type']}\n"
            if step.get('description'):
                step_text += f"Description: {step['description']}\n"
            if step.get('required_documents'):
                step_text += f"Documents: {', '.join(step['required_documents'])}\n"
            if step.get('conditions'):
                step_text += f"Conditions: {', '.join(step['conditions'])}\n"
            step_text += "---\n"
            formatted.append(step_text)
        
        return "\n".join(formatted)
    
    def _validate_dependencies(self, dependencies: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Validate and clean dependency data"""
        step_ids = {step['id'] for step in steps}
        validated = {}
        
        for step_id, deps in dependencies.items():
            if step_id in step_ids:
                validated[step_id] = {
                    'depends_on': [dep for dep in deps.get('depends_on', []) if dep in step_ids],
                    'outputs': deps.get('outputs', []),
                    'conditions': deps.get('conditions', [])
                }
        
        return validated
    
    def _fallback_dependency_analysis(self, steps: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Fallback dependency analysis using rule-based approach"""
        dependencies = {}
        
        # Create a mapping of document outputs to step IDs
        doc_to_step = {}
        for step in steps:
            if step.get('required_documents'):
                for doc in step['required_documents']:
                    doc_to_step[doc.lower()] = step['id']
        
        # Analyze each step for dependencies
        for i, step in enumerate(steps):
            step_id = step['id']
            depends_on = []
            outputs = []
            
            # Check for document dependencies
            if step.get('required_documents'):
                for doc in step['required_documents']:
                    doc_lower = doc.lower()
                    # Find which step produces this document
                    for other_step in steps:
                        if other_step['id'] != step_id:
                            # Check if other step mentions this document as output
                            other_content = f"{other_step.get('title', '')} {other_step.get('description', '')}".lower()
                            if doc_lower in other_content:
                                depends_on.append(other_step['id'])
            
            # Check for sequential dependencies (simple heuristic)
            if i > 0:
                # Assume some dependency on previous step
                depends_on.append(steps[i-1]['id'])
            
            # Identify outputs based on step type
            if step['type'] == 'DOCUMENT':
                outputs.extend(step.get('required_documents', []))
            elif step['type'] == 'ACTION':
                outputs.append(f"{step['title']} completion")
            
            dependencies[step_id] = {
                'depends_on': list(set(depends_on)),
                'outputs': outputs,
                'conditions': step.get('conditions', [])
            }
        
        return dependencies
    
    def update_step_dependencies(self, steps: List[Dict[str, Any]], dependencies: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update steps with dependency information"""
        updated_steps = []
        
        for step in steps:
            step_id = step['id']
            step_deps = dependencies.get(step_id, {})
            
            updated_step = step.copy()
            updated_step['depends_on'] = step_deps.get('depends_on', [])
            updated_step['outputs'] = step_deps.get('outputs', [])
            
            updated_steps.append(updated_step)
        
        return updated_steps 