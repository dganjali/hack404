from .base_agent import BaseAgent
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
import re

class PageChunkerAgent(BaseAgent):
    """Agent that breaks input content into logical chunks"""
    
    def __init__(self):
        super().__init__()
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert at analyzing government process content and breaking it into logical chunks.
        
        Given the following content from a government website or document, break it into logical sections.
        Each section should represent a distinct step or phase in the process.
        
        Content to analyze:
        {content}
        
        Return a JSON array where each object represents a chunk with:
        - id: unique identifier (e.g., "chunk_1")
        - title: descriptive title for the chunk
        - content: the actual text content
        - section_type: the type of section (e.g., "introduction", "step", "requirement", "form", "payment")
        
        Focus on identifying:
        1. Process steps and actions
        2. Required documents and forms
        3. Fees and payments
        4. Waiting periods and timelines
        5. Conditions and requirements
        
        Return only valid JSON:
        """)
    
    def process(self, content: str) -> List[Dict[str, Any]]:
        """Break content into logical chunks"""
        try:
            # Clean content
            cleaned_content = self._clean_content(content)
            
            # Create prompt
            prompt = self.prompt_template.format(content=cleaned_content)
            
            # Get response
            response = self.model.predict(prompt)
            
            # Extract JSON
            chunks = self.extract_json_from_response(response)
            
            if isinstance(chunks, list):
                return chunks
            elif isinstance(chunks, dict) and 'chunks' in chunks:
                return chunks['chunks']
            else:
                return self._fallback_chunking(cleaned_content)
                
        except Exception as e:
            print(f"Error in PageChunkerAgent: {e}")
            return self._fallback_chunking(content)
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        # Remove HTML tags if present
        content = re.sub(r'<[^>]+>', '', content)
        return content.strip()
    
    def _fallback_chunking(self, content: str) -> List[Dict[str, Any]]:
        """Fallback chunking using simple heuristics"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        chunk_id = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this looks like a new section
            if self._is_section_header(line):
                if current_chunk:
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "title": self._extract_title(current_chunk),
                        "content": "\n".join(current_chunk),
                        "section_type": self._classify_section(current_chunk)
                    })
                    chunk_id += 1
                    current_chunk = []
            
            current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "title": self._extract_title(current_chunk),
                "content": "\n".join(current_chunk),
                "section_type": self._classify_section(current_chunk)
            })
        
        return chunks
    
    def _is_section_header(self, line: str) -> bool:
        """Check if line looks like a section header"""
        # Check for numbered lists, bold text, or all caps
        patterns = [
            r'^\d+\.',  # Numbered list
            r'^[A-Z][A-Z\s]+$',  # All caps
            r'^[A-Z][a-z]+:',  # Title case with colon
        ]
        return any(re.match(pattern, line) for pattern in patterns)
    
    def _extract_title(self, chunk: List[str]) -> str:
        """Extract title from chunk"""
        if not chunk:
            return "Untitled"
        
        # Use first line as title
        first_line = chunk[0].strip()
        if len(first_line) < 100:  # Reasonable title length
            return first_line
        else:
            return first_line[:50] + "..."
    
    def _classify_section(self, chunk: List[str]) -> str:
        """Classify section type based on content"""
        content = " ".join(chunk).lower()
        
        if any(word in content for word in ['form', 'application', 'submit']):
            return "form"
        elif any(word in content for word in ['fee', 'cost', 'payment', '$']):
            return "payment"
        elif any(word in content for word in ['document', 'certificate', 'license']):
            return "document"
        elif any(word in content for word in ['wait', 'time', 'days', 'weeks']):
            return "wait"
        else:
            return "step" 