import pdfplumber
# # import pytesseract  # Skipped for now  # Skipped for now
from PIL import Image
import io
from typing import List, Dict, Any, Optional
import re
import os

class PDFProcessor:
    """PDF processor for government documents using pdfplumber and OCR"""
    
    def __init__(self):
        # Configure pytesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF file"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_content = []
                page_info = []
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    
                    # If no text found, try OCR
                    if not text or len(text.strip()) < 50:
                        text = self._ocr_page(page)
                    
                    if text:
                        text_content.append(text)
                        page_info.append({
                            'page_number': page_num + 1,
                            'text_length': len(text),
                            'has_text': bool(text.strip())
                        })
                
                return {
                    'success': True,
                    'file_path': pdf_path,
                    'total_pages': len(pdf.pages),
                    'text_content': text_content,
                    'page_info': page_info,
                    'full_text': '\n\n'.join(text_content)
                }
                
        except Exception as e:
            return {
                'success': False,
                'file_path': pdf_path,
                'error': str(e),
                'text_content': [],
                'page_info': [],
                'full_text': ''
            }
    
    def _ocr_page(self, page) -> str:
        """Extract text from page using OCR"""
        try:
            # OCR functionality disabled for now
            # To enable: install pytesseract and uncomment the import
            return ""
            
        except Exception as e:
            print(f"OCR failed: {e}")
            return ""
    
    def extract_forms_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract form information from PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                forms = []
                
                for page_num, page in enumerate(pdf.pages):
                    # Look for form elements
                    form_elements = self._extract_form_elements(page)
                    
                    if form_elements:
                        forms.append({
                            'page_number': page_num + 1,
                            'form_elements': form_elements
                        })
                
                return forms
                
        except Exception as e:
            print(f"Error extracting forms: {e}")
            return []
    
    def _extract_form_elements(self, page) -> List[Dict[str, Any]]:
        """Extract form elements from a page"""
        elements = []
        
        # Look for text boxes, checkboxes, etc.
        if hasattr(page, 'extract_words'):
            words = page.extract_words()
            for word in words:
                # Look for form-like patterns
                if self._looks_like_form_field(word['text']):
                    elements.append({
                        'type': 'text_field',
                        'text': word['text'],
                        'bbox': word['bbox']
                    })
        
        return elements
    
    def _looks_like_form_field(self, text: str) -> bool:
        """Check if text looks like a form field"""
        form_patterns = [
            r'^\s*_+\s*$',  # Underscores
            r'^\s*□\s*$',   # Checkbox
            r'^\s*\[\s*\]\s*$',  # Empty checkbox
            r'^\s*\(\s*\)\s*$',  # Empty radio button
            r'^\s*Name:\s*$',
            r'^\s*Address:\s*$',
            r'^\s*Phone:\s*$',
            r'^\s*Email:\s*$',
        ]
        
        return any(re.match(pattern, text) for pattern in form_patterns)
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                tables = []
                
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:  # At least header + one row
                            tables.append({
                                'page_number': page_num + 1,
                                'table_number': table_num + 1,
                                'data': table,
                                'rows': len(table),
                                'columns': len(table[0]) if table else 0
                            })
                
                return tables
                
        except Exception as e:
            print(f"Error extracting tables: {e}")
            return []
    
    def process_government_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a government PDF and extract structured information"""
        result = {
            'file_path': pdf_path,
            'success': False,
            'text_content': '',
            'forms': [],
            'tables': [],
            'metadata': {}
        }
        
        try:
            # Extract text
            text_result = self.extract_text_from_pdf(pdf_path)
            if text_result['success']:
                result['text_content'] = text_result['full_text']
                result['metadata']['total_pages'] = text_result['total_pages']
                result['metadata']['page_info'] = text_result['page_info']
            
            # Extract forms
            result['forms'] = self.extract_forms_from_pdf(pdf_path)
            
            # Extract tables
            result['tables'] = self.extract_tables_from_pdf(pdf_path)
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result

def process_pdf_file(pdf_path: str) -> Dict[str, Any]:
    """Convenience function to process a PDF file"""
    processor = PDFProcessor()
    return processor.process_government_pdf(pdf_path) 