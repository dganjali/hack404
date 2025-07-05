from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict, Any
import tempfile
import os

from scrapers.web_scraper import scrape_government_sites
from scrapers.pdf_processor import process_pdf_file

router = APIRouter()

@router.post("/web")
async def scrape_websites(urls: List[str]):
    """Scrape multiple government websites"""
    try:
        if not urls:
            raise HTTPException(status_code=400, detail="No URLs provided")
        
        # Validate URLs
        for url in urls:
            if not url.startswith(('http://', 'https://')):
                raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")
        
        # Scrape websites
        results = await scrape_government_sites(urls)
        
        # Process results
        successful_scrapes = []
        failed_scrapes = []
        
        for result in results:
            if result['success']:
                successful_scrapes.append({
                    'url': result['url'],
                    'title': result['title'],
                    'content_length': len(result['content']),
                    'metadata': result['metadata']
                })
            else:
                failed_scrapes.append({
                    'url': result['url'],
                    'error': result.get('error', 'Unknown error')
                })
        
        return {
            'total_urls': len(urls),
            'successful_scrapes': len(successful_scrapes),
            'failed_scrapes': len(failed_scrapes),
            'results': {
                'successful': successful_scrapes,
                'failed': failed_scrapes
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping websites: {str(e)}")

@router.post("/pdf")
async def process_pdf_upload(file: UploadFile = File(...)):
    """Process uploaded PDF file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process the PDF
            result = process_pdf_file(temp_file_path)
            
            return {
                'filename': file.filename,
                'file_size': len(content),
                'success': result['success'],
                'total_pages': result.get('metadata', {}).get('total_pages', 0),
                'text_content': result.get('text_content', ''),
                'forms_found': len(result.get('forms', [])),
                'tables_found': len(result.get('tables', [])),
                'error': result.get('error')
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.get("/status")
async def get_scraping_status():
    """Get scraping service status"""
    return {
        "status": "healthy",
        "services": {
            "web_scraping": "available",
            "pdf_processing": "available"
        },
        "supported_formats": [
            "HTML/Web pages",
            "PDF documents"
        ]
    } 