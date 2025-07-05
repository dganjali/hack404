import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import requests
from typing import List, Dict, Any, Optional
import re
from urllib.parse import urljoin, urlparse

class WebScraper:
    """Web scraper for government websites using Playwright"""
    
    def __init__(self):
        self.session = None
        self.browser = None
        self.page = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a single URL and extract content"""
        try:
            await self.page.goto(url, wait_until='networkidle')
            
            # Get page content
            content = await self.page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            return {
                'url': url,
                'title': metadata.get('title', ''),
                'content': main_content,
                'metadata': metadata,
                'links': self._extract_links(soup, url),
                'success': True
            }
            
        except Exception as e:
            return {
                'url': url,
                'title': '',
                'content': '',
                'metadata': {},
                'links': [],
                'success': False,
                'error': str(e)
            }
    
    async def scrape_multiple_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently"""
        tasks = [self.scrape_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'url': urls[i],
                    'title': '',
                    'content': '',
                    'metadata': {},
                    'links': [],
                    'success': False,
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content areas
        main_selectors = [
            'main',
            '[role="main"]',
            '.main-content',
            '.content',
            '#content',
            'article',
            '.article-content'
        ]
        
        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            # Fallback to body
            main_content = soup.find('body')
        
        if main_content:
            # Clean up the content
            text = main_content.get_text(separator='\n', strip=True)
            # Remove extra whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text.strip()
        
        return ""
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from the page"""
        metadata = {
            'title': '',
            'description': '',
            'keywords': [],
            'author': '',
            'last_modified': '',
            'domain': urlparse(url).netloc
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            
            if name == 'description':
                metadata['description'] = content
            elif name == 'keywords':
                metadata['keywords'] = [kw.strip() for kw in content.split(',')]
            elif name == 'author':
                metadata['author'] = content
        
        return metadata
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract links from the page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text(strip=True)
            
            if href and text:
                # Make relative URLs absolute
                absolute_url = urljoin(base_url, href)
                links.append({
                    'url': absolute_url,
                    'text': text
                })
        
        return links
    
    def _is_government_domain(self, url: str) -> bool:
        """Check if URL is from a government domain"""
        domain = urlparse(url).netloc.lower()
        gov_domains = [
            '.gov', '.gc.ca', '.gov.on.ca', '.gov.ab.ca', '.gov.bc.ca',
            '.gov.ns.ca', '.gov.nb.ca', '.gov.pe.ca', '.gov.nl.ca',
            '.gov.nt.ca', '.gov.nu.ca', '.gov.yt.ca', '.gov.sk.ca',
            '.gov.mb.ca', '.gov.qc.ca'
        ]
        return any(gov_domain in domain for gov_domain in gov_domains)

async def scrape_government_sites(urls: List[str]) -> List[Dict[str, Any]]:
    """Convenience function to scrape government sites"""
    async with WebScraper() as scraper:
        return await scraper.scrape_multiple_urls(urls) 