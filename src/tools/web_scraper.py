#!/usr/bin/env python3
"""
Web Scraper Tool - Provides functionality for extracting data from websites.
"""
from typing import Any, Dict, List, Optional, Union
import logging
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

# Configure logging
logger = logging.getLogger(__name__)


class WebScraperTool:
    """
    Tool for scraping data from websites.
    
    This tool provides:
    1. HTTP request handling with proper headers
    2. HTML parsing and extraction using BeautifulSoup
    3. Structured data extraction from web pages
    4. Rate limiting and retry functionality
    """

    def __init__(
        self,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        request_delay: float = 1.0,
        max_retries: int = 3,
    ):
        """
        Initialize the WebScraperTool.
        
        Args:
            user_agent: User-Agent header to use in requests
            request_delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts
        """
        self.user_agent = user_agent
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self.last_request_time = 0
    
    def _respect_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
            
        self.last_request_time = time.time()
    
    def fetch_page(
        self, 
        url: str, 
        params: Optional[Dict[str, str]] = None, 
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Fetch a web page with rate limiting and retries.
        
        Args:
            url: URL to fetch
            params: URL parameters to include
            headers: Additional HTTP headers
            
        Returns:
            HTML content of page or None if request failed
        """
        params = params or {}
        headers = headers or {}
        
        for attempt in range(self.max_retries + 1):
            try:
                self._respect_rate_limit()
                response = self.session.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                return response.text
            except RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{self.max_retries+1}): {e}")
                if attempt == self.max_retries:
                    logger.error(f"Maximum retries reached for URL: {url}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def parse_html(self, html_content: str) -> BeautifulSoup:
        """
        Parse HTML content with BeautifulSoup.
        
        Args:
            html_content: HTML content to parse
            
        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(html_content, "html.parser")
    
    def extract_text(self, soup: BeautifulSoup, selector: str) -> List[str]:
        """
        Extract text from elements matching selector.
        
        Args:
            soup: BeautifulSoup object
            selector: CSS selector
            
        Returns:
            List of text content from matching elements
        """
        elements = soup.select(selector)
        return [el.get_text(strip=True) for el in elements]
    
    def extract_links(
        self, 
        soup: BeautifulSoup, 
        selector: str = "a", 
        base_url: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Extract links from elements matching selector.
        
        Args:
            soup: BeautifulSoup object
            selector: CSS selector
            base_url: Base URL to resolve relative links
            
        Returns:
            List of dictionaries with href and text for each link
        """
        elements = soup.select(selector)
        links = []
        
        for el in elements:
            href = el.get("href")
            if not href:
                continue
                
            # Resolve relative URLs if base_url is provided
            if base_url and not bool(urlparse(href).netloc):
                from urllib.parse import urljoin
                href = urljoin(base_url, href)
                
            links.append({
                "href": href,
                "text": el.get_text(strip=True),
            })
            
        return links
    
    def extract_table(self, soup: BeautifulSoup, selector: str = "table") -> List[List[str]]:
        """
        Extract data from HTML table.
        
        Args:
            soup: BeautifulSoup object
            selector: CSS selector for table
            
        Returns:
            List of rows, each row being a list of cell values
        """
        table = soup.select_one(selector)
        if not table:
            return []
            
        rows = []
        for tr in table.find_all("tr"):
            row = []
            for cell in tr.find_all(["td", "th"]):
                row.append(cell.get_text(strip=True))
            if row:  # Skip empty rows
                rows.append(row)
                
        return rows
    
    def scrape_page(self, url: str, extraction_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape a page applying multiple extraction rules.
        
        Args:
            url: URL to scrape
            extraction_rules: Dictionary mapping output keys to extraction selectors and types
            
        Returns:
            Dictionary of scraped data
        """
        html = self.fetch_page(url)
        if not html:
            return {"error": "Failed to fetch page"}
            
        soup = self.parse_html(html)
        results = {}
        
        for key, rule in extraction_rules.items():
            selector = rule.get("selector", "")
            extract_type = rule.get("type", "text")
            
            if extract_type == "text":
                results[key] = self.extract_text(soup, selector)
            elif extract_type == "links":
                results[key] = self.extract_links(soup, selector, url)
            elif extract_type == "table":
                results[key] = self.extract_table(soup, selector)
            elif extract_type == "html":
                elements = soup.select(selector)
                results[key] = [str(el) for el in elements]
                
        return results