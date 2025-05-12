#!/usr/bin/env python3
"""
Web Scraper Tool - Provides functionality for extracting data from websites.
"""
from typing import Any, Dict, List, Optional, Union
import logging
import time
import json
import urllib
from urllib.parse import urlparse, quote_plus
import urllib.request

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
    5. Web search capabilities via search engines
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
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        })
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
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web for the given query using DuckDuckGo.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of dictionaries with title, snippet, and url
        """
        logger.info(f"Searching web for: {query}")
        
        # Using DuckDuckGo lite for searching (no JavaScript required)
        search_url = f"https://lite.duckduckgo.com/lite"
        
        try:
            # Format the query parameters
            params = {
                'q': query,
                'kl': 'us-en'  # Region and language
            }
            
            # Make the search request
            self._respect_rate_limit()
            response = self.session.post(search_url, data=params, timeout=30)
            response.raise_for_status()
            
            # Parse the response
            soup = self.parse_html(response.text)
            
            # Extract search results
            results = []
            
            # DuckDuckGo lite uses a specific table structure
            for tr in soup.select("tr.result-link"):
                try:
                    title_el = tr.select_one("td > a")
                    if not title_el:
                        continue
                        
                    title = title_el.get_text(strip=True)
                    url = title_el.get("href")
                    
                    # Get the snippet from the next row
                    snippet_tr = tr.find_next_sibling("tr")
                    if snippet_tr:
                        snippet_td = snippet_tr.select_one("td.result-snippet")
                        if snippet_td:
                            snippet = snippet_td.get_text(strip=True)
                        else:
                            snippet = ""
                    else:
                        snippet = ""
                    
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url
                    })
                    
                    if len(results) >= num_results:
                        break
                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue
            
            logger.info(f"Found {len(results)} search results")
            
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            
            # Return an empty list on error
            return []
    
    def search_and_scrape(self, query: str, max_pages: int = 3) -> Dict[str, Any]:
        """
        Search for query and scrape the top results.
        
        Args:
            query: Search query
            max_pages: Maximum number of pages to scrape
            
        Returns:
            Dictionary with search results and scraped content
        """
        logger.info(f"Performing search and scrape for query: {query}")
        
        # First, search the web
        search_results = self.search_web(query, num_results=max_pages)
        
        # Then scrape each result
        scraped_data = []
        
        for result in search_results:
            url = result.get("url")
            if not url:
                continue
                
            logger.info(f"Scraping search result: {url}")
            
            try:
                html = self.fetch_page(url)
                if not html:
                    logger.warning(f"Failed to fetch page: {url}")
                    continue
                    
                soup = self.parse_html(html)
                
                # Extract main content - try different selectors for main content
                content = []
                
                # Try article content first
                article_selectors = ["article", "main", ".post-content", ".entry-content", "#content", "#main"]
                for selector in article_selectors:
                    elements = soup.select(selector)
                    if elements:
                        # Extract paragraphs from the first matching element
                        paragraphs = elements[0].find_all("p")
                        content = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
                        break
                
                # If no content found with article selectors, just get all paragraphs
                if not content:
                    paragraphs = soup.find_all("p")
                    content = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
                
                # Extract title
                title = result.get("title", "")
                if not title and soup.title:
                    title = soup.title.get_text(strip=True)
                
                # Add to scraped data
                scraped_data.append({
                    "url": url,
                    "title": title,
                    "content": content,
                    "snippet": result.get("snippet", "")
                })
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                continue
        
        return {
            "query": query,
            "results": search_results,
            "scraped_data": scraped_data,
            "timestamp": time.time()
        }
    
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