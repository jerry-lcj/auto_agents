#!/usr/bin/env python3
"""
Web Scraper Tool - Provides functionality for extracting data from websites.
"""
from typing import Any, Dict, List, Optional, Union
import logging
import time
import json
import urllib
from urllib.parse import urlparse, quote_plus, urljoin
import urllib.request
import re

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

    def extract_main_content(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract the main content from a webpage using various strategies.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of content paragraphs
        """
        content = []
        
        # Try common content container selectors
        content_selectors = [
            "article", "main", ".post-content", ".entry-content", ".article-content",
            "#content", "#main", ".content", ".post", ".blog-post",
            "[role='main']", "[itemprop='articleBody']", ".story"
        ]
        
        # Try each content selector
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                # Extract paragraphs from the first matching element
                element = elements[0]
                paragraphs = element.find_all("p")
                if paragraphs:
                    content = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
                    if content:  # Found content, break the loop
                        break
        
        # If no content found with content selectors, get all paragraphs
        if not content:
            paragraphs = soup.find_all("p")
            content = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
        
        # If still no content, try getting text from divs with substantial text
        if not content:
            divs = soup.find_all("div")
            for div in divs:
                text = div.get_text(strip=True)
                if len(text) > 100:  # Only consider divs with substantial text
                    content.append(text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_content = []
        for item in content:
            normalized = ' '.join(item.split())  # Normalize whitespace
            if normalized not in seen and len(normalized) > 20:  # Only keep substantial content
                seen.add(normalized)
                unique_content.append(item)
        
        return unique_content
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web for the given query using multiple search methods.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of dictionaries with title, snippet, and url
        """
        logger.info(f"Searching web for: {query}")
        
        # Try multiple search methods until we get results
        methods = [
            self._search_duckduckgo,
            self._search_direct_sources,
        ]
        
        results = []
        for method in methods:
            try:
                method_results = method(query, num_results)
                if method_results:
                    results = method_results
                    logger.info(f"Got {len(results)} results from {method.__name__}")
                    break
            except Exception as e:
                logger.error(f"Error in search method {method.__name__}: {e}")
        
        return results
    
    def _search_duckduckgo(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search using DuckDuckGo lite."""
        # Using DuckDuckGo lite for searching (no JavaScript required)
        search_url = "https://lite.duckduckgo.com/lite"
        
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
            
            logger.info(f"Found {len(results)} search results from DuckDuckGo")
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _search_direct_sources(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Find relevant sources based on the query keywords.
        This is used when external search engines fail.
        """
        query_lower = query.lower()
        words = re.findall(r'\w+', query_lower)
        domains = []
        
        # First check for specific platform mentions in the query
        platform_domains = {
            "twitter": ["twitter.com", "nitter.net", "tweetdeck.twitter.com", "hootsuite.com/twitter", 
                      "socialmediatoday.com/twitter", "sproutsocial.com/insights/twitter-trends"],
            "facebook": ["facebook.com", "fb.com", "meta.com", "socialmediatoday.com/facebook"],
            "instagram": ["instagram.com", "socialbakers.com/instagram", "sproutsocial.com/instagram"],
            "linkedin": ["linkedin.com", "business.linkedin.com", "socialmediaexaminer.com/linkedin"],
            "reddit": ["reddit.com", "old.reddit.com", "subredditstats.com"],
            "tiktok": ["tiktok.com", "influencermarketinghub.com/tiktok-stats"],
        }
        
        # Check if any platform is explicitly mentioned in the query
        platform_mentioned = False
        for platform, platform_sites in platform_domains.items():
            if platform in query_lower:
                domains.extend(platform_sites)
                platform_mentioned = True
                logger.info(f"Detected platform in query: {platform}")
        
        # If we found platform-specific domains, prioritize them and add fewer general domains
        if platform_mentioned:
            # Map other keywords to relevant domains but give them lower priority
            keyword_to_domains = {
                "news": ["techcrunch.com", "theverge.com", "wired.com"],
                "tech": ["techcrunch.com", "wired.com", "theverge.com"],
                "ai": ["arxiv.org/abs/cs.AI", "ai.googleblog.com", "openai.com/blog"],
                "machine": ["machinelearning.org", "paperswithcode.com"],
                "learning": ["machinelearning.org", "paperswithcode.com"],
                "science": ["sciencemag.org", "pnas.org"],
                "research": ["researchgate.net", "scholar.google.com"],
                "trending": ["trendsmap.com", "trends.google.com", "buzzsumo.com"],
                "topics": ["buzzsumo.com", "trends.google.com"],
                "analysis": ["mediapost.com", "pewresearch.org"]
            }
            
            # Add additional domains based on query words, but with lower priority
            for word in words:
                if word in keyword_to_domains:
                    domains.extend(keyword_to_domains[word][:1])  # Only add 1 domain per keyword when platform is mentioned
        else:
            # If no platform mentioned, use the standard approach with broader domain mapping
            keyword_to_domains = {
                "news": ["techcrunch.com", "theverge.com", "wired.com", "cnn.com", "bbc.com", "reuters.com"],
                "tech": ["techcrunch.com", "wired.com", "theverge.com", "arstechnica.com", "cnet.com"],
                "ai": ["arxiv.org", "distill.pub", "ai.googleblog.com", "openai.com/blog", "deepmind.com"],
                "machine": ["arxiv.org", "machinelearning.org", "ai.googleblog.com", "paperswithcode.com"],
                "learning": ["arxiv.org", "machinelearning.org", "ai.googleblog.com", "paperswithcode.com"],
                "data": ["kaggle.com", "data.gov", "dataverse.org", "data.world", "figshare.com"],
                "science": ["nature.com", "sciencemag.org", "pnas.org", "arxiv.org", "plos.org"],
                "social": ["socialmediatoday.com", "sproutsocial.com", "buffer.com/resources"],
                "media": ["socialmediatoday.com", "sproutsocial.com", "buffer.com/resources"],
                "twitter": ["twitter.com", "nitter.net", "tweetdeck.twitter.com", "hootsuite.com/twitter"],
                "research": ["arxiv.org", "researchgate.net", "scholar.google.com", "sciencedirect.com"],
                "market": ["bloomberg.com", "marketwatch.com", "ft.com", "cnbc.com", "reuters.com"],
                "business": ["hbr.org", "bloomberg.com", "ft.com", "cnbc.com", "wsj.com"],
                "trending": ["trendsmap.com", "trends.google.com", "buzzsumo.com"],
                "topics": ["buzzsumo.com", "trends.google.com"],
                "analysis": ["mediapost.com", "pewresearch.org"]
            }
            
            # Add domains based on query words
            for word in words:
                if word in keyword_to_domains:
                    domains.extend(keyword_to_domains[word])
        
        # If no specific domains matched, use general knowledge sources
        if not domains:
            domains = ["wikipedia.org", "theverge.com", "medium.com", "techcrunch.com", "arxiv.org"]
        
        # Remove duplicates while preserving order
        unique_domains = []
        seen = set()
        for domain in domains:
            if domain not in seen:
                seen.add(domain)
                unique_domains.append(domain)
        
        # Generate results
        results = []
        for domain in unique_domains[:num_results]:
            # Create a basic URL for the domain
            if not domain.startswith(("http://", "https://")):
                url = "https://" + domain
            else:
                url = domain
                
            # Extract domain name for title
            domain_name = domain.split(".")[-2] if "." in domain else domain
            domain_name = domain_name.capitalize()
                
            results.append({
                "title": f"{domain_name} - Information related to {query}",
                "snippet": f"Content from {domain} that may be relevant to your search about {query}.",
                "url": url
            })
        
        logger.info(f"Generated {len(results)} direct source results")
        return results
    
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
                
                # Extract main content using our generic extraction method
                content = self.extract_main_content(soup)
                
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
                
                logger.info(f"Successfully extracted {len(content)} content items from {url}")
                
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
            elif extract_type == "main_content":
                results[key] = self.extract_main_content(soup)
                
        return results