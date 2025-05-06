#!/usr/bin/env python3
"""
UI Automation Tool - Provides browser and UI automation capabilities.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
import base64
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class UIAutomationTool:
    """
    Tool for UI automation using Playwright.
    
    This tool provides:
    1. Headless browser automation with Playwright
    2. Element interaction (click, type, etc.)
    3. Screenshot capture
    4. Form filling and submission
    5. Navigation and page state management
    """

    def __init__(
        self,
        headless: Optional[bool] = None,
        screenshot_dir: Optional[str] = None,
        slow_mo: Optional[int] = None,
    ):
        """
        Initialize the UIAutomationTool.
        
        Args:
            headless: Whether to run the browser in headless mode
            screenshot_dir: Directory to save screenshots
            slow_mo: Slow down operations by this many milliseconds
        """
        # Get settings from environment variables with defaults
        headless_str = os.getenv("HEADLESS_BROWSER", "true").lower()
        self.headless = headless if headless is not None else (headless_str == "true")
        self.screenshot_dir = screenshot_dir or os.getenv("SCREENSHOT_DIR", "./screenshots")
        self.slow_mo = slow_mo or int(os.getenv("SLOW_MO", "50"))
        self.browser = None
        self.page = None
        
        # Create screenshot directory if it doesn't exist
        os.makedirs(self.screenshot_dir, exist_ok=True)

    async def start_browser(self, browser_type: str = "chromium"):
        """
        Launch a browser instance.
        
        Args:
            browser_type: Type of browser to launch (chromium, firefox, or webkit)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import Playwright only when needed to avoid startup dependency issues
            from playwright.async_api import async_playwright
            
            self.playwright = await async_playwright().start()
            
            # Select browser type
            if browser_type == "chromium":
                browser_instance = self.playwright.chromium
            elif browser_type == "firefox":
                browser_instance = self.playwright.firefox
            elif browser_type == "webkit":
                browser_instance = self.playwright.webkit
            else:
                logger.error(f"Unsupported browser type: {browser_type}")
                return False
                
            # Launch browser
            self.browser = await browser_instance.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
            )
            
            # Create a new page
            self.page = await self.browser.new_page()
            
            logger.info(f"Started {browser_type} browser")
            return True
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            return False

    async def close_browser(self):
        """
        Close the browser instance.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.browser:
                await self.browser.close()
                await self.playwright.stop()
                self.browser = None
                self.page = None
                logger.info("Browser closed")
            return True
        except Exception as e:
            logger.error(f"Failed to close browser: {e}")
            return False

    async def navigate(self, url: str, wait_until: str = "load") -> bool:
        """
        Navigate to a URL.
        
        Args:
            url: URL to navigate to
            wait_until: Navigation wait condition
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.page:
                logger.error("Browser not started. Call start_browser() first.")
                return False
                
            response = await self.page.goto(url, wait_until=wait_until)
            if response:
                logger.info(f"Navigated to {url}, status: {response.status}")
                return response.ok
            return False
        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            return False

    async def take_screenshot(
        self, 
        name: str = "screenshot", 
        full_page: bool = False
    ) -> Optional[str]:
        """
        Take a screenshot of the current page.
        
        Args:
            name: Base name for the screenshot file
            full_page: Whether to capture the full page or just the viewport
            
        Returns:
            Path to the screenshot file or None if failed
        """
        try:
            if not self.page:
                logger.error("Browser not started. Call start_browser() first.")
                return None
                
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.png"
            filepath = os.path.join(self.screenshot_dir, filename)
            
            await self.page.screenshot(path=filepath, full_page=full_page)
            logger.info(f"Screenshot saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None

    async def click(
        self, 
        selector: str, 
        timeout: int = 30000,
        force: bool = False
    ) -> bool:
        """
        Click on an element.
        
        Args:
            selector: CSS selector for the element
            timeout: Maximum time to wait for the element in ms
            force: Whether to force-click the element
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.page:
                logger.error("Browser not started. Call start_browser() first.")
                return False
                
            await self.page.click(selector, timeout=timeout, force=force)
            logger.info(f"Clicked element: {selector}")
            return True
        except Exception as e:
            logger.error(f"Failed to click element {selector}: {e}")
            return False

    async def fill_form(
        self,
        form_data: Dict[str, str],
        submit_selector: Optional[str] = None
    ) -> bool:
        """
        Fill form fields and optionally submit the form.
        
        Args:
            form_data: Dictionary mapping selectors to values
            submit_selector: Selector for submit button (if None, form won't be submitted)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.page:
                logger.error("Browser not started. Call start_browser() first.")
                return False
                
            # Fill each form field
            for selector, value in form_data.items():
                await self.page.fill(selector, value)
                logger.info(f"Filled {selector} with value")
                
            # Submit form if requested
            if submit_selector:
                await self.click(submit_selector)
                logger.info("Form submitted")
                
            return True
        except Exception as e:
            logger.error(f"Failed to fill form: {e}")
            return False

    async def extract_text(self, selector: str) -> Optional[str]:
        """
        Extract text content from an element.
        
        Args:
            selector: CSS selector for the element
            
        Returns:
            Text content or None if failed
        """
        try:
            if not self.page:
                logger.error("Browser not started. Call start_browser() first.")
                return None
                
            element = await self.page.query_selector(selector)
            if element:
                text = await element.text_content()
                return text.strip()
            return None
        except Exception as e:
            logger.error(f"Failed to extract text from {selector}: {e}")
            return None

    async def extract_multiple_texts(self, selector: str) -> List[str]:
        """
        Extract text from multiple elements matching the selector.
        
        Args:
            selector: CSS selector for the elements
            
        Returns:
            List of text content from matching elements
        """
        try:
            if not self.page:
                logger.error("Browser not started. Call start_browser() first.")
                return []
                
            elements = await self.page.query_selector_all(selector)
            result = []
            
            for element in elements:
                text = await element.text_content()
                result.append(text.strip())
                
            return result
        except Exception as e:
            logger.error(f"Failed to extract multiple texts from {selector}: {e}")
            return []

    async def wait_for_navigation(self, timeout: int = 30000) -> bool:
        """
        Wait for navigation to complete.
        
        Args:
            timeout: Maximum time to wait in ms
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.page:
                logger.error("Browser not started. Call start_browser() first.")
                return False
                
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Failed waiting for navigation: {e}")
            return False


# Helper function to use the tool in synchronous context
def run_async(coroutine):
    """Run an async function from synchronous code."""
    try:
        return asyncio.get_event_loop().run_until_complete(coroutine)
    except RuntimeError:
        # If there's no running event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coroutine)
        finally:
            loop.close()