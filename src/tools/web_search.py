"""
Web search tool for LangGraph Lab.
"""
import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from ..config.settings import get_settings
from ..utils.logging import get_logger


class WebSearchTool:
    """
    Web search tool that can use multiple search providers.
    Supports DuckDuckGo (free) and optionally Serper/Tavily (API-based).
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("tools.web_search")
        
        # Determine available search providers
        self.providers = ["duckduckgo"]  # Always available
        
        if self.settings.serper_api_key:
            self.providers.append("serper")
        if self.settings.tavily_api_key:
            self.providers.append("tavily")
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        provider: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            provider: Specific provider to use (if None, uses best available)
            
        Returns:
            List of search result dictionaries
        """
        if not query.strip():
            self.logger.warning("Empty query provided")
            return []
        
        # Select provider
        if provider and provider in self.providers:
            selected_provider = provider
        else:
            # Use best available provider
            if "serper" in self.providers:
                selected_provider = "serper"
            elif "tavily" in self.providers:
                selected_provider = "tavily"
            else:
                selected_provider = "duckduckgo"
        
        self.logger.info(f"Searching with {selected_provider}: {query}")
        
        try:
            if selected_provider == "serper":
                return await self._search_serper(query, max_results)
            elif selected_provider == "tavily":
                return await self._search_tavily(query, max_results)
            else:
                return await self._search_duckduckgo(query, max_results)
        except Exception as e:
            self.logger.error(f"Search failed with {selected_provider}: {e}")
            # Fallback to DuckDuckGo if other providers fail
            if selected_provider != "duckduckgo":
                self.logger.info("Falling back to DuckDuckGo search")
                return await self._search_duckduckgo(query, max_results)
            return []
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (free, no API key required)."""
        try:
            # DuckDuckGo instant answer API
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
            
            results = []
            
            # Process instant answer
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", "DuckDuckGo Result"),
                    "snippet": data.get("Abstract"),
                    "url": data.get("AbstractURL", ""),
                    "source": "duckduckgo"
                })
            
            # Process related topics
            for topic in data.get("RelatedTopics", [])[:max_results-len(results)]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "").split(" - ")[0],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "source": "duckduckgo"
                    })
            
            # If we don't have enough results, use web scraping as fallback
            if len(results) < 2:
                web_results = await self._scrape_search_results(query, max_results)
                results.extend(web_results)
            
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    async def _search_serper(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Serper API."""
        if not self.settings.serper_api_key:
            raise ValueError("Serper API key not configured")
        
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.settings.serper_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": max_results
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                data = await response.json()
        
        results = []
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
                "source": "serper"
            })
        
        return results
    
    async def _search_tavily(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        if not self.settings.tavily_api_key:
            raise ValueError("Tavily API key not configured")
        
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                data = await response.json()
        
        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("content", ""),
                "url": item.get("url", ""),
                "source": "tavily"
            })
        
        return results
    
    async def _scrape_search_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback method using web scraping for search results."""
        try:
            # Use a simple search engine that doesn't require API
            search_url = f"https://www.startpage.com/sp/search?query={quote_plus(query)}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers) as response:
                    html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            # Extract search results (this is a simplified scraper)
            for result_elem in soup.find_all('div', class_='w-gl__result')[:max_results]:
                title_elem = result_elem.find('h3')
                link_elem = result_elem.find('a')
                snippet_elem = result_elem.find('p', class_='w-gl__description')
                
                if title_elem and link_elem:
                    results.append({
                        "title": title_elem.get_text(strip=True),
                        "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                        "url": link_elem.get('href', ''),
                        "source": "scraped"
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Web scraping error: {e}")
            return []
    
    async def get_page_content(self, url: str, max_length: int = 5000) -> str:
        """
        Fetch and extract text content from a web page.
        
        Args:
            url: URL to fetch
            max_length: Maximum length of content to return
            
        Returns:
            Extracted text content
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:max_length] if len(text) > max_length else text
            
        except Exception as e:
            self.logger.error(f"Error fetching page content from {url}: {e}")
            return ""