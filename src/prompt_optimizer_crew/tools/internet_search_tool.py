"""
Internet Search Tool for agents to gather contextual information.
"""

import logging
import time
from typing import Optional, Type

from crewai_tools import BaseTool
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException, DuckDuckGoSearchException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

logger = logging.getLogger(__name__)


class InternetSearchToolInput(BaseModel):
    """Input schema for InternetSearchTool."""
    query: str = Field(..., description="The search query string")


class InternetSearchTool(BaseTool):
    """
    Tool for searching the internet to gather contextual information.
    
    This tool allows agents to search for relevant information online
    to enhance their evaluation and understanding of prompts.
    """
    
    # Fix Pydantic V2 deprecation warning
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "Internet Search"
    description: str = (
        "Search the internet for information. Useful for finding current information, "
        "verifying terminology, understanding domain-specific concepts, or gathering "
        "context about a topic. Input should be a search query string."
    )
    args_schema: Type[BaseModel] = InternetSearchToolInput
    
    # Use PrivateAttr for attributes that shouldn't be part of the Pydantic model
    _max_results: int = PrivateAttr(default=5)
    _max_retries: int = PrivateAttr(default=3)
    _retry_delay: float = PrivateAttr(default=2.0)
    
    def __init__(self, max_results: int = 5, max_retries: int = 3, retry_delay: float = 2.0, **kwargs):
        """
        Initialize the internet search tool.
        
        Args:
            max_results: Maximum number of search results to return
            max_retries: Maximum number of retry attempts for rate limiting
            retry_delay: Initial delay between retries (exponential backoff)
        """
        super().__init__(**kwargs)
        self._max_results = max_results
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        logger.info("Internet Search Tool initialized")
    
    def _run(self, query: str) -> str:
        """
        Execute a search query.
        
        Args:
            query: The search query string
            
        Returns:
            Formatted search results as a string
        """
        return self.search(query)
    
    def _search_with_retry(self, query: str, max_results: int, search_type: str = "text") -> list:
        """
        Execute search with retry logic for rate limiting.
        
        Args:
            query: The search query
            max_results: Maximum number of results
            search_type: Type of search ("text" or "news")
            
        Returns:
            List of search results
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self._max_retries):
            try:
                with DDGS() as ddgs:
                    if search_type == "news":
                        results = list(ddgs.news(query, max_results=max_results))
                    else:
                        results = list(ddgs.text(query, max_results=max_results))
                return results
                
            except RatelimitException as e:
                last_exception = e
                if attempt < self._max_retries - 1:
                    # Exponential backoff
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{self._max_retries}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Rate limit exceeded after {self._max_retries} attempts")
                    
            except DuckDuckGoSearchException as e:
                last_exception = e
                logger.error(f"DuckDuckGo search error: {str(e)}")
                break  # Don't retry on non-rate-limit errors
                
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error during search: {str(e)}", exc_info=True)
                break
        
        # If we get here, all retries failed
        raise last_exception if last_exception else Exception("Search failed with unknown error")
    
    def search(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Search the internet using DuckDuckGo with retry logic.
        
        Args:
            query: The search query
            max_results: Optional override for max results
            
        Returns:
            Formatted string with search results or error message
        """
        if not query or not query.strip():
            return "Error: Empty search query provided"
        
        max_results = max_results or self._max_results
        
        try:
            logger.info(f"Searching for: {query}")
            
            # Use retry logic for search
            results = self._search_with_retry(query, max_results, "text")
            
            if not results:
                return f"No results found for query: {query}"
            
            # Format results
            formatted_results = [f"Search results for: {query}\n"]
            formatted_results.append("=" * 80 + "\n")
            
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                body = result.get('body', 'No description')
                url = result.get('href', 'No URL')
                
                formatted_results.append(f"\n{i}. {title}")
                formatted_results.append(f"   {body}")
                formatted_results.append(f"   URL: {url}\n")
            
            formatted_output = "\n".join(formatted_results)
            logger.debug(f"Found {len(results)} results")
            
            return formatted_output
            
        except RatelimitException as e:
            error_msg = (
                f"Search rate limit exceeded for query '{query}'. "
                "Please try again later or reduce search frequency."
            )
            logger.error(error_msg)
            return error_msg
            
        except DuckDuckGoSearchException as e:
            error_msg = f"DuckDuckGo search error for query '{query}': {str(e)}"
            logger.error(error_msg)
            return error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during search for query '{query}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
    
    def search_news(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Search for news articles with retry logic.
        
        Args:
            query: The search query
            max_results: Optional override for max results
            
        Returns:
            Formatted string with news results or error message
        """
        if not query or not query.strip():
            return "Error: Empty search query provided"
        
        max_results = max_results or self._max_results
        
        try:
            logger.info(f"Searching news for: {query}")
            
            # Use retry logic for news search
            results = self._search_with_retry(query, max_results, "news")
            
            if not results:
                return f"No news found for query: {query}"
            
            # Format results
            formatted_results = [f"News results for: {query}\n"]
            formatted_results.append("=" * 80 + "\n")
            
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                body = result.get('body', 'No description')
                url = result.get('url', 'No URL')
                date = result.get('date', 'No date')
                
                formatted_results.append(f"\n{i}. {title}")
                formatted_results.append(f"   Date: {date}")
                formatted_results.append(f"   {body}")
                formatted_results.append(f"   URL: {url}\n")
            
            return "\n".join(formatted_results)
            
        except RatelimitException as e:
            error_msg = (
                f"News search rate limit exceeded for query '{query}'. "
                "Please try again later or reduce search frequency."
            )
            logger.error(error_msg)
            return error_msg
            
        except DuckDuckGoSearchException as e:
            error_msg = f"DuckDuckGo news search error for query '{query}': {str(e)}"
            logger.error(error_msg)
            return error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during news search for query '{query}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
