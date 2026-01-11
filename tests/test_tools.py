"""
Unit tests for custom tools.
"""

import pytest
from unittest.mock import Mock, patch

from src.prompt_optimizer_crew.tools.domain_classifier_tool import DomainClassifierTool
from src.prompt_optimizer_crew.tools.internet_search_tool import InternetSearchTool


class TestDomainClassifierTool:
    """Test cases for DomainClassifierTool."""
    
    @pytest.fixture
    def classifier(self):
        """Create a classifier instance for testing."""
        with patch('src.prompt_optimizer_crew.tools.domain_classifier_tool.SentenceTransformer'):
            tool = DomainClassifierTool()
            tool.model = Mock()
            return tool
    
    def test_initialization(self, classifier):
        """Test tool initialization."""
        assert classifier.name == "Domain Classifier"
        assert classifier.model is not None
    
    def test_classify_with_model(self, classifier):
        """Test classification with a working model."""
        # Mock the model's encode method
        mock_embedding = Mock()
        classifier.model.encode.return_value = mock_embedding
        
        # Mock similarity calculation
        with patch('src.prompt_optimizer_crew.tools.domain_classifier_tool.util.cos_sim') as mock_sim:
            mock_sim.return_value = Mock(max=Mock(return_value=0.8))
            
            result = classifier.classify("Write a Python function")
            assert result in classifier.DOMAIN_EXAMPLES.keys()
    
    def test_classify_without_model(self):
        """Test classification fallback when model is not available."""
        with patch('src.prompt_optimizer_crew.tools.domain_classifier_tool.SentenceTransformer', side_effect=Exception):
            tool = DomainClassifierTool()
            result = tool.classify("Test prompt")
            assert result == "general"
    
    def test_get_domain_info(self, classifier):
        """Test getting domain information."""
        info = classifier.get_domain_info("software_development")
        assert info["domain"] == "software_development"
        assert "examples" in info
        assert isinstance(info["examples"], list)


class TestInternetSearchTool:
    """Test cases for InternetSearchTool."""
    
    @pytest.fixture
    def search_tool(self):
        """Create a search tool instance for testing."""
        return InternetSearchTool(max_results=3)
    
    def test_initialization(self, search_tool):
        """Test tool initialization."""
        assert search_tool.name == "Internet Search"
        assert search_tool.max_results == 3
    
    def test_empty_query(self, search_tool):
        """Test search with empty query."""
        result = search_tool.search("")
        assert "Error" in result
    
    @patch('src.prompt_optimizer_crew.tools.internet_search_tool.DDGS')
    def test_successful_search(self, mock_ddgs, search_tool):
        """Test successful search execution."""
        # Mock search results
        mock_results = [
            {"title": "Result 1", "body": "Description 1", "href": "http://example.com/1"},
            {"title": "Result 2", "body": "Description 2", "href": "http://example.com/2"}
        ]
        
        mock_ddgs_instance = Mock()
        mock_ddgs_instance.__enter__ = Mock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = Mock(return_value=None)
        mock_ddgs_instance.text.return_value = mock_results
        mock_ddgs.return_value = mock_ddgs_instance
        
        result = search_tool.search("test query")
        
        assert "Result 1" in result
        assert "Result 2" in result
        assert "http://example.com/1" in result
    
    @patch('src.prompt_optimizer_crew.tools.internet_search_tool.DDGS')
    def test_search_error_handling(self, mock_ddgs, search_tool):
        """Test error handling during search."""
        mock_ddgs.side_effect = Exception("Network error")
        
        result = search_tool.search("test query")
        assert "Error" in result
    
    @patch('src.prompt_optimizer_crew.tools.internet_search_tool.DDGS')
    def test_news_search(self, mock_ddgs, search_tool):
        """Test news search functionality."""
        mock_results = [
            {
                "title": "News 1",
                "body": "News description",
                "url": "http://news.com/1",
                "date": "2024-01-01"
            }
        ]
        
        mock_ddgs_instance = Mock()
        mock_ddgs_instance.__enter__ = Mock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = Mock(return_value=None)
        mock_ddgs_instance.news.return_value = mock_results
        mock_ddgs.return_value = mock_ddgs_instance
        
        result = search_tool.search_news("test news")
        
        assert "News 1" in result
        assert "2024-01-01" in result
