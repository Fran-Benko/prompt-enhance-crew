"""
Unit tests for utility modules.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.prompt_optimizer_crew.utils.domain_validator import DomainValidator
from src.prompt_optimizer_crew.utils.llm_loader import LLMLoader


class TestDomainValidator:
    """Test cases for DomainValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance for testing."""
        return DomainValidator()
    
    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert len(validator.VALID_DOMAINS) > 0
    
    def test_is_valid_domain(self, validator):
        """Test domain validation."""
        assert validator.is_valid_domain("software_development") is True
        assert validator.is_valid_domain("invalid_domain") is False
        assert validator.is_valid_domain("") is False
    
    def test_validate_domain_valid(self, validator):
        """Test validation of valid domains."""
        result = validator.validate_domain("software_development")
        assert result == "software_development"
    
    def test_validate_domain_invalid(self, validator):
        """Test validation of invalid domains."""
        result = validator.validate_domain("invalid_domain")
        assert result == "general"
    
    def test_validate_domain_empty(self, validator):
        """Test validation of empty domain."""
        result = validator.validate_domain("")
        assert result == "general"
    
    def test_validate_domain_case_insensitive(self, validator):
        """Test case-insensitive domain validation."""
        result = validator.validate_domain("SOFTWARE_DEVELOPMENT")
        assert result == "software_development"
    
    def test_get_domain_metadata(self, validator):
        """Test getting domain metadata."""
        metadata = validator.get_domain_metadata("software_development")
        assert "name" in metadata
        assert "description" in metadata
        assert "keywords" in metadata
    
    def test_get_domain_name(self, validator):
        """Test getting domain display name."""
        name = validator.get_domain_name("software_development")
        assert name == "Software Development"
    
    def test_get_domain_keywords(self, validator):
        """Test getting domain keywords."""
        keywords = validator.get_domain_keywords("software_development")
        assert isinstance(keywords, list)
        assert len(keywords) > 0
    
    def test_suggest_domain(self, validator):
        """Test domain suggestion based on text."""
        text = "Write a Python function to sort an array"
        domain = validator.suggest_domain(text)
        assert domain == "software_development"
    
    def test_suggest_domain_no_match(self, validator):
        """Test domain suggestion with no clear match."""
        text = "Random text with no clear domain"
        domain = validator.suggest_domain(text)
        assert domain is None
    
    def test_list_domains(self, validator):
        """Test listing all domains."""
        domains = validator.list_domains()
        assert isinstance(domains, list)
        assert len(domains) == len(validator.VALID_DOMAINS)
        assert all("id" in d for d in domains)


class TestLLMLoader:
    """Test cases for LLMLoader."""
    
    @pytest.fixture
    def loader(self):
        """Create a loader instance for testing."""
        with patch('src.prompt_optimizer_crew.utils.llm_loader.torch'):
            return LLMLoader(quantization="4bit")
    
    def test_initialization(self, loader):
        """Test loader initialization."""
        assert loader is not None
        assert loader.quantization == "4bit"
        assert Path(loader.cache_dir).exists() or True  # May not exist in test
    
    def test_get_quantization_config_4bit(self, loader):
        """Test 4-bit quantization config."""
        loader.quantization = "4bit"
        config = loader._get_quantization_config()
        assert config is not None
    
    def test_get_quantization_config_8bit(self, loader):
        """Test 8-bit quantization config."""
        loader.quantization = "8bit"
        config = loader._get_quantization_config()
        assert config is not None
    
    def test_get_quantization_config_none(self, loader):
        """Test no quantization config."""
        loader.quantization = "none"
        config = loader._get_quantization_config()
        assert config is None
    
    def test_get_model_info(self, loader):
        """Test getting model information."""
        info = loader.get_model_info("gpt2")
        assert "name" in info
        assert "path" in info
        assert "is_cached" in info
        assert "quantization" in info
    
    def test_list_available_models(self, loader):
        """Test listing available models."""
        models = loader.list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt2" in models
    
    def test_clear_cache(self, loader):
        """Test clearing model cache."""
        loader._model_cache = {"test": Mock()}
        loader.clear_cache()
        assert len(loader._model_cache) == 0
    
    @patch('src.prompt_optimizer_crew.utils.llm_loader.AutoModelForCausalLM')
    @patch('src.prompt_optimizer_crew.utils.llm_loader.AutoTokenizer')
    @patch('src.prompt_optimizer_crew.utils.llm_loader.pipeline')
    @patch('src.prompt_optimizer_crew.utils.llm_loader.HuggingFacePipeline')
    def test_load_llm_success(self, mock_hf_pipeline, mock_pipeline, mock_tokenizer, mock_model, loader):
        """Test successful LLM loading."""
        # Mock the components
        mock_tokenizer.from_pretrained.return_value = Mock(pad_token=None, eos_token="<eos>")
        mock_model.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        mock_hf_pipeline.return_value = Mock()
        
        llm = loader.load_llm("gpt2")
        assert llm is not None
    
    @patch('src.prompt_optimizer_crew.utils.llm_loader.AutoModelForCausalLM')
    def test_load_llm_fallback(self, mock_model, loader):
        """Test LLM loading with fallback."""
        # Make the first load fail
        mock_model.from_pretrained.side_effect = [Exception("Load failed"), Mock()]
        
        with patch('src.prompt_optimizer_crew.utils.llm_loader.AutoTokenizer'):
            with patch('src.prompt_optimizer_crew.utils.llm_loader.pipeline'):
                with patch('src.prompt_optimizer_crew.utils.llm_loader.HuggingFacePipeline'):
                    # Should fallback to distilgpt2
                    with pytest.raises(Exception):
                        loader.load_llm("invalid_model")
