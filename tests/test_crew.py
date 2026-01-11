"""
Unit tests for the main Crew orchestration.
"""

import pytest
from unittest.mock import Mock, patch

from src.prompt_optimizer_crew.crew import PromptOptimizerCrew, OptimizationResult


class TestPromptOptimizerCrew:
    """Test cases for PromptOptimizerCrew class."""
    
    @pytest.fixture
    def crew(self):
        """Create a crew instance for testing."""
        with patch('src.prompt_optimizer_crew.crew.LLMLoader'):
            with patch('src.prompt_optimizer_crew.crew.DomainClassifierTool'):
                with patch('src.prompt_optimizer_crew.crew.InternetSearchTool'):
                    return PromptOptimizerCrew()
    
    def test_initialization(self, crew):
        """Test crew initialization."""
        assert crew is not None
        assert hasattr(crew, 'agents_config')
        assert hasattr(crew, 'tasks_config')
        assert hasattr(crew, 'domain_mapping')
    
    def test_load_yaml_success(self, crew, tmp_path):
        """Test successful YAML loading."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\n")
        
        result = crew._load_yaml(yaml_file)
        assert result == {"key": "value"}
    
    def test_load_yaml_not_found(self, crew, tmp_path):
        """Test YAML loading with missing file."""
        result = crew._load_yaml(tmp_path / "nonexistent.yaml")
        assert result == {}
    
    def test_select_relevant_agents(self, crew):
        """Test agent selection based on domain."""
        crew.domain_mapping = {
            "domains": {
                "software_development": {
                    "agents": ["security", "quality"]
                }
            }
        }
        
        agents = crew._select_relevant_agents("software_development")
        assert "security" in agents
        assert "quality" in agents
        assert "coordinator" in agents
    
    @patch('src.prompt_optimizer_crew.crew.Crew')
    def test_optimize_prompt_basic(self, mock_crew_class, crew):
        """Test basic prompt optimization."""
        # Mock the crew execution
        mock_crew_instance = Mock()
        mock_crew_instance.kickoff.return_value = "Optimized prompt"
        mock_crew_class.return_value = mock_crew_instance
        
        # Mock domain classifier
        crew.domain_classifier.classify = Mock(return_value="general")
        
        result = crew.optimize_prompt("Test prompt")
        
        assert isinstance(result, OptimizationResult)
        assert result.optimized_prompt == "Optimized prompt"
        assert result.domain == "general"
    
    def test_optimization_result_model(self):
        """Test OptimizationResult model."""
        result = OptimizationResult(
            optimized_prompt="Test",
            domain="general",
            agents_used=["security", "quality"],
            execution_time=1.5
        )
        
        assert result.optimized_prompt == "Test"
        assert result.domain == "general"
        assert len(result.agents_used) == 2
        assert result.execution_time == 1.5
