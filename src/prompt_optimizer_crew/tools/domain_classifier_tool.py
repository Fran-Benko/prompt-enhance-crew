"""
Domain Classifier Tool for identifying prompt domains.
"""

import logging
from typing import Any, Dict, List, Optional

from crewai_tools import BaseTool
from pydantic import Field, PrivateAttr
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class DomainClassifierTool(BaseTool):
    """
    Tool for classifying prompts into domains using semantic similarity.
    
    This tool uses sentence embeddings to determine which domain a prompt
    belongs to, enabling dynamic agent selection.
    """
    
    name: str = "Domain Classifier"
    description: str = (
        "Classifies a prompt into a specific domain (e.g., software_development, "
        "scientific_research, marketing, etc.) based on semantic similarity."
    )
    
    # Use PrivateAttr for attributes that shouldn't be part of the Pydantic model
    _model: Optional[Any] = PrivateAttr(default=None)
    _domain_embeddings: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    # Domain definitions with example prompts
    DOMAIN_EXAMPLES: Dict[str, List[str]] = {
        "software_development": [
            "Write a function to sort an array",
            "Create a REST API endpoint",
            "Debug this code snippet",
            "Implement a binary search algorithm",
            "Design a database schema"
        ],
        "scientific_research": [
            "Explain quantum entanglement",
            "Analyze this dataset statistically",
            "Write a research paper abstract",
            "Describe the scientific method",
            "Review this hypothesis"
        ],
        "marketing": [
            "Create a social media campaign",
            "Write compelling ad copy",
            "Develop a brand strategy",
            "Analyze market trends",
            "Design a customer persona"
        ],
        "data_analysis": [
            "Analyze this CSV data",
            "Create a data visualization",
            "Perform statistical analysis",
            "Clean and preprocess data",
            "Build a predictive model"
        ],
        "education": [
            "Explain this concept to students",
            "Create a lesson plan",
            "Design educational materials",
            "Develop a curriculum",
            "Write study questions"
        ],
        "general": [
            "Help me with this task",
            "Provide information about",
            "Explain how to",
            "What is the best way to",
            "Can you help me understand"
        ]
    }
    
    def __init__(self, **kwargs):
        """Initialize the domain classifier with a lightweight embedding model."""
        super().__init__(**kwargs)
        try:
            # Use a lightweight model that fits in memory constraints
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Pre-compute embeddings for domain examples
            self._domain_embeddings = {}
            for domain, examples in self.DOMAIN_EXAMPLES.items():
                embeddings = self._model.encode(examples, convert_to_tensor=True)
                self._domain_embeddings[domain] = embeddings
            
            logger.info("Domain Classifier initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Domain Classifier: {e}")
            self._model = None
            self._domain_embeddings = {}
    
    def _run(self, prompt: str) -> str:
        """
        Classify the prompt into a domain.
        
        Args:
            prompt: The prompt to classify
            
        Returns:
            The detected domain name
        """
        return self.classify(prompt)
    
    def classify(self, prompt: str) -> str:
        """
        Classify a prompt into a domain using semantic similarity.
        
        Args:
            prompt: The prompt to classify
            
        Returns:
            The domain with the highest similarity score
        """
        if not self._model or not self._domain_embeddings:
            logger.warning("Domain classifier not properly initialized, returning 'general'")
            return "general"
        
        try:
            # Encode the input prompt
            prompt_embedding = self._model.encode(prompt, convert_to_tensor=True)
            
            # Calculate similarity with each domain
            domain_scores = {}
            for domain, domain_embs in self._domain_embeddings.items():
                # Calculate cosine similarity with all examples in the domain
                similarities = util.cos_sim(prompt_embedding, domain_embs)
                # Use the maximum similarity as the domain score
                domain_scores[domain] = float(similarities.max())
            
            # Select domain with highest score
            best_domain = max(domain_scores, key=domain_scores.get)
            best_score = domain_scores[best_domain]
            
            logger.info(f"Classified prompt as '{best_domain}' (score: {best_score:.3f})")
            logger.debug(f"All scores: {domain_scores}")
            
            # If the best score is too low, default to 'general'
            if best_score < 0.3:
                logger.info("Low confidence, defaulting to 'general' domain")
                return "general"
            
            return best_domain
            
        except Exception as e:
            logger.error(f"Error during classification: {e}", exc_info=True)
            return "general"
    
    def get_domain_info(self, domain: str) -> Dict[str, Any]:
        """
        Get information about a specific domain.
        
        Args:
            domain: The domain name
            
        Returns:
            Dictionary with domain information
        """
        return {
            "domain": domain,
            "examples": self.DOMAIN_EXAMPLES.get(domain, []),
            "has_embeddings": domain in self._domain_embeddings
        }
