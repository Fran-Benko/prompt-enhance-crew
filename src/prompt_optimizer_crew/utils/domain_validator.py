"""
Domain Validator utility for validating and managing domain classifications.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DomainValidator:
    """
    Utility for validating domain classifications and managing domain metadata.
    """
    
    VALID_DOMAINS = [
        "software_development",
        "scientific_research",
        "marketing",
        "data_analysis",
        "education",
        "general"
    ]
    
    DOMAIN_METADATA = {
        "software_development": {
            "name": "Software Development",
            "description": "Programming, coding, software engineering tasks",
            "keywords": ["code", "function", "algorithm", "API", "debug"],
            "priority": "high"
        },
        "scientific_research": {
            "name": "Scientific Research",
            "description": "Research, academic, scientific analysis",
            "keywords": ["research", "study", "hypothesis", "experiment"],
            "priority": "high"
        },
        "marketing": {
            "name": "Marketing",
            "description": "Marketing, advertising, content creation",
            "keywords": ["campaign", "brand", "advertising", "content"],
            "priority": "medium"
        },
        "data_analysis": {
            "name": "Data Analysis",
            "description": "Data analysis, statistics, visualization",
            "keywords": ["data", "analysis", "visualization", "statistics"],
            "priority": "high"
        },
        "education": {
            "name": "Education",
            "description": "Educational content, teaching, learning",
            "keywords": ["teach", "learn", "student", "education"],
            "priority": "medium"
        },
        "general": {
            "name": "General",
            "description": "General purpose tasks",
            "keywords": ["help", "explain", "understand", "information"],
            "priority": "medium"
        }
    }
    
    def __init__(self):
        """Initialize the domain validator."""
        logger.info("Domain Validator initialized")
    
    def is_valid_domain(self, domain: str) -> bool:
        """
        Check if a domain is valid.
        
        Args:
            domain: Domain name to validate
            
        Returns:
            True if valid, False otherwise
        """
        return domain in self.VALID_DOMAINS
    
    def validate_domain(self, domain: str) -> str:
        """
        Validate and normalize a domain name.
        
        Args:
            domain: Domain name to validate
            
        Returns:
            Validated domain name (defaults to 'general' if invalid)
        """
        if not domain:
            logger.warning("Empty domain provided, defaulting to 'general'")
            return "general"
        
        domain = domain.lower().strip()
        
        if self.is_valid_domain(domain):
            return domain
        
        # Try to find a close match
        for valid_domain in self.VALID_DOMAINS:
            if domain in valid_domain or valid_domain in domain:
                logger.info(f"Matched '{domain}' to '{valid_domain}'")
                return valid_domain
        
        logger.warning(f"Invalid domain '{domain}', defaulting to 'general'")
        return "general"
    
    def get_domain_metadata(self, domain: str) -> Dict:
        """
        Get metadata for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Dictionary with domain metadata
        """
        domain = self.validate_domain(domain)
        return self.DOMAIN_METADATA.get(domain, self.DOMAIN_METADATA["general"])
    
    def get_domain_name(self, domain: str) -> str:
        """
        Get the display name for a domain.
        
        Args:
            domain: Domain identifier
            
        Returns:
            Human-readable domain name
        """
        metadata = self.get_domain_metadata(domain)
        return metadata.get("name", domain)
    
    def get_domain_keywords(self, domain: str) -> List[str]:
        """
        Get keywords associated with a domain.
        
        Args:
            domain: Domain identifier
            
        Returns:
            List of keywords
        """
        metadata = self.get_domain_metadata(domain)
        return metadata.get("keywords", [])
    
    def suggest_domain(self, text: str) -> Optional[str]:
        """
        Suggest a domain based on text content using keyword matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Suggested domain or None
        """
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, metadata in self.DOMAIN_METADATA.items():
            keywords = metadata.get("keywords", [])
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            logger.info(f"Suggested domain: {best_domain} (score: {domain_scores[best_domain]})")
            return best_domain
        
        return None
    
    def list_domains(self) -> List[Dict]:
        """
        List all available domains with metadata.
        
        Returns:
            List of domain information dictionaries
        """
        return [
            {
                "id": domain,
                **self.DOMAIN_METADATA[domain]
            }
            for domain in self.VALID_DOMAINS
        ]
