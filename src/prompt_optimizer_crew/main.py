"""
Main entry point for the Prompt Optimizer Crew CLI.
"""

import argparse
import asyncio
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .crew import PromptOptimizerCrew

# Load environment variables
load_dotenv()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="crewai_tools")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="litellm")

# Suppress asyncio event loop warnings
if sys.platform == "win32":
    # Set event loop policy for Windows to avoid warnings
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/crew.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize prompts using a multi-agent AI system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  optimize-prompt "Write a function to sort a list"
  optimize-prompt "Explain quantum computing" --domain scientific_research
  optimize-prompt "Create a marketing campaign" --agents quality usability
        """
    )
    
    parser.add_argument(
        "prompt",
        type=str,
        help="The prompt to optimize"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Specify the domain (auto-detected if not provided)"
    )
    
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help="Specific agents to use (e.g., security quality usability)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (prints to stdout if not provided)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--no-internet",
        action="store_true",
        help="Disable internet search for agents"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit", "none"],
        default=None,
        help="Override quantization setting"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI."""
    try:
        args = parse_arguments()
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose mode enabled")
        
        logger.info("Initializing Prompt Optimizer Crew...")
        
        # Initialize the crew with custom settings
        crew = PromptOptimizerCrew(
            use_quantization=args.quantization,
            enable_internet_search=not args.no_internet
        )
        
        logger.info(f"Optimizing prompt: {args.prompt[:50]}...")
        
        # Optimize the prompt
        result = crew.optimize_prompt(
            prompt=args.prompt,
            domain=args.domain,
            selected_agents=args.agents
        )
        
        # Output the result
        error_section = ""
        if result.errors:
            error_section = f"""
{'='*80}
WARNINGS/ERRORS:
{'='*80}
{chr(10).join(f"- {error}" for error in result.errors)}

"""
        
        output_text = f"""
{'='*80}
ORIGINAL PROMPT:
{'='*80}
{args.prompt}

{'='*80}
OPTIMIZED PROMPT:
{'='*80}
{result.optimized_prompt}

{'='*80}
METADATA:
{'='*80}
Domain: {result.domain}
Agents Used: {', '.join(result.agents_used)}
Execution Time: {result.execution_time:.2f}s
{'='*80}
{error_section}"""
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text)
            logger.info(f"Results saved to {args.output}")
        else:
            print(output_text)
        
        if result.errors:
            logger.warning(f"Optimization completed with {len(result.errors)} warning(s)")
        else:
            logger.info("Optimization completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
