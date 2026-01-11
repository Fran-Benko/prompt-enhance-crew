# Prompt Optimizer Crew

## 📖 Description / Descripción

### English

A sophisticated multi-agent system for analyzing and optimizing prompts using CrewAI with local open-source LLMs via Ollama. This system employs a debate-based approach where specialized AI agents evaluate prompts from different perspectives (security, quality, usability, structure, and role definition) to produce significantly improved, robust, clear, and safe prompts. The system uses semantic similarity for domain classification and dynamically selects relevant agents based on the prompt's domain.

**Key Features:**
- Multi-agent collaboration with 6 specialized agents
- Automatic domain detection using semantic similarity
- Dynamic agent selection based on detected domain
- Local LLM support via Ollama (privacy-focused)
- Internet search capabilities with DuckDuckGo
- Extensible YAML-based configuration
- Comprehensive error handling and logging

### Español

Un sofisticado sistema multi-agente para analizar y optimizar prompts utilizando CrewAI con LLMs de código abierto locales a través de Ollama. Este sistema emplea un enfoque basado en debate donde agentes de IA especializados evalúan prompts desde diferentes perspectivas (seguridad, calidad, usabilidad, estructura y definición de roles) para producir prompts significativamente mejorados, robustos, claros y seguros. El sistema utiliza similitud semántica para la clasificación de dominios y selecciona dinámicamente agentes relevantes según el dominio del prompt.

**Características Principales:**
- Colaboración multi-agente con 6 agentes especializados
- Detección automática de dominio usando similitud semántica
- Selección dinámica de agentes basada en el dominio detectado
- Soporte para LLMs locales vía Ollama (enfocado en privacidad)
- Capacidades de búsqueda en internet con DuckDuckGo
- Configuración extensible basada en YAML
- Manejo integral de errores y registro de logs

## 🎯 Overview

This system employs a debate-based approach where specialized AI agents evaluate prompts from different perspectives (security, quality, usability, structure, and role definition) to produce significantly improved, robust, clear, and safe prompts. The system uses semantic similarity for domain classification and dynamically selects relevant agents based on the prompt's domain.

## 🏗️ Architecture

### Agent Roles

1. **Security Agent**: Evaluates vulnerabilities, biases, ethical concerns, and potential misuse scenarios
2. **Quality Agent**: Assesses completeness, clarity, precision, and identifies ambiguities
3. **Usability Agent**: Focuses on ease of understanding and intuitiveness for both LLMs and humans
4. **Structure Agent**: Analyzes logical organization, formatting, and information hierarchy
5. **Role Definition Agent**: Verifies role clarity, perspective definition, and tone consistency
6. **Coordinator Agent**: Synthesizes feedback from all evaluators and generates the final optimized prompt

### Workflow

```
Input Prompt → Domain Classification (Semantic Similarity) → 
Agent Selection (Domain-Based) → Parallel Evaluation → 
Feedback Collection → Synthesis → Optimized Prompt Output
```

### Technology Stack

- **Framework**: CrewAI 0.80.0 with litellm integration
- **LLM Provider**: Ollama (local models)
- **Default Model**: llama3.2:3b
- **Domain Classification**: sentence-transformers (all-MiniLM-L6-v2)
- **Internet Search**: DuckDuckGo Search with retry logic
- **Configuration**: YAML-based agent and task definitions

## 🚀 Features

- **Multi-Agent Collaboration**: 6 specialized agents working in parallel
- **Domain-Aware**: Automatic domain detection using semantic similarity
- **Dynamic Agent Selection**: Selects relevant agents based on detected domain
- **Local LLM Support**: Uses Ollama for running local models
- **Internet Access**: Optional DuckDuckGo search with rate limit handling
- **Extensible**: YAML-based configuration for easy customization
- **Error Resilient**: Comprehensive error handling and fallback mechanisms
- **Observable**: Built-in logging to file and console

## 📋 Requirements

### Hardware
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB+ for model cache
- **Internet**: Required for initial model download and optional search

### Software
- Python 3.11-3.13
- Poetry (package manager)
- Ollama (for local LLM serving)

## 🛠️ Installation

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai/)

### 2. Pull the Required Model

```bash
ollama pull llama3.2:3b
```

### 3. Clone the Repository

```bash
git clone <repository-url>
cd prompt_optimizer_crew
```

### 4. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or on Windows (PowerShell):
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### 5. Install Dependencies

```bash
poetry install
```

### 6. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration (default values work out of the box)
```

### 7. Verify Installation

```bash
poetry run python check_cuda.py  # Optional: Check GPU availability
```

## 🎮 Usage

### Command Line Interface

**Basic usage:**
```bash
poetry run optimize-prompt "Write a function to calculate fibonacci numbers"
```

**Specify domain:**
```bash
poetry run optimize-prompt "Explain quantum computing" --domain scientific_research
```

**Select specific agents:**
```bash
poetry run optimize-prompt "Create a marketing campaign" --agents quality usability role_definition
```

**Save output to file:**
```bash
poetry run optimize-prompt "Your prompt here" --output results.txt
```

**Disable internet search:**
```bash
poetry run optimize-prompt "Your prompt here" --no-internet
```

**Verbose mode:**
```bash
poetry run optimize-prompt "Your prompt here" --verbose
```

### Python API

**Basic usage:**
```python
from prompt_optimizer_crew.crew import PromptOptimizerCrew

# Initialize the crew
crew = PromptOptimizerCrew()

# Optimize a prompt
result = crew.optimize_prompt(
    prompt="Write a function to calculate fibonacci numbers"
)

print(result.optimized_prompt)
print(f"Domain: {result.domain}")
print(f"Agents used: {', '.join(result.agents_used)}")
print(f"Execution time: {result.execution_time:.2f}s")
```

**Advanced usage:**
```python
from prompt_optimizer_crew.crew import PromptOptimizerCrew

# Custom configuration
crew = PromptOptimizerCrew(
    enable_internet_search=True,
    max_iterations=5
)

# Optimize with specific domain and agents
result = crew.optimize_prompt(
    prompt="Your prompt here",
    domain="software_development",
    selected_agents=["security", "quality", "structure"]
)

# Handle errors
if result.errors:
    print("Warnings/Errors:")
    for error in result.errors:
        print(f"  - {error}")
```

## 📁 Project Structure

```
prompt_optimizer_crew/
├── .env                          # Environment configuration
├── .gitignore                    # Git ignore rules
├── pyproject.toml                # Poetry dependencies and config
├── README.md                     # This file
├── check_cuda.py                 # GPU availability checker
├── logs/                         # Log files
│   └── crew.log
├── models/                       # Model cache directory
│   └── cache/
└── src/
    └── prompt_optimizer_crew/
        ├── __init__.py           # Package initialization
        ├── main.py               # CLI entry point
        ├── crew.py               # Main crew orchestration
        ├── config/               # Configuration files
        │   ├── agents.yaml       # Agent definitions
        │   ├── tasks.yaml        # Task definitions
        │   └── domain_agent_mapping.yaml  # Domain-agent mappings
        ├── tools/                # Custom tools
        │   ├── __init__.py
        │   ├── domain_classifier_tool.py  # Semantic domain classifier
        │   └── internet_search_tool.py    # DuckDuckGo search
        ├── utils/                # Utilities
        │   ├── __init__.py
        │   ├── llm_loader.py     # LLM loading with Ollama
        │   └── domain_validator.py
        ├── models/               # Model management
        │   └── __init__.py
        ├── agents/               # Agent implementations
        │   └── __init__.py
        └── flows/                # CrewAI Flows
            └── __init__.py
tests/                            # Test suite
├── __init__.py
├── test_crew.py
├── test_tools.py
└── test_utils.py
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# LLM Provider Configuration
LLM_PROVIDER=ollama                    # Provider: ollama, openai
OLLAMA_BASE_URL=http://localhost:11434 # Ollama server URL
OLLAMA_MODEL=llama3.2:3b               # Model to use

# Model Configuration
DEFAULT_MAX_TOKENS=512                 # Max tokens per generation
DEFAULT_TEMPERATURE=0.7                # Sampling temperature

# Agent Configuration
ENABLE_INTERNET_SEARCH=true            # Enable/disable internet search
MAX_ITERATIONS=3                       # Max iterations per agent task

# Logging
LOG_LEVEL=INFO                         # Logging level
LOG_FILE=./logs/crew.log               # Log file path

# CrewAI
CREWAI_TELEMETRY_OPT_OUT=true         # Opt out of telemetry
```

### Agent Configuration (agents.yaml)

Customize agent behaviors, goals, backstories, and LLM assignments. Each agent can have:
- `role`: Agent's role description
- `goal`: What the agent aims to achieve
- `backstory`: Agent's expertise and background
- `llm`: Specific model to use (optional)
- `allow_internet_search`: Enable internet search for this agent
- `allow_delegation`: Allow agent to delegate tasks

### Task Configuration (tasks.yaml)

Define evaluation tasks for each agent with:
- `description`: Task instructions with placeholders
- `expected_output`: What the agent should produce

### Domain Mapping (domain_agent_mapping.yaml)

Configure which agents are relevant for each domain:
- Domain definitions with descriptions
- Agent lists per domain
- Priority levels
- Domain detection keywords (fallback)

## 🧪 Testing

Run the test suite:

```bash
poetry run pytest
```

With coverage report:

```bash
poetry run pytest --cov=src/prompt_optimizer_crew --cov-report=html
```

Run specific test file:

```bash
poetry run pytest tests/test_crew.py -v
```

## 🎯 Supported Domains

The system automatically detects and optimizes prompts for:

- **Software Development**: Code, algorithms, APIs, debugging
- **Scientific Research**: Research papers, experiments, data analysis
- **Marketing**: Campaigns, content creation, brand strategy
- **Data Analysis**: Data processing, visualization, statistics
- **Education**: Lesson plans, educational content, tutorials
- **General**: Miscellaneous tasks and queries

Domain detection uses semantic similarity with pre-computed embeddings for accurate classification.

## 🔍 How It Works

1. **Domain Classification**: Uses sentence-transformers to classify the prompt into a domain based on semantic similarity with example prompts
2. **Agent Selection**: Selects relevant agents based on the detected domain (configurable in domain_agent_mapping.yaml)
3. **Parallel Evaluation**: Each selected agent evaluates the prompt from their specialized perspective
4. **Synthesis**: The coordinator agent collects all feedback and generates the optimized prompt
5. **Error Handling**: Comprehensive error handling ensures partial results even if some agents fail

## 🛡️ Error Handling

The system includes robust error handling:
- **Rate Limiting**: Automatic retry with exponential backoff for internet searches
- **Agent Failures**: Continues with available agents if some fail
- **Partial Results**: Returns best-effort results even with errors
- **Detailed Logging**: All errors logged to file and console

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure tests pass (`poetry run pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 🙏 Acknowledgments

- Built with [CrewAI](https://www.crewai.com/) - Multi-agent orchestration framework
- Powered by [Ollama](https://ollama.ai/) - Local LLM serving
- Uses [sentence-transformers](https://www.sbert.net/) - Semantic similarity
- Search via [DuckDuckGo](https://duckduckgo.com/) - Privacy-focused search

## 📧 Contact

fran.benko@gmail.com

## 🔮 Roadmap

- [ ] Support for additional LLM providers (OpenAI, Anthropic, etc.)
- [ ] Web UI interface for easier interaction
- [ ] Prompt versioning and history tracking
- [ ] A/B testing framework for prompt comparison
- [ ] Multi-language support for international prompts
- [ ] Custom agent creation wizard
- [ ] Batch processing for multiple prompts
- [ ] Integration with popular prompt libraries
- [ ] Performance metrics and analytics dashboard
- [ ] Export optimized prompts in various formats

## 📊 Performance

Typical execution times (on recommended hardware):
- Domain classification: <1s
- Agent evaluation: 5-15s per agent
- Total optimization: 30-90s depending on complexity and number of agents

## 🐛 Troubleshooting

**Ollama connection error:**
```bash
# Ensure Ollama is running
ollama serve
```

**Model not found:**
```bash
# Pull the required model
ollama pull llama3.2:3b
```

**Memory issues:**
- Reduce `MAX_ITERATIONS` in .env
- Use a smaller model (e.g., `llama3.2:1b`)
- Disable internet search with `--no-internet`

**Import errors:**
```bash
# Reinstall dependencies
poetry install --no-cache
```

---

Made with ❤️ using CrewAI and Ollama
