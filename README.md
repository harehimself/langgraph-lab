# LangGraph Lab

A comprehensive LangGraph quickstart application demonstrating AI agent workflows with proper architecture and best practices. This project serves as a solid, well-architected foundation that developers can easily clone and extend with custom agentic application features.

## Features

- **Complete LangGraph Workflow Implementation**: Demonstrates stateful, multi-actor applications with conditional routing
- **Research Agent System**: AI agents capable of web search, analysis, and synthesis
- **Modular Architecture**: Clean separation of concerns with agents, tools, and workflows
- **Multiple Search Providers**: Support for DuckDuckGo (free), Serper, and Tavily APIs
- **Text Processing Tools**: Advanced text analysis, summarization, and extraction capabilities
- **Production-Ready Configuration**: Environment management, logging, and error handling
- **Interactive CLI**: Multiple ways to interact with the system
- **Comprehensive Examples**: Detailed usage examples and demonstrations

## Architecture

```
langgraph-lab/
├── src/
│   ├── agents/          # AI agent implementations
│   ├── workflows/       # LangGraph workflow definitions
│   ├── tools/           # Custom tools (search, text processing)
│   ├── config/          # Configuration management
│   └── utils/           # Utilities and logging
├── examples/            # Usage examples
├── tests/              # Test suite
└── main.py             # CLI entry point
```

### Core Components

- **Research Workflow**: LangGraph-based workflow with conditional routing and state management
- **Research Agent**: Specialized agent for conducting research tasks
- **Web Search Tool**: Multi-provider web search with fallback strategies
- **Text Processor**: Advanced text analysis and summarization
- **Configuration System**: Centralized settings with environment variable support

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- Optional: Serper or Tavily API keys for enhanced search

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/harehimself/langgraph-lab.git
   cd langgraph-lab
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the demo**:
   ```bash
   python main.py demo
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# Optional - Enhanced Search
SERPER_API_KEY=your_serper_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Optional - LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=langgraph-lab
```

### Configuration Options

- `APP_NAME`: Application name (default: "LangGraph Lab")
- `LOG_LEVEL`: Logging level (default: "INFO")
- `MAX_ITERATIONS`: Maximum workflow iterations (default: 10)
- `TIMEOUT_SECONDS`: Workflow timeout (default: 300)

## Usage

### Command Line Interface

#### Run Demo
```bash
python main.py demo
```

#### Conduct Research
```bash
python main.py research "What are the latest developments in AI agents?"
```

#### Interactive Session
```bash
python main.py interactive
```

#### Show Configuration
```bash
python main.py info
```

### Programmatic Usage

```python
import asyncio
from src.workflows.research_workflow import research_topic

async def main():
    # Conduct research
    results = await research_topic(
        "How to build AI agents with LangGraph",
        research_depth="standard"
    )
    
    print(f"Analysis: {results['analysis']}")
    print(f"Sources: {results['sources']}")
    print(f"Key Insights: {results['key_insights']}")

asyncio.run(main())
```

### Custom Agent Development

```python
from src.agents.base_agent import ToolAgent

class CustomAgent(ToolAgent):
    def _get_default_system_prompt(self) -> str:
        return "You are a specialized AI agent for..."
    
    async def process(self, state):
        # Implement custom logic
        return updated_state
```

## Core Workflows

### Research Workflow

The main workflow demonstrates:

1. **Query Analysis**: Understanding research requirements
2. **Conditional Routing**: Different paths based on complexity
3. **Agent Coordination**: Multiple agents working together
4. **State Management**: Maintaining context across steps
5. **Result Validation**: Quality checks and follow-ups

### Workflow States

- `basic`: Quick research with single search
- `standard`: Comprehensive research with analysis
- `deep`: Multi-iteration research with synthesis

## Tools and Integrations

### Web Search Tool

- **DuckDuckGo**: Free web search (no API key required)
- **Serper**: Google search API with high-quality results
- **Tavily**: AI-optimized search with content processing

### Text Processing Tool

- **Summarization**: Intelligent text summarization
- **Key Point Extraction**: Automatic insight identification
- **Sentiment Analysis**: Emotion and tone detection
- **Entity Extraction**: Named entity recognition

## Examples

### Basic Research
```python
# Simple research query
results = await research_topic(
    "Benefits of using LangGraph for AI applications",
    "basic"
)
```

### Deep Research
```python
# Complex multi-faceted research
results = await research_topic(
    "Compare different AI agent frameworks for enterprise use",
    "deep"
)
```

### Custom Tools
```python
from src.tools.web_search import WebSearchTool

search_tool = WebSearchTool()
results = await search_tool.search("LangGraph tutorials", max_results=5)
```

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Development

### Code Quality

```bash
# Format code
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Type checking
mypy src/

# Linting
flake8 src/ tests/ examples/
```

### Adding New Agents

1. Create agent class in `src/agents/`
2. Inherit from `BaseAgent` or `ToolAgent`
3. Implement required methods
4. Add to workflow in `src/workflows/`

### Adding New Tools

1. Create tool class in `src/tools/`
2. Implement async methods
3. Add to agent tool registry
4. Update documentation

## Deployment

### Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py", "demo"]
```

### Environment Setup

```bash
# Production deployment
export OPENAI_API_KEY="your_key"
export LOG_LEVEL="WARNING"
export DEBUG="false"

python main.py research "your query"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure code quality checks pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for the amazing workflow framework
- [LangChain](https://github.com/langchain-ai/langchain) for the foundation
- [OpenAI](https://openai.com/) for the language models

## Support

- **Issues**: [GitHub Issues](https://github.com/harehimself/langgraph-lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/harehimself/langgraph-lab/discussions)
- **Documentation**: Check the `examples/` directory for detailed usage examples

## Roadmap

- [ ] Add more specialized agents (analysis, writing, coding)
- [ ] Implement persistent state storage
- [ ] Add web interface with Streamlit/Gradio
- [ ] Enhanced tool integrations (databases, APIs)
- [ ] Multi-modal capabilities (images, documents)
- [ ] Distributed workflow execution

---

**Built with ❤️ by [Mike Hare](https://github.com/harehimself)**