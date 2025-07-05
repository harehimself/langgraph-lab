"""
Basic research example for LangGraph Lab.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflows.research_workflow import research_topic
from src.agents.research_agent import ResearchAgent
from src.tools.web_search import WebSearchTool
from src.tools.text_processor import TextProcessorTool


async def basic_workflow_example():
    """Demonstrate basic workflow usage."""
    print("🔬 Basic Research Workflow Example")
    print("=" * 50)
    
    # Example 1: Simple research query
    query1 = "What is LangGraph and how is it used in AI applications?"
    print(f"\n📝 Query: {query1}")
    
    try:
        results = await research_topic(query1, "basic")
        
        print(f"✅ Research completed!")
        print(f"📊 Found {len(results.get('sources', []))} sources")
        print(f"🔍 Analysis length: {len(results.get('analysis', ''))}")
        print(f"💡 Key insights: {len(results.get('key_insights', []))}")
        
        if results.get('analysis'):
            print(f"\n📋 Analysis Preview:")
            print(results['analysis'][:200] + "..." if len(results['analysis']) > 200 else results['analysis'])
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def agent_usage_example():
    """Demonstrate direct agent usage."""
    print("\n🤖 Direct Agent Usage Example")
    print("=" * 50)
    
    # Create a research agent
    agent = ResearchAgent()
    
    # Example state
    state = {
        "query": "Benefits of using graph-based AI workflows",
        "research_depth": "standard"
    }
    
    try:
        # Process with the agent
        result = await agent.process(state)
        
        print(f"✅ Agent processing completed!")
        print(f"📊 Research completed: {result.get('research_completed', False)}")
        
        if result.get('key_insights'):
            print(f"\n💡 Key Insights:")
            for i, insight in enumerate(result['key_insights'][:3], 1):
                print(f"  {i}. {insight}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


async def tools_usage_example():
    """Demonstrate individual tool usage."""
    print("\n🛠️ Individual Tools Usage Example")
    print("=" * 50)
    
    # Web Search Tool
    print("\n🌐 Web Search Tool:")
    search_tool = WebSearchTool()
    
    try:
        search_results = await search_tool.search("LangGraph tutorial", max_results=3)
        print(f"✅ Found {len(search_results)} search results")
        
        for i, result in enumerate(search_results[:2], 1):
            print(f"  {i}. {result.get('title', 'No title')}")
    
    except Exception as e:
        print(f"❌ Search error: {e}")
    
    # Text Processor Tool
    print("\n📝 Text Processor Tool:")
    text_tool = TextProcessorTool()
    
    sample_text = """
    LangGraph is a library for building stateful, multi-actor applications with LLMs. 
    It extends LangChain by adding support for creating complex workflows with 
    multiple agents, state management, and conditional routing. This makes it 
    particularly useful for building sophisticated AI applications that require 
    coordination between different components.
    """
    
    try:
        # Summarize text
        summary = await text_tool.summarize(sample_text, max_length=100)
        print(f"✅ Summary: {summary}")
        
        # Extract key points
        key_points = await text_tool.extract_key_points(sample_text, max_points=3)
        print(f"✅ Key points: {len(key_points)}")
        for i, point in enumerate(key_points, 1):
            print(f"  {i}. {point}")
    
    except Exception as e:
        print(f"❌ Text processing error: {e}")


async def advanced_workflow_example():
    """Demonstrate advanced workflow features."""
    print("\n🚀 Advanced Workflow Example")
    print("=" * 50)
    
    # Complex query requiring deep research
    complex_query = "Compare the advantages and disadvantages of different AI agent frameworks for enterprise applications"
    
    print(f"📝 Complex Query: {complex_query}")
    
    try:
        # Run deep research
        results = await research_topic(complex_query, "deep")
        
        print(f"✅ Deep research completed!")
        print(f"📊 Iteration count: {results.get('iteration_count', 0)}")
        print(f"🔍 Sources found: {len(results.get('sources', []))}")
        
        # Show comprehensive analysis if available
        if results.get('comprehensive_analysis'):
            print(f"\n📋 Comprehensive Analysis Preview:")
            preview = results['comprehensive_analysis'][:300]
            print(preview + "..." if len(results['comprehensive_analysis']) > 300 else preview)
        
        # Show follow-up questions
        if results.get('follow_up_questions'):
            print(f"\n❓ Generated Follow-up Questions:")
            for i, question in enumerate(results['follow_up_questions'], 1):
                print(f"  {i}. {question}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all examples."""
    print("🎯 LangGraph Lab - Usage Examples")
    print("=" * 60)
    
    # Check if environment is set up
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        
        if not settings.openai_api_key:
            print("❌ OpenAI API key not found!")
            print("Please set OPENAI_API_KEY in your .env file")
            return
        
        print(f"✅ Environment configured")
        print(f"🔧 Using model: {settings.openai_model}")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Run examples
    await basic_workflow_example()
    await agent_usage_example()
    await tools_usage_example()
    await advanced_workflow_example()
    
    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("  - Modify the queries to test with your own research topics")
    print("  - Explore the workflow configuration in src/workflows/")
    print("  - Add custom agents in src/agents/")
    print("  - Create new tools in src/tools/")


if __name__ == "__main__":
    asyncio.run(main())