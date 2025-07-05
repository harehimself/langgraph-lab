"""
Main entry point for LangGraph Lab.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.workflows.research_workflow import research_topic
from src.config.settings import get_settings
from src.utils.logging import setup_logging


console = Console()


async def run_research_demo():
    """Run a demonstration of the research workflow."""
    console.print(Panel.fit("🔬 LangGraph Lab - Research Workflow Demo", style="bold blue"))
    
    # Sample research queries
    demo_queries = [
        {
            "query": "What are the latest developments in artificial intelligence for 2024?",
            "depth": "standard"
        },
        {
            "query": "How does machine learning impact software development?",
            "depth": "basic"
        },
        {
            "query": "Compare different approaches to building AI agents",
            "depth": "deep"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        console.print(f"\n[bold cyan]Demo {i}: {demo['query']}[/bold cyan]")
        console.print(f"Research Depth: {demo['depth']}")
        
        with console.status(f"[bold green]Researching..."):
            try:
                results = await research_topic(demo["query"], demo["depth"])
                
                # Display results
                display_research_results(results)
                
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
        
        if i < len(demo_queries):
            console.print("\n" + "="*50)


def display_research_results(results: Dict[str, Any]):
    """Display research results in a formatted way."""
    if results.get("error"):
        console.print(f"[bold red]Error:[/bold red] {results['error']}")
        return
    
    # Create results table
    table = Table(title="Research Results", show_header=True, header_style="bold magenta")
    table.add_column("Attribute", style="cyan", width=20)
    table.add_column("Value", style="white")
    
    # Add basic info
    table.add_row("Query", results.get("query", "N/A"))
    table.add_row("Research Depth", results.get("research_depth", "N/A"))
    table.add_row("Sources Found", str(len(results.get("sources", []))))
    table.add_row("Iterations", str(results.get("iteration_count", 0)))
    
    console.print(table)
    
    # Display analysis
    if results.get("analysis"):
        console.print("\n[bold yellow]Analysis:[/bold yellow]")
        analysis_panel = Panel(
            results["analysis"],
            title="Research Analysis",
            border_style="yellow"
        )
        console.print(analysis_panel)
    
    # Display key insights
    if results.get("key_insights"):
        console.print("\n[bold green]Key Insights:[/bold green]")
        for i, insight in enumerate(results["key_insights"], 1):
            console.print(f"{i}. {insight}")
    
    # Display sources
    if results.get("sources"):
        console.print("\n[bold blue]Sources:[/bold blue]")
        for i, source in enumerate(results["sources"], 1):
            console.print(f"{i}. {source}")
    
    # Display follow-up questions
    if results.get("follow_up_questions"):
        console.print("\n[bold purple]Follow-up Questions:[/bold purple]")
        for i, question in enumerate(results["follow_up_questions"], 1):
            console.print(f"{i}. {question}")


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--log-file', help='Log file path')
def cli(debug, log_file):
    """LangGraph Lab - A comprehensive LangGraph quickstart application."""
    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(log_level=log_level, log_file=log_file)
    
    # Check environment setup
    settings = get_settings()
    if not settings.openai_api_key:
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not found in environment")
        console.print("Please create a .env file based on .env.example")
        sys.exit(1)


@cli.command()
def demo():
    """Run the research workflow demonstration."""
    asyncio.run(run_research_demo())


@cli.command()
@click.argument('query')
@click.option('--depth', default='standard', type=click.Choice(['basic', 'standard', 'deep']),
              help='Research depth level')
def research(query, depth):
    """Conduct research on a specific query."""
    async def run_single_research():
        console.print(Panel.fit(f"🔍 Researching: {query}", style="bold green"))
        
        with console.status("[bold green]Researching..."):
            try:
                results = await research_topic(query, depth)
                display_research_results(results)
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
    
    asyncio.run(run_single_research())


@cli.command()
def interactive():
    """Start interactive research session."""
    async def interactive_session():
        console.print(Panel.fit("🔬 Interactive Research Session", style="bold blue"))
        console.print("Type 'quit' to exit, 'help' for commands")
        
        while True:
            try:
                query = console.input("\n[bold cyan]Research Query:[/bold cyan] ")
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'help':
                    console.print("\nCommands:")
                    console.print("  - Enter a research query to start research")
                    console.print("  - 'quit' or 'exit' to stop")
                    console.print("  - 'help' to show this message")
                    continue
                elif not query.strip():
                    continue
                
                # Ask for research depth
                depth = console.input("[bold yellow]Research Depth (basic/standard/deep):[/bold yellow] ") or "standard"
                if depth not in ['basic', 'standard', 'deep']:
                    depth = 'standard'
                
                with console.status("[bold green]Researching..."):
                    results = await research_topic(query, depth)
                    display_research_results(results)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
        
        console.print("\n[bold green]Session ended.[/bold green]")
    
    asyncio.run(interactive_session())


@cli.command()
def info():
    """Show application information and configuration."""
    settings = get_settings()
    
    info_table = Table(title="LangGraph Lab Configuration", show_header=True, header_style="bold magenta")
    info_table.add_column("Setting", style="cyan", width=25)
    info_table.add_column("Value", style="white")
    
    info_table.add_row("App Name", settings.app_name)
    info_table.add_row("Version", settings.app_version)
    info_table.add_row("Debug Mode", str(settings.debug))
    info_table.add_row("Log Level", settings.log_level)
    info_table.add_row("OpenAI Model", settings.openai_model)
    info_table.add_row("Max Iterations", str(settings.max_iterations))
    info_table.add_row("Timeout (seconds)", str(settings.timeout_seconds))
    
    # Check optional configurations
    info_table.add_row("Serper API", "✓ Configured" if settings.serper_api_key else "✗ Not configured")
    info_table.add_row("Tavily API", "✓ Configured" if settings.tavily_api_key else "✗ Not configured")
    info_table.add_row("LangSmith Tracing", "✓ Enabled" if settings.langchain_tracing_v2 else "✗ Disabled")
    
    console.print(info_table)


def main():
    """Main function for direct script execution."""
    if len(sys.argv) == 1:
        # If no arguments provided, run demo
        console.print("[yellow]No command provided. Running demo...[/yellow]")
        asyncio.run(run_research_demo())
    else:
        cli()


if __name__ == "__main__":
    main()