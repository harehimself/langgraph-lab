"""
Research agent implementation for LangGraph Lab.
"""
from typing import Dict, Any, List
from .base_agent import ToolAgent
from ..tools.web_search import WebSearchTool
from ..tools.text_processor import TextProcessorTool


class ResearchAgent(ToolAgent):
    """
    Agent specialized in conducting research tasks.
    Capable of web searching, information synthesis, and analysis.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="research_agent", **kwargs)
        
        # Initialize tools
        self.web_search = WebSearchTool()
        self.text_processor = TextProcessorTool()
        
        # Add tools to the agent
        self.add_tool("web_search", self.web_search.search, "Search the web for information")
        self.add_tool("summarize_text", self.text_processor.summarize, "Summarize long text content")
        self.add_tool("extract_key_points", self.text_processor.extract_key_points, "Extract key points from text")
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the research agent."""
        return """You are a research specialist AI agent. Your role is to:

1. Conduct thorough web searches on given topics
2. Analyze and synthesize information from multiple sources
3. Extract key insights and relevant data points
4. Provide well-structured, factual summaries
5. Identify knowledge gaps and suggest follow-up research

Guidelines:
- Always verify information from multiple sources when possible
- Clearly distinguish between facts and opinions
- Cite sources when presenting findings
- Be objective and unbiased in your analysis
- If information is unclear or contradictory, acknowledge this

Available tools:
- web_search: Search the web for current information
- summarize_text: Create concise summaries of long content
- extract_key_points: Identify main points from text

You should be thorough but efficient in your research process."""
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process research request and update state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with research results
        """
        query = state.get("query", "")
        research_depth = state.get("research_depth", "standard")
        
        if not query:
            self.logger.warning("No query provided for research")
            return {**state, "error": "No research query provided"}
        
        self.logger.info(f"Starting research for query: {query}")
        
        try:
            # Step 1: Conduct web search
            search_results = await self._use_tool("web_search", query=query, max_results=5)
            
            # Step 2: Process and analyze results
            if search_results:
                analysis = await self._analyze_search_results(search_results, query)
                
                # Step 3: Extract key insights
                key_points = await self._use_tool("extract_key_points", text=analysis)
                
                # Update state with research results
                updated_state = {
                    **state,
                    "research_completed": True,
                    "search_results": search_results,
                    "analysis": analysis,
                    "key_insights": key_points,
                    "sources": [result.get("url", "") for result in search_results if result.get("url")]
                }
            else:
                updated_state = {
                    **state,
                    "research_completed": True,
                    "error": "No search results found",
                    "analysis": f"Unable to find relevant information for query: {query}"
                }
            
            self._log_state_update(state, updated_state)
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Error during research process: {e}")
            return {
                **state,
                "error": f"Research failed: {str(e)}",
                "research_completed": True
            }
    
    async def _analyze_search_results(self, search_results: List[Dict], query: str) -> str:
        """
        Analyze search results and provide synthesized information.
        
        Args:
            search_results: List of search result dictionaries
            query: Original search query
            
        Returns:
            Synthesized analysis of the search results
        """
        # Combine all search result content
        combined_content = ""
        for result in search_results:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            url = result.get("url", "")
            
            combined_content += f"\nSource: {title} ({url})\nContent: {snippet}\n"
        
        # Create analysis prompt
        analysis_prompt = f"""
        Based on the following search results for the query "{query}", provide a comprehensive analysis:

        Search Results:
        {combined_content}

        Please provide:
        1. A summary of the key findings
        2. Important facts and data points
        3. Different perspectives or viewpoints found
        4. Any contradictions or uncertainties
        5. Areas where more research might be needed

        Keep your analysis factual, well-structured, and cite the sources when relevant.
        """
        
        messages = [{"role": "user", "content": analysis_prompt}]
        formatted_messages = self._build_messages(messages)
        
        return await self._invoke_llm(formatted_messages)
    
    async def conduct_deep_research(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct deeper research with multiple search iterations.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with comprehensive research results
        """
        query = state.get("query", "")
        self.logger.info(f"Starting deep research for: {query}")
        
        # Generate related queries for comprehensive research
        related_queries = await self._generate_related_queries(query)
        
        all_results = []
        all_analyses = []
        
        # Research each query
        for related_query in [query] + related_queries:
            search_results = await self._use_tool("web_search", query=related_query, max_results=3)
            if search_results:
                analysis = await self._analyze_search_results(search_results, related_query)
                all_results.extend(search_results)
                all_analyses.append(analysis)
        
        # Synthesize all findings
        comprehensive_analysis = await self._synthesize_analyses(all_analyses, query)
        
        return {
            **state,
            "deep_research_completed": True,
            "comprehensive_analysis": comprehensive_analysis,
            "all_search_results": all_results,
            "related_queries": related_queries
        }
    
    async def _generate_related_queries(self, original_query: str) -> List[str]:
        """Generate related search queries for comprehensive research."""
        prompt = f"""
        Given the original query: "{original_query}"
        
        Generate 3 related search queries that would help provide a more comprehensive 
        understanding of this topic. The queries should explore different aspects, 
        perspectives, or related concepts.
        
        Return only the queries, one per line, without numbering or additional text.
        """
        
        messages = [{"role": "user", "content": prompt}]
        formatted_messages = self._build_messages(messages)
        response = await self._invoke_llm(formatted_messages)
        
        return [q.strip() for q in response.split('\n') if q.strip()]
    
    async def _synthesize_analyses(self, analyses: List[str], original_query: str) -> str:
        """Synthesize multiple analyses into a comprehensive report."""
        combined_analyses = "\n\n".join([f"Analysis {i+1}:\n{analysis}" for i, analysis in enumerate(analyses)])
        
        prompt = f"""
        Based on multiple research analyses for the query "{original_query}", 
        create a comprehensive synthesis that includes:

        1. Executive Summary
        2. Key Findings
        3. Supporting Evidence
        4. Different Perspectives
        5. Conclusions and Implications
        6. Recommendations for Further Research

        Research Analyses:
        {combined_analyses}

        Provide a well-structured, comprehensive report that synthesizes all the information.
        """
        
        messages = [{"role": "user", "content": prompt}]
        formatted_messages = self._build_messages(messages)
        
        return await self._invoke_llm(formatted_messages)