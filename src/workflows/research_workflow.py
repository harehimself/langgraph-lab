"""
Research workflow implementation using LangGraph.
"""
from typing import Dict, Any, List, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from ..agents.research_agent import ResearchAgent
from ..config.settings import get_settings
from ..utils.logging import get_logger


class ResearchState(TypedDict):
    """State schema for the research workflow."""
    query: str
    research_depth: str  # "basic", "standard", "deep"
    messages: List[Dict[str, Any]]
    research_completed: bool
    search_results: List[Dict[str, Any]]
    analysis: str
    key_insights: List[str]
    sources: List[str]
    error: str
    iteration_count: int
    needs_clarification: bool
    follow_up_questions: List[str]


class ResearchWorkflow:
    """
    LangGraph workflow for conducting research tasks.
    Demonstrates state management, conditional routing, and agent coordination.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("workflow.research")
        
        # Initialize agents
        self.research_agent = ResearchAgent()
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("basic_research", self._basic_research)
        workflow.add_node("deep_research", self._deep_research)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("generate_follow_ups", self._generate_follow_ups)
        workflow.add_node("finalize_results", self._finalize_results)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_query",
            self._decide_research_path,
            {
                "basic": "basic_research",
                "deep": "deep_research",
                "clarify": "generate_follow_ups"
            }
        )
        
        workflow.add_conditional_edges(
            "basic_research",
            self._check_research_quality,
            {
                "sufficient": "validate_results",
                "needs_more": "deep_research",
                "error": "finalize_results"
            }
        )
        
        workflow.add_edge("deep_research", "validate_results")
        
        workflow.add_conditional_edges(
            "validate_results",
            self._check_validation,
            {
                "complete": "finalize_results",
                "needs_follow_up": "generate_follow_ups"
            }
        )
        
        workflow.add_edge("generate_follow_ups", "finalize_results")
        workflow.add_edge("finalize_results", END)
        
        return workflow.compile()
    
    async def run(self, query: str, research_depth: str = "standard") -> Dict[str, Any]:
        """
        Run the research workflow.
        
        Args:
            query: Research query
            research_depth: Depth of research ("basic", "standard", "deep")
            
        Returns:
            Final workflow state with research results
        """
        self.logger.info(f"Starting research workflow for: {query}")
        
        # Initialize state
        initial_state = ResearchState(
            query=query,
            research_depth=research_depth,
            messages=[],
            research_completed=False,
            search_results=[],
            analysis="",
            key_insights=[],
            sources=[],
            error="",
            iteration_count=0,
            needs_clarification=False,
            follow_up_questions=[]
        )
        
        try:
            # Execute the workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            self.logger.info("Research workflow completed successfully")
            return final_state
            
        except Exception as e:
            self.logger.error(f"Research workflow failed: {e}")
            return {
                **initial_state,
                "error": f"Workflow execution failed: {str(e)}",
                "research_completed": True
            }
    
    async def _analyze_query(self, state: ResearchState) -> ResearchState:
        """Analyze the query to understand what type of research is needed."""
        self.logger.info("Analyzing research query")
        
        query = state["query"]
        
        # Simple query analysis (can be enhanced with NLP)
        query_lower = query.lower()
        
        # Check if query needs clarification
        unclear_indicators = ["what", "how", "explain", "tell me about", "describe"]
        if any(indicator in query_lower for indicator in unclear_indicators) and len(query.split()) < 4:
            state["needs_clarification"] = True
        
        # Determine complexity
        complex_indicators = ["compare", "analyze", "evaluate", "assess", "comprehensive", "detailed"]
        if any(indicator in query_lower for indicator in complex_indicators):
            if state["research_depth"] == "basic":
                state["research_depth"] = "standard"
        
        state["iteration_count"] += 1
        return state
    
    async def _basic_research(self, state: ResearchState) -> ResearchState:
        """Perform basic research using the research agent."""
        self.logger.info("Performing basic research")
        
        try:
            # Use research agent to conduct basic research
            updated_state = await self.research_agent.process(state)
            updated_state["iteration_count"] = state["iteration_count"] + 1
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Basic research failed: {e}")
            return {
                **state,
                "error": f"Basic research failed: {str(e)}",
                "research_completed": True,
                "iteration_count": state["iteration_count"] + 1
            }
    
    async def _deep_research(self, state: ResearchState) -> ResearchState:
        """Perform deep research with multiple iterations."""
        self.logger.info("Performing deep research")
        
        try:
            # Use research agent's deep research capability
            updated_state = await self.research_agent.conduct_deep_research(state)
            updated_state["iteration_count"] = state["iteration_count"] + 1
            updated_state["research_completed"] = True
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"Deep research failed: {e}")
            return {
                **state,
                "error": f"Deep research failed: {str(e)}",
                "research_completed": True,
                "iteration_count": state["iteration_count"] + 1
            }
    
    async def _validate_results(self, state: ResearchState) -> ResearchState:
        """Validate research results for completeness and accuracy."""
        self.logger.info("Validating research results")
        
        # Check if we have sufficient results
        has_results = bool(state.get("search_results") or state.get("analysis"))
        has_sources = bool(state.get("sources"))
        has_insights = bool(state.get("key_insights"))
        
        # Simple validation criteria
        validation_score = sum([has_results, has_sources, has_insights])
        
        if validation_score >= 2:
            self.logger.info("Research results validated successfully")
            state["research_completed"] = True
        else:
            self.logger.warning("Research results may need improvement")
            if state["iteration_count"] < self.settings.max_iterations:
                state["needs_clarification"] = True
        
        return state
    
    async def _generate_follow_ups(self, state: ResearchState) -> ResearchState:
        """Generate follow-up questions or suggestions."""
        self.logger.info("Generating follow-up questions")
        
        query = state["query"]
        analysis = state.get("analysis", "")
        
        # Generate follow-up questions based on the research
        follow_ups = []
        
        if state["needs_clarification"]:
            follow_ups.extend([
                f"What specific aspects of '{query}' would you like to explore further?",
                "Are you looking for recent developments or historical information?",
                "Do you need this information for a particular purpose or context?"
            ])
        
        if analysis and len(state.get("sources", [])) > 0:
            follow_ups.extend([
                "Would you like me to dive deeper into any particular aspect?",
                "Are there related topics you'd like me to research?",
                "Do you need more recent information on this topic?"
            ])
        
        state["follow_up_questions"] = follow_ups[:3]  # Limit to 3 questions
        return state
    
    async def _finalize_results(self, state: ResearchState) -> ResearchState:
        """Finalize and format the research results."""
        self.logger.info("Finalizing research results")
        
        # Ensure research is marked as completed
        state["research_completed"] = True
        
        # Add a completion message
        if not state.get("error"):
            completion_msg = {
                "role": "assistant",
                "content": f"Research completed for query: '{state['query']}'. "
                          f"Found {len(state.get('sources', []))} sources and "
                          f"generated {len(state.get('key_insights', []))} key insights."
            }
            state["messages"].append(completion_msg)
        
        return state
    
    def _decide_research_path(self, state: ResearchState) -> Literal["basic", "deep", "clarify"]:
        """Decide which research path to take based on query analysis."""
        if state["needs_clarification"]:
            return "clarify"
        elif state["research_depth"] == "deep":
            return "deep"
        else:
            return "basic"
    
    def _check_research_quality(self, state: ResearchState) -> Literal["sufficient", "needs_more", "error"]:
        """Check if basic research results are sufficient."""
        if state.get("error"):
            return "error"
        
        # Check quality indicators
        has_results = bool(state.get("search_results"))
        has_analysis = bool(state.get("analysis"))
        sufficient_sources = len(state.get("sources", [])) >= 2
        
        if has_results and has_analysis and sufficient_sources:
            return "sufficient"
        elif state["iteration_count"] < self.settings.max_iterations:
            return "needs_more"
        else:
            return "sufficient"  # Accept what we have if we've tried enough
    
    def _check_validation(self, state: ResearchState) -> Literal["complete", "needs_follow_up"]:
        """Check validation results."""
        if state["needs_clarification"] and state["iteration_count"] < self.settings.max_iterations:
            return "needs_follow_up"
        else:
            return "complete"


# Convenience function for easy workflow execution
async def research_topic(query: str, research_depth: str = "standard") -> Dict[str, Any]:
    """
    Convenience function to run research workflow.
    
    Args:
        query: Research query
        research_depth: Depth of research ("basic", "standard", "deep")
        
    Returns:
        Research results
    """
    workflow = ResearchWorkflow()
    return await workflow.run(query, research_depth)