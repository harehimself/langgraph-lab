"""
Base agent class for LangGraph Lab.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from ..config.settings import get_settings
from ..utils.logging import get_logger


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the LangGraph workflow.
    """
    
    def __init__(
        self,
        name: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name
            model: Language model to use (defaults to settings)
            temperature: Model temperature for response generation
            max_tokens: Maximum tokens for response
            system_prompt: System prompt for the agent
        """
        self.name = name
        self.logger = get_logger(f"agent.{name}")
        self.settings = get_settings()
        
        # Initialize language model
        self.llm = ChatOpenAI(
            model=model or self.settings.openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.settings.openai_api_key
        )
        
        # Set system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        self.logger.info(f"Initialized agent: {self.name}")
    
    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for this agent."""
        pass
    
    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the current state and return updated state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state dictionary
        """
        pass
    
    def _build_messages(
        self,
        messages: List[Dict[str, Any]],
        include_system: bool = True
    ) -> List[BaseMessage]:
        """
        Build a list of messages for the language model.
        
        Args:
            messages: List of message dictionaries
            include_system: Whether to include system prompt
            
        Returns:
            List of BaseMessage objects
        """
        formatted_messages = []
        
        if include_system and self.system_prompt:
            formatted_messages.append(SystemMessage(content=self.system_prompt))
        
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                formatted_messages.append(SystemMessage(content=msg["content"]))
        
        return formatted_messages
    
    async def _invoke_llm(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> str:
        """
        Invoke the language model with messages.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional parameters for the LLM
            
        Returns:
            Generated response text
        """
        try:
            response = await self.llm.ainvoke(messages, **kwargs)
            return response.content.strip()
        except Exception as e:
            self.logger.error(f"Error invoking LLM: {e}")
            raise
    
    def _log_state_update(self, old_state: Dict[str, Any], new_state: Dict[str, Any]):
        """Log state updates for debugging."""
        if self.settings.debug:
            self.logger.debug(f"State update by {self.name}:")
            for key, value in new_state.items():
                if key not in old_state or old_state[key] != value:
                    self.logger.debug(f"  {key}: {value}")


class ToolAgent(BaseAgent):
    """
    Base class for agents that can use tools.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tools = {}
    
    def add_tool(self, name: str, tool_func: callable, description: str = ""):
        """
        Add a tool to this agent.
        
        Args:
            name: Tool name
            tool_func: Tool function
            description: Tool description
        """
        self.tools[name] = {
            "function": tool_func,
            "description": description
        }
        self.logger.info(f"Added tool '{name}' to agent {self.name}")
    
    async def _use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Use a specific tool.
        
        Args:
            tool_name: Name of the tool to use
            **kwargs: Tool parameters
            
        Returns:
            Tool result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in agent {self.name}")
        
        try:
            tool_func = self.tools[tool_name]["function"]
            result = await tool_func(**kwargs) if callable(tool_func) else tool_func
            self.logger.info(f"Tool '{tool_name}' executed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error using tool '{tool_name}': {e}")
            raise