"""
Text processing tool for LangGraph Lab.
"""
import re
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from ..config.settings import get_settings
from ..utils.logging import get_logger


class TextProcessorTool:
    """
    Tool for processing, analyzing, and summarizing text content.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("tools.text_processor")
        
        # Initialize language model for text processing
        self.llm = ChatOpenAI(
            model=self.settings.openai_model,
            temperature=0.1,
            api_key=self.settings.openai_api_key
        )
    
    async def summarize(
        self,
        text: str,
        max_length: int = 500,
        style: str = "concise"
    ) -> str:
        """
        Summarize text content.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            style: Summary style ("concise", "detailed", "bullet_points")
            
        Returns:
            Summarized text
        """
        if not text.strip():
            return ""
        
        if len(text) <= max_length:
            return text
        
        self.logger.info(f"Summarizing text of length {len(text)} to {max_length} characters")
        
        # Create style-specific prompt
        style_prompts = {
            "concise": "Create a concise summary that captures the main points.",
            "detailed": "Create a comprehensive summary that includes key details and context.",
            "bullet_points": "Create a summary using bullet points for easy reading."
        }
        
        prompt = f"""
        {style_prompts.get(style, style_prompts["concise"])}
        
        Original text:
        {text}
        
        Summary (max {max_length} characters):
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert at creating clear, accurate summaries."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            summary = response.content.strip()
            
            # Ensure summary doesn't exceed max length
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing text: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    async def extract_key_points(
        self,
        text: str,
        max_points: int = 10
    ) -> List[str]:
        """
        Extract key points from text.
        
        Args:
            text: Text to analyze
            max_points: Maximum number of key points to extract
            
        Returns:
            List of key points
        """
        if not text.strip():
            return []
        
        self.logger.info(f"Extracting key points from text of length {len(text)}")
        
        prompt = f"""
        Extract the most important key points from the following text. 
        Return up to {max_points} points, each as a clear, concise statement.
        Format each point on a new line starting with "- ".
        
        Text:
        {text}
        
        Key points:
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert at identifying and extracting key information from text."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            points_text = response.content.strip()
            
            # Parse points from response
            points = []
            for line in points_text.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    points.append(line[2:].strip())
                elif line and not line.startswith(('Key points:', 'Points:')):
                    # Handle points that might not have the "- " prefix
                    points.append(line)
            
            return points[:max_points]
            
        except Exception as e:
            self.logger.error(f"Error extracting key points: {e}")
            return []
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0, "explanation": "Empty text"}
        
        prompt = f"""
        Analyze the sentiment of the following text and provide:
        1. Overall sentiment (positive, negative, or neutral)
        2. Confidence level (0.0 to 1.0)
        3. Brief explanation of the sentiment

        Text:
        {text}

        Respond in this exact format:
        Sentiment: [positive/negative/neutral]
        Confidence: [0.0-1.0]
        Explanation: [brief explanation]
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert at sentiment analysis."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            analysis = response.content.strip()
            
            # Parse response
            result = {"sentiment": "neutral", "confidence": 0.0, "explanation": ""}
            
            for line in analysis.split('\n'):
                if line.startswith('Sentiment:'):
                    result["sentiment"] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('Confidence:'):
                    try:
                        result["confidence"] = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        result["confidence"] = 0.0
                elif line.startswith('Explanation:'):
                    result["explanation"] = line.split(':', 1)[1].strip()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {"sentiment": "neutral", "confidence": 0.0, "explanation": f"Error: {e}"}
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using simple regex patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with entity types and found entities
        """
        entities = {
            "emails": [],
            "urls": [],
            "phone_numbers": [],
            "dates": [],
            "numbers": []
        }
        
        if not text.strip():
            return entities
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities["emails"] = re.findall(email_pattern, text)
        
        # URL pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
        entities["urls"] = re.findall(url_pattern, text)
        
        # Phone number pattern (simple)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        entities["phone_numbers"] = re.findall(phone_pattern, text)
        
        # Date pattern (MM/DD/YYYY or MM-DD-YYYY)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b'
        entities["dates"] = re.findall(date_pattern, text)
        
        # Number pattern (integers and decimals)
        number_pattern = r'\b\d+\.?\d*\b'
        entities["numbers"] = re.findall(number_pattern, text)
        
        return entities
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this is not the last chunk, try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                last_period = text.rfind('.', end - 100, end)
                last_exclamation = text.rfind('!', end - 100, end)
                last_question = text.rfind('?', end - 100, end)
                
                sentence_end = max(last_period, last_exclamation, last_question)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunks.append(text[start:end].strip())
            start = end - overlap
        
        return chunks