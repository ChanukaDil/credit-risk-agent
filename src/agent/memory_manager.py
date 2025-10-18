"""
Memory Manager for Credit Risk Agent
Handles conversation history, context management, and session persistence
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging
from pathlib import Path

# LangChain imports
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain.schema import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """
    Manages conversation history and context for the credit risk agent
    
    Features:
    - Store conversation history
    - Retrieve relevant past interactions
    - Summarize long conversations
    - Persist sessions to disk
    - Load previous sessions
    """
    
    def __init__(
        self,
        memory_type: str = "buffer_window",
        window_size: int = 10,
        max_token_limit: int = 2000,
        session_dir: str = "results/sessions"
    ):
        """
        Initialize memory manager
        
        Args:
            memory_type: Type of memory ('buffer', 'buffer_window', 'summary')
            window_size: Number of recent messages to keep (for buffer_window)
            max_token_limit: Maximum tokens to keep in memory
            session_dir: Directory to save session files
        """
        self.memory_type = memory_type
        self.window_size = window_size
        self.max_token_limit = max_token_limit
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory
        self.memory = self._create_memory()
        
        # Session tracking
        self.session_id = None
        self.session_metadata = {}
        self.conversation_count = 0
        
        logger.info(f"Memory manager initialized: {memory_type}")
    
    def _create_memory(self) -> Any:
        """Create appropriate memory object based on type"""
        
        if self.memory_type == "buffer":
            return ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        
        elif self.memory_type == "buffer_window":
            return ConversationBufferWindowMemory(
                k=self.window_size,
                return_messages=True,
                memory_key="chat_history"
            )
        
        elif self.memory_type == "summary":
            # Note: Requires LLM for summarization
            return ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        
        else:
            logger.warning(f"Unknown memory type: {self.memory_type}, using buffer")
            return ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to conversation history
        
        Args:
            role: 'human' or 'ai'
            content: Message content
            metadata: Optional metadata (timestamp, risk_score, etc.)
        """
        if role == "human":
            message = HumanMessage(content=content)
        elif role == "ai":
            message = AIMessage(content=content)
        else:
            logger.warning(f"Unknown role: {role}")
            return
        
        # Add to memory
        if role == "human":
            self.memory.chat_memory.add_user_message(content)
        else:
            self.memory.chat_memory.add_ai_message(content)
        
        # Track conversation count
        self.conversation_count += 1
        
        # Store metadata if provided
        if metadata:
            self._store_metadata(metadata)
        
        logger.debug(f"Added {role} message to memory")
    
    def get_history(self, last_n: Optional[int] = None) -> List[BaseMessage]:
        """
        Get conversation history
        
        Args:
            last_n: Return only last n messages (None = all)
            
        Returns:
            List of messages
        """
        messages = self.memory.chat_memory.messages
        
        if last_n:
            return messages[-last_n:]
        return messages
    
    def get_history_as_string(self, last_n: Optional[int] = None) -> str:
        """
        Get conversation history as formatted string
        
        Args:
            last_n: Return only last n messages
            
        Returns:
            Formatted string
        """
        messages = self.get_history(last_n)
        
        formatted = []
        for msg in messages:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            formatted.append(f"{role}: {msg.content}")
        
        return "\n\n".join(formatted)
    
    def get_context_for_llm(self) -> str:
        """Get formatted context suitable for LLM prompt"""
        return self.memory.load_memory_variables({})['chat_history']
    
    def search_history(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search conversation history for relevant past interactions
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant message pairs
        """
        messages = self.get_history()
        results = []
        
        # Simple keyword search (can be enhanced with embeddings)
        query_lower = query.lower()
        
        for i, msg in enumerate(messages):
            content_lower = msg.content.lower()
            if query_lower in content_lower:
                # Get context (previous and next message)
                context = {
                    'index': i,
                    'message': msg.content,
                    'type': 'human' if isinstance(msg, HumanMessage) else 'ai',
                    'timestamp': datetime.now().isoformat()  # Would be actual timestamp
                }
                
                # Add previous message for context
                if i > 0:
                    context['previous'] = messages[i-1].content
                
                # Add next message for context
                if i < len(messages) - 1:
                    context['next'] = messages[i+1].content
                
                results.append(context)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def _store_metadata(self, metadata: Dict):
        """Store metadata for current session"""
        timestamp = datetime.now().isoformat()
        
        if 'metadata' not in self.session_metadata:
            self.session_metadata['metadata'] = []
        
        self.session_metadata['metadata'].append({
            'timestamp': timestamp,
            **metadata
        })
    
    def clear(self):
        """Clear conversation history"""
        self.memory.clear()
        self.conversation_count = 0
        logger.info("Memory cleared")
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_id = session_id
        self.session_metadata = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'conversation_count': 0,
            'metadata': []
        }
        
        logger.info(f"Started session: {session_id}")
        return session_id
    
    def end_session(self, auto_save: bool = True) -> Dict:
        """
        End current session
        
        Args:
            auto_save: Automatically save session to disk
            
        Returns:
            Session summary
        """
        if self.session_id is None:
            logger.warning("No active session to end")
            return {}
        
        self.session_metadata['end_time'] = datetime.now().isoformat()
        self.session_metadata['conversation_count'] = self.conversation_count
        self.session_metadata['total_messages'] = len(self.get_history())
        
        # Save if requested
        if auto_save:
            self.save_session()
        
        summary = self.session_metadata.copy()
        
        # Clear for next session
        self.session_id = None
        self.session_metadata = {}
        
        logger.info("Session ended")
        return summary
    
    def save_session(self, filename: Optional[str] = None):
        """
        Save current session to disk
        
        Args:
            filename: Optional custom filename
        """
        if self.session_id is None:
            logger.warning("No active session to save")
            return
        
        # Prepare data
        session_data = {
            **self.session_metadata,
            'messages': [
                {
                    'type': 'human' if isinstance(msg, HumanMessage) else 'ai',
                    'content': msg.content
                }
                for msg in self.get_history()
            ]
        }
        
        # Determine filename
        if filename is None:
            filename = f"{self.session_id}.json"
        
        filepath = self.session_dir / filename
        
        # Save to disk
        try:
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Session saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
    
    def load_session(self, filename: str) -> bool:
        """
        Load session from disk
        
        Args:
            filename: Session file to load
            
        Returns:
            True if successful
        """
        filepath = self.session_dir / filename
        
        if not filepath.exists():
            logger.error(f"Session file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            # Clear current memory
            self.clear()
            
            # Restore metadata
            self.session_id = session_data.get('session_id')
            self.session_metadata = {
                k: v for k, v in session_data.items() 
                if k != 'messages'
            }
            
            # Restore messages
            for msg_data in session_data.get('messages', []):
                self.add_message(msg_data['type'], msg_data['content'])
            
            logger.info(f"Session loaded from: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return False
    
    def list_sessions(self) -> List[Dict]:
        """
        List all saved sessions
        
        Returns:
            List of session summaries
        """
        sessions = []
        
        for filepath in self.session_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                sessions.append({
                    'filename': filepath.name,
                    'session_id': data.get('session_id'),
                    'start_time': data.get('start_time'),
                    'end_time': data.get('end_time'),
                    'message_count': len(data.get('messages', []))
                })
                
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")
        
        # Sort by start time (most recent first)
        sessions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return sessions
    
    def get_summary(self) -> Dict:
        """
        Get summary of current conversation
        
        Returns:
            Summary statistics
        """
        messages = self.get_history()
        
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        
        return {
            'session_id': self.session_id,
            'total_messages': len(messages),
            'human_messages': human_count,
            'ai_messages': ai_count,
            'conversation_count': self.conversation_count,
            'memory_type': self.memory_type
        }


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create memory manager
    memory = ConversationMemoryManager(
        memory_type="buffer_window",
        window_size=5
    )
    
    # Start session
    session_id = memory.start_session()
    print(f"Started session: {session_id}")
    
    # Add some conversations
    memory.add_message("human", "What are the lending criteria?")
    memory.add_message("ai", "The lending criteria include...")
    
    memory.add_message("human", "Can you assess this customer?", 
                       metadata={'risk_score': 15.2})
    memory.add_message("ai", "Based on the analysis, this customer is low risk...")
    
    # Get history
    print("\nConversation History:")
    print(memory.get_history_as_string())
    
    # Get summary
    print("\nSession Summary:")
    print(json.dumps(memory.get_summary(), indent=2))
    
    # Save session
    memory.save_session()
    
    # End session
    summary = memory.end_session()
    print("\nSession ended")
    print(json.dumps(summary, indent=2))
    
    # List all sessions
    print("\nAvailable Sessions:")
    sessions = memory.list_sessions()
    for session in sessions:
        print(f"- {session['filename']} ({session['message_count']} messages)")