"""
Conversation Memory for multi-turn chat
"""
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    In-memory conversation history manager for multi-turn chat
    """
    def __init__(self, max_turns: int = 20):
        """
        Args:
            max_turns: Maximum number of turns to keep in memory
        """
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation history
        Args:
            role: 'user' or 'assistant'
            content: message text
        """
        self.history.append({"role": role, "content": content})
        # Truncate history if needed
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
        logger.debug(f"Added message: {role} - {content[:50]}")

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the full conversation history
        Returns:
            List of message dicts
        """
        return self.history

    def get_history_summary(self) -> str:
        """
        Get a summary of the conversation history for prompt context
        Returns:
            String summary of previous turns
        """
        summary = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in self.history
        ])
        return summary

    def clear(self):
        """
        Clear the conversation history
        """
        self.history = []
        logger.info("Conversation history cleared.")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    memory = ConversationMemory(max_turns=5)
    memory.add_message("user", "Hello!")
    memory.add_message("assistant", "Hi, how can I help?")
    memory.add_message("user", "Tell me about RAG systems.")
    print(memory.get_history_summary())
    memory.clear()