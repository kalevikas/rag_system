
"""
LLM Handler for OpenAI GPT-4o-mini
"""
import logging
import os
from typing import Generator, List, Dict, Any, Optional, Tuple
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Master system prompt ──────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are an expert AI assistant that answers questions based STRICTLY on the provided reference documents.

## Core Rules
1. **Context-only answers** — Use ONLY the information in the "Reference Documents" section. Do NOT rely on external knowledge or training data.
2. **Honest fallback** — If the context does NOT contain enough information to answer the question, respond EXACTLY with:
   > ❌ **Not Found in Documents**
   > I couldn't find information about this in the uploaded documents. Please make sure the relevant documents have been uploaded and indexed.
3. **Never fabricate** — Do not invent names, numbers, dates, URLs, or any facts not present in the context.
4. **Be precise** — Quote or paraphrase specific parts of the documents. Prefer precision over verbosity.

## Formatting Rules
- Use **bold** for key terms, names, values, and important facts.
- Use bullet points (`•`) or numbered lists for multi-step processes or enumerable items.
- Use `inline code` for file names, commands, technical values, and identifiers.
- Use `##` section headers when the answer covers multiple distinct topics.
- Use tables for comparative or structured data.
- Keep responses focused and well-structured. Avoid filler sentences.

## Citation Rules
- Cite only real source names/URLs from the context (for example: filenames, API URL, webpage URL).
- Never cite internal labels like "Document 1", "Document 2", "Source 1", or "chunk".
- If source names are not available, skip the citation section instead of inventing labels.
"""


class LLMHandler:
	"""Handler for OpenAI LLM interactions"""
    
	def __init__(self,
				 api_key: Optional[str] = None,
				 model: str = "gpt-4o-mini",
				 temperature: float = 0.1,
				 max_tokens: int = 2048,
				 top_p: float = 0.95):
		"""
		Initialize LLM handler
        
		Args:
			api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
			model: Model name
			temperature: Sampling temperature
			max_tokens: Maximum tokens to generate
			top_p: Nucleus sampling parameter
		"""
		self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
		if not self.api_key:
			raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
		self.model = model
		self.temperature = temperature
		self.max_tokens = max_tokens
		self.top_p = top_p
        
		# Initialize client
		self.client = OpenAI(api_key=self.api_key)
        
		logger.info(f"LLMHandler initialized with model: {model}")
    
	def generate(self,
				 prompt: str,
				 system_message: Optional[str] = None,
				 temperature: Optional[float] = None,
				 max_tokens: Optional[int] = None) -> str:
		"""
		Generate response from LLM
        
		Args:
			prompt: User prompt
			system_message: System message to set behavior
			temperature: Override default temperature
			max_tokens: Override default max tokens
            
		Returns:
			Generated text
		"""
		messages = []
        
		# Add system message if provided
		if system_message:
			messages.append({"role": "system", "content": system_message})
        
		# Add user prompt
		messages.append({"role": "user", "content": prompt})
        
		try:
			response = self.client.chat.completions.create(
				model=self.model,
				messages=messages,
				temperature=temperature or self.temperature,
				max_tokens=max_tokens or self.max_tokens,
				top_p=self.top_p
			)
            
			answer = response.choices[0].message.content
			logger.debug(f"Generated response: {len(answer)} characters")
            
			return answer
            
		except Exception as e:
			logger.error(f"Error generating response: {e}")
			return f"Error: {str(e)}"

	def generate_stream(
		self,
		prompt: str,
		system_message: Optional[str] = None,
		temperature: Optional[float] = None,
		max_tokens: Optional[int] = None,
	) -> Generator[str, None, None]:
		"""
		Stream response from LLM token by token.

		Yields:
			Text chunks as they arrive from the API.
		"""
		messages = []
		if system_message:
			messages.append({"role": "system", "content": system_message})
		messages.append({"role": "user", "content": prompt})

		try:
			stream = self.client.chat.completions.create(
				model=self.model,
				messages=messages,
				temperature=temperature or self.temperature,
				max_tokens=max_tokens or self.max_tokens,
				top_p=self.top_p,
				stream=True,
			)
			for chunk in stream:
				delta = chunk.choices[0].delta
				if delta and delta.content:
					yield delta.content
		except Exception as e:
			logger.error(f"Error in streaming generation: {e}")
			yield f"Error: {str(e)}"

	def generate_with_context(self,
							 query: str,
							 context: str,
							 system_message: Optional[str] = None) -> str:
		"""
		Generate response using retrieved context
        
		Args:
			query: User query
			context: Retrieved context
			system_message: Optional system message
            
		Returns:
			Generated answer
		"""
		# Default system message for RAG
		if system_message is None:
			system_message = """You are a helpful AI assistant. Answer questions based on the provided context. 
If the context doesn't contain enough information to fully answer the question, say so clearly.
Be concise, accurate, and cite specific parts of the context when relevant."""
        
		# Construct prompt
		prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
		return self.generate(prompt, system_message=system_message)
    
	def generate_with_conversation(self,
								  query: str,
								  context: str,
								  conversation_history: List[Dict[str, str]]) -> str:
		"""
		Generate response with conversation history
        
		Args:
			query: Current query
			context: Retrieved context
			conversation_history: List of previous messages
            
		Returns:
			Generated answer
		"""
		messages = [
			{
				"role": "system",
				"content": "You are a helpful AI assistant. Use the provided context and conversation history to answer questions accurately."
			}
		]
        
		# Add conversation history
		messages.extend(conversation_history)
        
		# Add current query with context
		current_message = f"""Context:
{context}

Question: {query}

Answer:"""
        
		messages.append({"role": "user", "content": current_message})
        
		try:
			response = self.client.chat.completions.create(
				model=self.model,
				messages=messages,
				temperature=self.temperature,
				max_tokens=self.max_tokens,
				top_p=self.top_p
			)
            
			return response.choices[0].message.content
            
		except Exception as e:
			logger.error(f"Error in conversation generation: {e}")
			return f"Error: {str(e)}"
    
	def summarize(self, text: str, max_length: int = 200) -> str:
		"""
		Summarize text
        
		Args:
			text: Text to summarize
			max_length: Maximum length of summary
            
		Returns:
			Summary
		"""
		prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"
		return self.generate(prompt, temperature=0.3)
    
	def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
		"""
		Extract key points from text
        
		Args:
			text: Text to analyze
			num_points: Number of key points to extract
            
		Returns:
			List of key points
		"""
		prompt = f"""Extract {num_points} key points from the following text. 
Return each point on a new line, starting with a dash (-).

Text:
{text}

Key Points:"""
        
		response = self.generate(prompt, temperature=0.2)
        
		# Parse response into list
		points = [
			line.strip().lstrip('-').strip() 
			for line in response.split('\n') 
			if line.strip().startswith('-')
		]
        
		return points[:num_points]


class PromptTemplates:
	"""Collection of prompt templates for RAG"""

	@staticmethod
	def rag_answer_prompt(
		context: str,
		query: str,
		history_summary: str = "",
	) -> Tuple[str, str]:
		"""
		Primary RAG prompt — returns (system_message, user_message).

		Uses the master RAG_SYSTEM_PROMPT as system_message so that
		anti-hallucination + formatting rules are always enforced.
		"""
		history_block = ""
		if history_summary.strip():
			history_block = f"\n\n## 💬 Conversation History\n{history_summary}\n"

		user_message = f"""## 📄 Reference Documents
{'─' * 60}
{context}
{'─' * 60}
{history_block}
## ❓ Question
{query}

## 📝 Instructions
Provide a clear, well-formatted answer using **only** the Reference Documents above.
If the answer is not in the documents, use the exact fallback phrase specified in your instructions.
"""
		return RAG_SYSTEM_PROMPT, user_message

	# ── Backward-compatible methods (kept for any external callers) ───────────

	@staticmethod
	def basic_rag_prompt(context: str, query: str) -> str:
		_, user_msg = PromptTemplates.rag_answer_prompt(context, query)
		return user_msg

	@staticmethod
	def detailed_rag_prompt(context: str, query: str) -> str:
		_, user_msg = PromptTemplates.rag_answer_prompt(context, query)
		return user_msg

	@staticmethod
	def conversational_prompt(context: str, query: str, history_summary: str) -> str:
		_, user_msg = PromptTemplates.rag_answer_prompt(context, query, history_summary)
		return user_msg

	@staticmethod
	def analytical_prompt(context: str, query: str) -> str:
		_, user_msg = PromptTemplates.rag_answer_prompt(context, query)
		return user_msg


# Example usage
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
    
	# Initialize LLM handler
	try:
		llm = LLMHandler(model="gpt-4o-mini")
        
		# Test generation
		context = "Machine learning is a subset of AI that enables computers to learn from data without explicit programming."
		query = "What is machine learning?"
        
		answer = llm.generate_with_context(query, context)
		print(f"\nQuery: {query}")
		print(f"Answer: {answer}")
        
	except ValueError as e:
		print(f"Error: {e}")
		print("Please set OPENAI_API_KEY environment variable")
