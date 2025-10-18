

"""
LLM Interface - HuggingFace Llama 3.2 3B Wrapper
Handles all LLM interactions for the credit risk agent
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Wrapper for HuggingFace Llama 3.2 3B model
    Handles text generation, prompting, and token management
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        use_quantization: bool = True,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "auto"
    ):
        """
        Initialize LLM interface
        
        Args:
            model_name: HuggingFace model identifier
            use_quantization: Use 8-bit quantization for memory efficiency
            max_length: Maximum generation length
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        logger.info(f"Loading LLM: {model_name}")
        
        # Configure quantization for memory efficiency
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Create text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        logger.info(f"LLM loaded successfully on {self.model.device}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: User prompt
            max_new_tokens: Override default max length
            temperature: Override default temperature
            system_message: System context/instructions
            
        Returns:
            Generated text
        """
        # Format prompt with system message if provided
        if system_message:
            formatted_prompt = self._format_chat_prompt(system_message, prompt)
        else:
            formatted_prompt = prompt
        
        # Override parameters if provided
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_length,
            "temperature": temperature or self.temperature,
            "top_p": self.top_p,
            "do_sample": True
        }
        
        # Generate
        try:
            outputs = self.pipeline(
                formatted_prompt,
                **gen_kwargs
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Remove prompt from output
            if generated_text.startswith(formatted_prompt):
                generated_text = generated_text[len(formatted_prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _format_chat_prompt(self, system_message: str, user_message: str) -> str:
        """
        Format chat prompt in Llama 3.2 format
        
        Args:
            system_message: System instructions
            user_message: User query
            
        Returns:
            Formatted prompt
        """
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def generate_with_context(
        self,
        query: str,
        context: Dict[str, any],
        system_message: str
    ) -> str:
        """
        Generate response with contextual information
        
        Args:
            query: User query
            context: Dictionary with risk_score, policies, etc.
            system_message: System instructions
            
        Returns:
            Generated response
        """
        # Build context string
        context_str = self._build_context_string(context)
        
        # Create full prompt
        full_prompt = f"""Context Information:
{context_str}

User Query: {query}

Please provide a clear, professional response based on the context provided."""
        
        return self.generate(
            prompt=full_prompt,
            system_message=system_message
        )
    
    def _build_context_string(self, context: Dict[str, any]) -> str:
        """Build formatted context string from dictionary"""
        context_parts = []
        
        if "risk_score" in context:
            context_parts.append(f"Risk Score: {context['risk_score']}")
        
        if "risk_category" in context:
            context_parts.append(f"Risk Category: {context['risk_category']}")
        
        if "action" in context:
            context_parts.append(f"Recommended Action: {context['action']}")
        
        if "policies" in context:
            context_parts.append(f"\nRelevant Policies:\n{context['policies']}")
        
        if "similar_cases" in context:
            context_parts.append(f"\nSimilar Cases:\n{context['similar_cases']}")
        
        return "\n".join(context_parts)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum token count"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return self.tokenizer.decode(tokens)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Convenience method for generation"""
        return self.generate(prompt, **kwargs)


# Example usage
if __name__ == "__main__":
    # Initialize LLM
    llm = LLMInterface(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        use_quantization=True,
        temperature=0.7
    )
    
    # Test generation
    system_msg = "You are a helpful banking assistant specializing in credit risk assessment."
    user_query = "What factors should I consider when assessing credit risk?"
    
    response = llm.generate(
        prompt=user_query,
        system_message=system_msg,
        max_new_tokens=256
    )
    
    print("Response:", response)
    print(f"Tokens: {llm.count_tokens(response)}")