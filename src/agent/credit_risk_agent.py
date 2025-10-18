

"""
Credit Risk Agent - Main Orchestrator
Integrates: Autoencoder + RAG + LLM into a conversational agent
Uses: LangChain for agent framework
"""

import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import our components
from src.agent.llm_interface import LLMInterface
from src.agent.rag_system import RAGSystem
from src.agent.agent_tools import create_agent_tools

logger = logging.getLogger(__name__)


class CreditRiskAgent:
    """
    Main Credit Risk Assessment Agent
    
    Capabilities:
    1. Assess customer credit risk (using autoencoder)
    2. Retrieve relevant policies (using RAG)
    3. Find similar cases
    4. Explain decisions in natural language (using LLM)
    5. Answer questions about credit risk
    """
    
    def __init__(
        self,
        model_path: str = "models/autoencoder/default_autoencoder.pth",
        preprocessor_path: str = "models/preprocessor/preprocessor.pkl",
        val_errors_path: str = "results/validation_errors.npy",
        knowledge_base_path: str = "knowledge_base",
        vector_db_path: str = "models/vector_db/chroma_db",
        llm_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        use_memory: bool = True
    ):
        """
        Initialize Credit Risk Agent
        
        Args:
            model_path: Path to autoencoder model
            preprocessor_path: Path to preprocessor
            val_errors_path: Path to validation errors
            knowledge_base_path: Path to knowledge base documents
            vector_db_path: Path to vector database
            llm_model: HuggingFace LLM model name
            use_memory: Use conversation memory
        """
        logger.info("Initializing Credit Risk Agent")
        
        # Initialize LLM
        logger.info("Loading LLM...")
        self.llm_interface = LLMInterface(
            model_name=llm_model,
            use_quantization=True,
            temperature=0.7
        )
        
        # Initialize RAG system
        logger.info("Loading RAG system...")
        self.rag_system = RAGSystem(
            knowledge_base_path=knowledge_base_path,
            vector_db_path=vector_db_path
        )
        
        # Create LangChain-compatible LLM wrapper
        self.llm = self._create_langchain_llm_wrapper()
        
        # Create tools
        logger.info("Creating agent tools...")
        self.tools = create_agent_tools(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            val_errors_path=val_errors_path,
            rag_system=self.rag_system
        )
        
        # Create agent prompt
        self.agent_prompt = self._create_agent_prompt()
        
        # Create agent
        logger.info("Creating agent executor...")
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.agent_prompt
        )
        
        # Create memory
        if use_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        else:
            self.memory = None
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        logger.info("Credit Risk Agent initialized successfully")
    
    def _create_langchain_llm_wrapper(self):
        """Create LangChain-compatible LLM wrapper"""
        from langchain.llms.base import LLM
        from typing import Any, List, Mapping, Optional
        
        class LlamaLLM(LLM):
            """LangChain wrapper for our LLM interface"""
            llm_interface: Any = None
            
            @property
            def _llm_type(self) -> str:
                return "llama"
            
            def _call(
                self,
                prompt: str,
                stop: Optional[List[str]] = None,
                **kwargs: Any,
            ) -> str:
                return self.llm_interface.generate(prompt)
            
            @property
            def _identifying_params(self) -> Mapping[str, Any]:
                return {"model": "llama-3.2-3b"}
        
        # Create and return wrapper
        llm_wrapper = LlamaLLM()
        llm_wrapper.llm_interface = self.llm_interface
        return llm_wrapper
    
    def _create_agent_prompt(self) -> PromptTemplate:
        """Create agent prompt template"""
        template = """You are an expert credit risk assessment assistant for a bank.

You have access to the following tools:
{tools}

Your capabilities:
1. Assess customer credit risk using deep learning models
2. Retrieve relevant bank policies and regulations
3. Find similar historical cases
4. Explain decisions clearly and professionally

When answering:
- Be professional and clear
- Provide specific recommendations
- Cite policies when relevant
- Use data and evidence
- Format responses clearly

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        return PromptTemplate(
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            },
            template=template
        )
    
    def query(self, question: str) -> Dict[str, str]:
        """
        Process user query
        
        Args:
            question: User question
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            logger.info(f"Processing query: {question}")
            
            # Run agent
            result = self.agent_executor.invoke({"input": question})
            
            return {
                "question": question,
                "answer": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error: {str(e)}",
                "error": str(e),
                "success": False
            }
    
    def assess_customer(self, customer_data: Dict) -> Dict[str, str]:
        """
        Assess a specific customer
        
        Args:
            customer_data: Customer data dictionary
            
        Returns:
            Assessment result
        """
        import json
        customer_json = json.dumps(customer_data)
        
        question = f"Please assess the credit risk for this customer: {customer_json}"
        return self.query(question)
    
    def explain_decision(
        self,
        risk_score: float,
        risk_category: str,
        action: str
    ) -> str:
        """
        Generate natural language explanation for a decision
        
        Args:
            risk_score: Risk score (0-100)
            risk_category: Risk category (LOW/MEDIUM/HIGH)
            action: Recommended action
            
        Returns:
            Natural language explanation
        """
        context = {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "action": action
        }
        
        # Retrieve relevant policies
        policy_query = f"lending guidelines for {risk_category.lower()} risk customers"
        policies = self.rag_system.get_context_for_query(policy_query)
        context["policies"] = policies
        
        # Generate explanation using LLM
        system_message = """You are a credit risk analyst. 
Explain the decision clearly and professionally, citing relevant policies."""
        
        query = f"""The customer received a risk score of {risk_score}/100, 
categorized as {risk_category} risk. 
The recommended action is: {action}

Explain this decision to the customer in a clear, professional manner."""
        
        explanation = self.llm_interface.generate_with_context(
            query=query,
            context=context,
            system_message=system_message
        )
        
        return explanation
    
    def chat(self):
        """Interactive chat mode"""
        print("\n" + "="*60)
        print("Credit Risk Assessment Agent")
        print("="*60)
        print("Type 'quit' or 'exit' to end the conversation\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Get response
                result = self.query(user_input)
                
                print(f"\nAgent: {result['answer']}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize agent
    agent = CreditRiskAgent(
        model_path="models/autoencoder/default_autoencoder.pth",
        preprocessor_path="models/preprocessor/preprocessor.pkl",
        val_errors_path="results/validation_errors.npy",
        knowledge_base_path="knowledge_base",
        vector_db_path="models/vector_db/chroma_db"
    )
    
    # Example queries
    print("\n" + "="*60)
    print("EXAMPLE 1: Policy Question")
    print("="*60)
    result = agent.query("What are the lending criteria for high-risk customers?")
    print(f"\nAnswer: {result['answer']}")
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Decision Explanation")
    print("="*60)
    explanation = agent.explain_decision(
        risk_score=15.2,
        risk_category="LOW",
        action="APPROVE"
    )
    print(f"\nExplanation: {explanation}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Starting Interactive Mode")
    print("="*60)
    agent.chat()