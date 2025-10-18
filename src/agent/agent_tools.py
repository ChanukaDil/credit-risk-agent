

"""
Agent Tools - LangChain Tool Definitions
Integrates: Autoencoder Risk Scorer + RAG System + Business Logic
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# LangChain imports
from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

# Import existing components
from src.risk_scoring import CreditRiskScorer
from src.agent.rag_system import RAGSystem

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# TOOL 1: RISK ASSESSMENT TOOL
# ═══════════════════════════════════════════════════════════════

class RiskAssessmentInput(BaseModel):
    """Input schema for risk assessment"""
    customer_data: str = Field(
        description="Customer data as JSON string with all required features"
    )


class RiskAssessmentTool:
    """Tool for assessing customer credit risk using autoencoder"""
    
    def __init__(
        self,
        model_path: str = "models/autoencoder/default_autoencoder.pth",
        preprocessor_path: str = "models/preprocessor/preprocessor.pkl",
        val_errors_path: str = "results/validation_errors.npy"
    ):
        """Initialize risk assessment tool"""
        logger.info("Initializing Risk Assessment Tool")
        
        # Load risk scorer
        self.scorer = CreditRiskScorer(
            model_path=model_path,
            preprocessor_path=preprocessor_path
        )
        
        # Calibrate
        val_errors = np.load(val_errors_path)
        self.scorer.calibrate_error_range(val_errors)
        
        logger.info("Risk Assessment Tool ready")
    
    def assess_risk(self, customer_data: str) -> str:
        """
        Assess customer credit risk
        
        Args:
            customer_data: Customer data as JSON string
            
        Returns:
            Risk assessment result as formatted string
        """
        try:
            # Parse customer data (simplified - in practice, handle JSON)
            # For now, assume we get a dictionary
            import json
            customer_dict = json.loads(customer_data)
            
            # Convert to DataFrame
            customer_df = pd.DataFrame([customer_dict])
            
            # Get prediction
            result = self.scorer.predict(customer_df)
            
            # Format response
            response = f"""
Risk Assessment Result:
━━━━━━━━━━━━━━━━━━━━━━
Risk Score: {result['risk_score']:.2f}/100
Risk Category: {result['risk_category']}
Recommended Action: {result['action']}
Confidence: {result['confidence']:.2f}

Reconstruction Error: {result['reconstruction_error']:.2f}
Threshold: {result['threshold']:.2f}

Business Recommendation:
{self._get_business_recommendation(result)}
"""
            return response
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return f"Error assessing risk: {str(e)}"
    
    def _get_business_recommendation(self, result: Dict) -> str:
        """Generate business recommendation based on risk result"""
        category = result['risk_category']
        
        recommendations = {
            'LOW': (
                "✅ APPROVE with standard terms\n"
                "- Standard interest rate\n"
                "- Normal documentation required\n"
                "- Monthly payment monitoring"
            ),
            'MEDIUM': (
                "⚠️ APPROVE WITH CONDITIONS\n"
                "- Higher interest rate (+2-3%)\n"
                "- Additional documentation required\n"
                "- Guarantor may be needed\n"
                "- Enhanced monitoring"
            ),
            'HIGH': (
                "❌ REJECT or REQUIRE MAJOR CONDITIONS\n"
                "- High default probability\n"
                "- Consider: larger down payment, co-signer\n"
                "- Alternative products may be suitable"
            )
        }
        
        return recommendations.get(category, "Unknown risk category")


# ═══════════════════════════════════════════════════════════════
# TOOL 2: POLICY RETRIEVAL TOOL
# ═══════════════════════════════════════════════════════════════

class PolicyRetrievalInput(BaseModel):
    """Input schema for policy retrieval"""
    query: str = Field(description="Query about bank policies or regulations")
    category: Optional[str] = Field(
        default=None,
        description="Specific category: bank_policies, regulations, case_studies, faq"
    )


class PolicyRetrievalTool:
    """Tool for retrieving relevant policies and regulations"""
    
    def __init__(self, rag_system: RAGSystem):
        """Initialize policy retrieval tool"""
        self.rag = rag_system
        logger.info("Policy Retrieval Tool ready")
    
    def retrieve_policy(self, query: str, category: Optional[str] = None) -> str:
        """
        Retrieve relevant policies
        
        Args:
            query: Policy question
            category: Filter by category
            
        Returns:
            Formatted policy information
        """
        try:
            if category:
                context = self.rag.get_context_for_query(query, categories=[category])
            else:
                context = self.rag.get_context_for_query(query)
            
            if not context or context == "No relevant information found.":
                return "No relevant policies found for this query."
            
            response = f"""
Relevant Policy Information:
━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
"""
            return response
            
        except Exception as e:
            logger.error(f"Policy retrieval error: {e}")
            return f"Error retrieving policies: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# TOOL 3: SIMILAR CASES TOOL
# ═══════════════════════════════════════════════════════════════

class SimilarCasesInput(BaseModel):
    """Input schema for similar cases search"""
    customer_profile: str = Field(
        description="Customer profile description"
    )
    num_cases: int = Field(
        default=3,
        description="Number of similar cases to retrieve"
    )


class SimilarCasesTool:
    """Tool for finding similar customer cases"""
    
    def __init__(self, rag_system: RAGSystem):
        """Initialize similar cases tool"""
        self.rag = rag_system
        logger.info("Similar Cases Tool ready")
    
    def find_similar_cases(
        self,
        customer_profile: str,
        num_cases: int = 3
    ) -> str:
        """
        Find similar customer cases
        
        Args:
            customer_profile: Customer description
            num_cases: Number of cases to retrieve
            
        Returns:
            Formatted similar cases
        """
        try:
            cases = self.rag.search_similar_cases(customer_profile, k=num_cases)
            
            if not cases or cases == "No relevant information found.":
                return "No similar cases found."
            
            response = f"""
Similar Customer Cases:
━━━━━━━━━━━━━━━━━━━━━━━
{cases}
"""
            return response
            
        except Exception as e:
            logger.error(f"Similar cases search error: {e}")
            return f"Error finding similar cases: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# TOOL FACTORY: CREATE LANGCHAIN TOOLS
# ═══════════════════════════════════════════════════════════════

def create_agent_tools(
    model_path: str = "models/autoencoder/default_autoencoder.pth",
    preprocessor_path: str = "models/preprocessor/preprocessor.pkl",
    val_errors_path: str = "results/validation_errors.npy",
    rag_system: Optional[RAGSystem] = None
) -> list:
    """
    Create all LangChain tools for the agent
    
    Args:
        model_path: Path to autoencoder model
        preprocessor_path: Path to preprocessor
        val_errors_path: Path to validation errors
        rag_system: Initialized RAG system (optional, will create if not provided)
        
    Returns:
        List of LangChain tools
    """
    logger.info("Creating agent tools")
    
    # Initialize components
    risk_tool = RiskAssessmentTool(model_path, preprocessor_path, val_errors_path)
    
    if rag_system is None:
        rag_system = RAGSystem()
    
    policy_tool = PolicyRetrievalTool(rag_system)
    cases_tool = SimilarCasesTool(rag_system)
    
    # Create LangChain tools
    tools = [
        Tool(
            name="assess_credit_risk",
            description=(
                "Assess customer credit risk using the autoencoder model. "
                "Input should be customer data as JSON string with all required features. "
                "Returns risk score (0-100), category (LOW/MEDIUM/HIGH), and business recommendation."
            ),
            func=risk_tool.assess_risk
        ),
        
        Tool(
            name="retrieve_bank_policy",
            description=(
                "Retrieve relevant bank policies, lending guidelines, or regulations. "
                "Input should be a question about policies. "
                "Returns formatted policy information from the knowledge base."
            ),
            func=policy_tool.retrieve_policy
        ),
        
        Tool(
            name="find_similar_cases",
            description=(
                "Find similar customer cases from historical data. "
                "Input should be a customer profile description. "
                "Returns similar cases with outcomes and decisions."
            ),
            func=cases_tool.find_similar_cases
        ),
    ]
    
    logger.info(f"Created {len(tools)} tools")
    return tools


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create tools
    tools = create_agent_tools()
    
    # Test risk assessment
    print("\n" + "="*60)
    print("Testing Risk Assessment Tool")
    print("="*60)
    
    sample_customer = {
        "NET_RENTAL": 10000,
        "AGE": 32,
        "INCOME": 55000,
        "FINANCE_AMOUNT": 300000,
        # ... add all required features
    }
    
    import json
    result = tools[0].func(json.dumps(sample_customer))
    print(result)
    
    # Test policy retrieval
    print("\n" + "="*60)
    print("Testing Policy Retrieval Tool")
    print("="*60)
    
    policy_result = tools[1].func("What are the lending criteria?")
    print(policy_result)