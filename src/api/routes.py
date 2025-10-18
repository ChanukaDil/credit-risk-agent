"""
API Routes for Credit Risk Agent
Modular route definitions (alternative to all-in-one fastapi_app.py)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List
import logging

from .schemas import (
    CustomerData,
    QueryRequest,
    RiskAssessmentResponse,
    QueryResponse
)

logger = logging.getLogger(__name__)

# Create routers
assessment_router = APIRouter(prefix="/assessment", tags=["Risk Assessment"])
query_router = APIRouter(prefix="/query", tags=["Agent Queries"])
policy_router = APIRouter(prefix="/policies", tags=["Policy Search"])


# ═══════════════════════════════════════════════════════════════
# RISK ASSESSMENT ROUTES
# ═══════════════════════════════════════════════════════════════

@assessment_router.post("/assess", response_model=RiskAssessmentResponse)
async def assess_customer(
    customer: CustomerData,
    include_explanation: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    Assess customer credit risk
    
    **Parameters:**
    - customer: Customer data with all required fields
    - include_explanation: Include LLM-generated explanation
    
    **Returns:**
    - Risk assessment with score, category, and recommendation
    """
    try:
        # Note: This would use the agent instance
        # In practice, agent would be passed via dependency injection
        
        logger.info("Risk assessment requested")
        
        # Placeholder response
        # In actual implementation, call: agent.assess_customer(customer.dict())
        
        return RiskAssessmentResponse(
            risk_score=15.2,
            risk_category="LOW",
            action="APPROVE",
            confidence=0.85,
            reconstruction_error=100.5,
            timestamp="2025-10-17T12:00:00",
            explanation="Customer profile indicates low risk..." if include_explanation else None
        )
        
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@assessment_router.post("/batch-assess")
async def batch_assess_customers(
    customers: List[CustomerData],
    max_batch_size: int = 100
):
    """
    Batch risk assessment for multiple customers
    
    **Parameters:**
    - customers: List of customer data
    - max_batch_size: Maximum customers per batch
    
    **Returns:**
    - List of risk assessments
    """
    if len(customers) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {max_batch_size}"
        )
    
    try:
        results = []
        
        for i, customer in enumerate(customers):
            # Process each customer
            result = {
                "customer_index": i,
                "customer_data": customer.dict(),
                "assessment": {
                    "risk_score": 15.0,  # Placeholder
                    "risk_category": "LOW",
                    "action": "APPROVE"
                }
            }
            results.append(result)
        
        return {
            "total_processed": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# AGENT QUERY ROUTES
# ═══════════════════════════════════════════════════════════════

@query_router.post("/ask", response_model=QueryResponse)
async def query_agent(query: QueryRequest):
    """
    Ask the agent a question using natural language
    
    **Parameters:**
    - question: User question
    - include_context: Include retrieved context in response
    
    **Returns:**
    - Agent's response with metadata
    """
    try:
        logger.info(f"Query received: {query.question[:50]}...")
        
        # In actual implementation: result = agent.query(query.question)
        
        return QueryResponse(
            question=query.question,
            answer="Based on our lending policies...",  # Placeholder
            success=True,
            timestamp="2025-10-17T12:00:00",
            execution_time=2.5
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@query_router.post("/explain")
async def explain_decision(
    risk_score: float,
    risk_category: str,
    action: str
):
    """
    Generate explanation for a credit decision
    
    **Parameters:**
    - risk_score: Risk score (0-100)
    - risk_category: LOW/MEDIUM/HIGH
    - action: APPROVE/CONDITIONAL/REJECT
    
    **Returns:**
    - Natural language explanation
    """
    # Validate inputs
    if not 0 <= risk_score <= 100:
        raise HTTPException(
            status_code=400,
            detail="Risk score must be between 0 and 100"
        )
    
    if risk_category not in ["LOW", "MEDIUM", "HIGH"]:
        raise HTTPException(
            status_code=400,
            detail="Risk category must be LOW, MEDIUM, or HIGH"
        )
    
    try:
        # In actual implementation:
        # explanation = agent.explain_decision(risk_score, risk_category, action)
        
        explanation = f"""
Based on the risk assessment:

**Decision:** {action}
**Risk Level:** {risk_category} ({risk_score}/100)

**Explanation:**
The customer's risk profile indicates {risk_category.lower()} credit risk...
        """.strip()
        
        return {
            "explanation": explanation,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "action": action
        }
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# POLICY SEARCH ROUTES
# ═══════════════════════════════════════════════════════════════

@policy_router.get("/search")
async def search_policies(
    query: str,
    category: str = None,
    top_k: int = 5
):
    """
    Search knowledge base for policies
    
    **Parameters:**
    - query: Search query
    - category: Filter by category (optional)
    - top_k: Number of results to return
    
    **Returns:**
    - Relevant policy documents
    """
    if len(query) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 3 characters"
        )
    
    try:
        # In actual implementation:
        # results = agent.rag_system.get_context_for_query(query, categories=[category] if category else None)
        
        results = f"""
**Relevant Policies for: "{query}"**

[1] Lending Policy Section 3.2
Content: Standard approval criteria...

[2] Risk Assessment Guidelines
Content: Risk categorization...
        """.strip()
        
        return {
            "query": query,
            "category": category,
            "results": results,
            "count": 2
        }
        
    except Exception as e:
        logger.error(f"Policy search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@policy_router.get("/categories")
async def list_policy_categories():
    """
    List available policy categories
    
    **Returns:**
    - List of category names
    """
    categories = [
        "bank_policies",
        "regulations",
        "case_studies",
        "faq"
    ]
    
    return {
        "categories": categories,
        "count": len(categories)
    }


@policy_router.get("/similar-cases")
async def find_similar_cases(
    customer_profile: str,
    num_cases: int = 3
):
    """
    Find similar historical customer cases
    
    **Parameters:**
    - customer_profile: Customer description
    - num_cases: Number of cases to return
    
    **Returns:**
    - Similar cases with outcomes
    """
    try:
        # In actual implementation:
        # cases = agent.rag_system.search_similar_cases(customer_profile, k=num_cases)
        
        cases = f"""
**Similar Cases:**

[1] Case: Young professional, age 28
Outcome: Approved with standard terms

[2] Case: First-time borrower, age 25
Outcome: Approved with co-signer

[3] Case: Self-employed, age 35
Outcome: Approved with conditions
        """.strip()
        
        return {
            "query": customer_profile,
            "cases": cases,
            "count": num_cases
        }
        
    except Exception as e:
        logger.error(f"Similar cases error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# COMBINE ALL ROUTERS
# ═══════════════════════════════════════════════════════════════

def get_all_routers() -> List[APIRouter]:
    """
    Get all API routers
    
    Returns:
        List of routers to include in main app
    """
    return [
        assessment_router,
        query_router,
        policy_router
    ]


# ═══════════════════════════════════════════════════════════════
# USAGE IN MAIN APP
# ═══════════════════════════════════════════════════════════════

"""
In your main fastapi_app.py, include these routes like this:

from fastapi import FastAPI
from src.api.routes import get_all_routers

app = FastAPI()

# Include all routers
for router in get_all_routers():
    app.include_router(router)

Now you'll have:
- /assessment/assess
- /assessment/batch-assess
- /query/ask
- /query/explain
- /policies/search
- /policies/categories
- /policies/similar-cases
"""