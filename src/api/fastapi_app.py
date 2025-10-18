
"""
FastAPI Application - Credit Risk Agent API
RESTful API for credit risk assessment
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Import agent
from src.agent.credit_risk_agent import CreditRiskAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# PYDANTIC MODELS (API Schemas)
# ═══════════════════════════════════════════════════════════

class CustomerData(BaseModel):
    """Customer data schema"""
    NET_RENTAL: float = Field(..., description="Monthly payment amount")
    NO_OF_RENTAL: int = Field(..., description="Total number of payments")
    PAID_RENTALS: int = Field(..., description="Number of payments made")
    CB_ARREARS_AGE: float = Field(0, description="Days overdue")
    YOM: int = Field(..., description="Vehicle year of manufacture")
    FINANCE_AMOUNT: float = Field(..., description="Loan amount")
    CUSTOMER_VALUATION: float = Field(..., description="Asset value")
    EFFECTIVE_RATE: float = Field(..., description="Interest rate")
    AGE: int = Field(..., description="Customer age")
    INCOME: float = Field(..., description="Annual income")
    EXPENSE: float = Field(..., description="Annual expenses")
    
    # Add categorical features (simplified for API)
    GENDER: Optional[str] = Field("M", description="Gender")
    MARITAL_STATUS: Optional[str] = Field("Single", description="Marital status")
    
    @validator('AGE')
    def validate_age(cls, v):
        if v < 18 or v > 100:
            raise ValueError('Age must be between 18 and 100')
        return v
    
    @validator('INCOME')
    def validate_income(cls, v):
        if v < 0:
            raise ValueError('Income must be positive')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "NET_RENTAL": 10000.0,
                "NO_OF_RENTAL": 36,
                "PAID_RENTALS": 12,
                "CB_ARREARS_AGE": 0.0,
                "YOM": 2020,
                "FINANCE_AMOUNT": 300000.0,
                "CUSTOMER_VALUATION": 350000.0,
                "EFFECTIVE_RATE": 8.5,
                "AGE": 32,
                "INCOME": 55000.0,
                "EXPENSE": 30000.0,
                "GENDER": "M",
                "MARITAL_STATUS": "Single"
            }
        }


class QueryRequest(BaseModel):
    """Query request schema"""
    question: str = Field(..., description="User question", min_length=5)
    include_context: bool = Field(True, description="Include context in response")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What are the lending criteria for high-risk customers?",
                "include_context": True
            }
        }


class RiskAssessmentResponse(BaseModel):
    """Risk assessment response schema"""
    risk_score: float
    risk_category: str
    action: str
    confidence: float
    reconstruction_error: float
    timestamp: str
    explanation: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response schema"""
    question: str
    answer: str
    success: bool
    timestamp: str
    execution_time: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    agent_loaded: bool
    timestamp: str


# ═══════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="Credit Risk Assessment API",
    description="AI-powered credit risk assessment system with LLM agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent: Optional[CreditRiskAgent] = None


# ═══════════════════════════════════════════════════════════
# STARTUP/SHUTDOWN EVENTS
# ═══════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent
    
    logger.info("Starting Credit Risk Agent API...")
    
    try:
        agent = CreditRiskAgent(
            model_path="models/autoencoder/default_autoencoder.pth",
            preprocessor_path="models/preprocessor/preprocessor.pkl",
            val_errors_path="results/validation_errors.npy",
            knowledge_base_path="knowledge_base",
            vector_db_path="models/vector_db/chroma_db"
        )
        logger.info("Agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        agent = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Credit Risk Agent API...")


# ═══════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Assessment API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if agent is not None else "degraded",
        agent_loaded=agent is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/assess", response_model=RiskAssessmentResponse)
async def assess_risk(
    customer: CustomerData,
    include_explanation: bool = True,
    background_tasks: BackgroundTasks = None
):
    """
    Assess customer credit risk
    
    Args:
        customer: Customer data
        include_explanation: Include LLM explanation
        
    Returns:
        Risk assessment result
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Convert to dict
        customer_dict = customer.dict()
        
        # Assess risk
        import time
        start_time = time.time()
        
        result = agent.assess_customer(customer_dict)
        
        execution_time = time.time() - start_time
        
        # Extract risk information from result
        # Note: This is simplified - adjust based on your actual response format
        response_text = result['answer']
        
        # Parse risk score from response (simplified)
        # In practice, you'd want more robust parsing
        risk_score = 15.0  # Placeholder - extract from actual result
        risk_category = "LOW"  # Placeholder
        action = "APPROVE"  # Placeholder
        
        # Generate explanation if requested
        explanation = None
        if include_explanation and result.get('success'):
            explanation = agent.explain_decision(
                risk_score=risk_score,
                risk_category=risk_category,
                action=action
            )
        
        # Log assessment (background task)
        if background_tasks:
            background_tasks.add_task(
                log_assessment,
                customer_dict,
                risk_score,
                risk_category
            )
        
        return RiskAssessmentResponse(
            risk_score=risk_score,
            risk_category=risk_category,
            action=action,
            confidence=0.85,
            reconstruction_error=100.5,
            timestamp=datetime.now().isoformat(),
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_agent(query: QueryRequest):
    """
    Query the agent with natural language
    
    Args:
        query: User question
        
    Returns:
        Agent response
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        import time
        start_time = time.time()
        
        result = agent.query(query.question)
        
        execution_time = time.time() - start_time
        
        return QueryResponse(
            question=query.question,
            answer=result['answer'],
            success=result['success'],
            timestamp=datetime.now().isoformat(),
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain_decision(
    risk_score: float = Field(..., ge=0, le=100),
    risk_category: str = Field(..., regex="^(LOW|MEDIUM|HIGH)$"),
    action: str = Field(..., regex="^(APPROVE|APPROVE_WITH_CONDITIONS|REJECT)$")
):
    """
    Generate explanation for a decision
    
    Args:
        risk_score: Risk score (0-100)
        risk_category: Risk category
        action: Recommended action
        
    Returns:
        Natural language explanation
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        explanation = agent.explain_decision(
            risk_score=risk_score,
            risk_category=risk_category,
            action=action
        )
        
        return {
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/policies")
async def search_policies(
    query: str = Field(..., min_length=3),
    category: Optional[str] = None
):
    """
    Search knowledge base for policies
    
    Args:
        query: Search query
        category: Filter by category (optional)
        
    Returns:
        Relevant policy documents
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        if category:
            context = agent.rag_system.get_context_for_query(
                query,
                categories=[category]
            )
        else:
            context = agent.rag_system.get_context_for_query(query)
        
        return {
            "query": query,
            "results": context,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Policy search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-assess")
async def batch_assess(customers: List[CustomerData]):
    """
    Batch risk assessment for multiple customers
    
    Args:
        customers: List of customer data
        
    Returns:
        List of risk assessments
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if len(customers) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 customers per batch"
        )
    
    try:
        results = []
        
        for customer in customers:
            result = agent.assess_customer(customer.dict())
            results.append({
                "customer": customer.dict(),
                "assessment": result
            })
        
        return {
            "total": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def log_assessment(customer_data: Dict, risk_score: float, risk_category: str):
    """Log assessment for analytics (background task)"""
    logger.info(f"Assessment logged: Score={risk_score}, Category={risk_category}")
    # In production: Save to database


# ═══════════════════════════════════════════════════════════
# EXCEPTION HANDLERS
# ═══════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


# ═══════════════════════════════════════════════════════════
# RUN APPLICATION
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )