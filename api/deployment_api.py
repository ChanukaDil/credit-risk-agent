"""
FastAPI Deployment for Credit Risk Prediction
Production-ready API with monitoring and logging
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import json
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Credit Risk Prediction API",
    description="Production API for credit default risk assessment using XGBoost",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
MODEL = None
FEATURE_ENGINEER = None
THRESHOLD = None
METADATA = None
PREDICTION_LOG = []


# ============================================================================
# Pydantic Models
# ============================================================================

class LoanApplication(BaseModel):
    """Input data for a single loan application"""
    NET_RENTAL: float = Field(..., description="Monthly rental payment")
    NO_OF_RENTAL: int = Field(..., description="Total number of rental payments")
    PAID_RENTALS: int = Field(..., description="Number of rentals already paid")
    CB_ARREARS_AGE: float = Field(0, description="Credit bureau arrears age in days")
    YOM: int = Field(..., description="Year of manufacture of vehicle")
    FINANCE_AMOUNT: float = Field(..., description="Total loan amount")
    CUSTOMER_VALUATION: float = Field(..., description="Customer's valuation of asset")
    EFFECTIVE_RATE: float = Field(..., description="Effective interest rate")
    AGE: int = Field(..., description="Customer age")
    INCOME: float = Field(..., description="Monthly income")
    EXPENSE: float = Field(..., description="Monthly expenses")
    PRODUCT_CODE_encoded: float = 0
    PRODUCT_NAME_encoded: float = 0
    PRODUCT_CATEGORY_encoded: float = 0
    CONTRACT_NO_encoded: float = 0
    CONTRACT_STATUS_encoded: float = 0
    CONTRACT_DATE_encoded: float = 0
    RECOVERY_STATUS_encoded: float = 0
    LAST_PAYMENT_DATE_encoded: float = 0
    DUE_FREQUENCY_encoded: float = 0
    ASSET_TYPE_NAME_encoded: float = 0
    MAKE_encoded: float = 0
    MODEL_NAME_encoded: float = 0
    REGISTRATION_encoded: float = 0
    REGISTRATION_NO_encoded: float = 0
    GENDER_encoded: float = 0
    CITY_encoded: float = 0
    DISTRICT_NAME_encoded: float = 0
    PROVINCE_NAME_encoded: float = 0
    MARITAL_STATUS_encoded: float = 0
    
    class Config:
        schema_extra = {
            "example": {
                "NET_RENTAL": 15000.0,
                "NO_OF_RENTAL": 60,
                "PAID_RENTALS": 12,
                "CB_ARREARS_AGE": 0,
                "YOM": 2020,
                "FINANCE_AMOUNT": 800000.0,
                "CUSTOMER_VALUATION": 1000000.0,
                "EFFECTIVE_RATE": 12.5,
                "AGE": 35,
                "INCOME": 75000.0,
                "EXPENSE": 40000.0,
                "PRODUCT_CODE_encoded": 0,
                "PRODUCT_NAME_encoded": 0,
                "PRODUCT_CATEGORY_encoded": 0,
                "CONTRACT_NO_encoded": 0,
                "CONTRACT_STATUS_encoded": 0,
                "CONTRACT_DATE_encoded": 0,
                "RECOVERY_STATUS_encoded": 0,
                "LAST_PAYMENT_DATE_encoded": 0,
                "DUE_FREQUENCY_encoded": 0,
                "ASSET_TYPE_NAME_encoded": 0,
                "MAKE_encoded": 0,
                "MODEL_NAME_encoded": 0,
                "REGISTRATION_encoded": 0,
                "REGISTRATION_NO_encoded": 0,
                "GENDER_encoded": 0,
                "CITY_encoded": 0,
                "DISTRICT_NAME_encoded": 0,
                "PROVINCE_NAME_encoded": 0,
                "MARITAL_STATUS_encoded": 0
            }
        }


class BatchLoanApplications(BaseModel):
    """Batch of loan applications"""
    applications: List[LoanApplication]


class PredictionResponse(BaseModel):
    """Response for single prediction"""
    application_id: str
    prediction: str  # "DEFAULT" or "NO_DEFAULT"
    default_probability: float
    risk_score: float  # 0-100
    confidence: str  # "High", "Medium", "Low"
    recommendation: str
    risk_factors: List[str]
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    total_applications: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    predictions: List[PredictionResponse]
    processing_time_seconds: float


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    version: str
    accuracy: float
    recall: float
    precision: float
    f1_score: float
    roc_auc: float
    threshold: float
    total_features: int
    last_trained: str


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global MODEL, FEATURE_ENGINEER, THRESHOLD, METADATA
    
    logger.info("Loading models and metadata...")
    
    try:
        # Load model
        model_path = Path(__file__).parent.parent / 'models' / 'improved' / 'rank1_xgboost.pkl'
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        
        MODEL = model_info['model']
        THRESHOLD = model_info['threshold']
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Threshold: {THRESHOLD:.4f}")
        
        # Try to load feature engineer
        fe_path = Path(__file__).parent.parent / 'models' / 'improved' / 'feature_engineer.pkl'
        if fe_path.exists():
            with open(fe_path, 'rb') as f:
                fe_dict = pickle.load(f)
            
            from feature_engineering import CreditRiskFeatureEngineer
            FEATURE_ENGINEER = CreditRiskFeatureEngineer()
            FEATURE_ENGINEER.scaler = fe_dict['scaler']
            FEATURE_ENGINEER.feature_names = fe_dict['feature_names']
            FEATURE_ENGINEER.original_features = fe_dict['original_features']
            logger.info(f"✅ Feature engineer loaded")
        else:
            logger.info(f"⚠️  Feature engineer not found, using raw features")
        
        # Load metadata
        metadata_path = Path(__file__).parent.parent / 'data' / 'processed' / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                METADATA = json.load(f)
            logger.info(f"✅ Metadata loaded")
        
        logger.info("🚀 API ready to serve predictions!")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Save logs on shutdown"""
    logger.info("Shutting down API...")
    
    # Save prediction log
    log_path = Path(__file__).parent.parent / 'logs' / 'predictions'
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(log_path / f'predictions_{timestamp}.json', 'w') as f:
        json.dump(PREDICTION_LOG, f, indent=2)
    
    logger.info(f"💾 Prediction log saved: {len(PREDICTION_LOG)} predictions")


# ============================================================================
# Helper Functions
# ============================================================================

def preprocess_input(loan_app: LoanApplication) -> np.ndarray:
    """Convert loan application to model input"""
    # Convert to dict and then to DataFrame
    data_dict = loan_app.dict()
    
    # Create DataFrame with correct column order
    if METADATA:
        feature_names = METADATA['feature_names']
        df = pd.DataFrame([data_dict])[feature_names]
    else:
        df = pd.DataFrame([data_dict])
    
    # Convert to numpy array
    X = df.values.astype(np.float32)
    
    # Apply feature engineering if available
    if FEATURE_ENGINEER:
        X = FEATURE_ENGINEER.transform(X, feature_names)
    
    return X


def calculate_risk_factors(loan_app: LoanApplication, probability: float) -> List[str]:
    """Identify risk factors contributing to default risk"""
    risk_factors = []
    
    # 1. High debt-to-income
    if loan_app.FINANCE_AMOUNT / (loan_app.INCOME + 1) > 10:
        risk_factors.append("High debt-to-income ratio")
    
    # 2. High payment-to-income
    if loan_app.NET_RENTAL / (loan_app.INCOME + 1) > 0.3:
        risk_factors.append("High payment burden (>30% of income)")
    
    # 3. High expense-to-income
    if loan_app.EXPENSE / (loan_app.INCOME + 1) > 0.7:
        risk_factors.append("High expense ratio (>70% of income)")
    
    # 4. Credit bureau arrears
    if loan_app.CB_ARREARS_AGE > 30:
        risk_factors.append(f"Credit bureau arrears: {loan_app.CB_ARREARS_AGE:.0f} days")
    
    # 5. Low payment completion
    payment_rate = loan_app.PAID_RENTALS / (loan_app.NO_OF_RENTAL + 1)
    if payment_rate < 0.3:
        risk_factors.append(f"Low payment completion: {payment_rate*100:.0f}%")
    
    # 6. Old vehicle
    vehicle_age = 2025 - loan_app.YOM
    if vehicle_age > 10:
        risk_factors.append(f"Old vehicle: {vehicle_age} years")
    
    # 7. High interest rate
    if loan_app.EFFECTIVE_RATE > 15:
        risk_factors.append(f"High interest rate: {loan_app.EFFECTIVE_RATE:.1f}%")
    
    # 8. Young borrower
    if loan_app.AGE < 25:
        risk_factors.append("Young borrower (under 25)")
    
    # 9. Low loan-to-value
    ltv = loan_app.FINANCE_AMOUNT / (loan_app.CUSTOMER_VALUATION + 1)
    if ltv > 0.9:
        risk_factors.append(f"High loan-to-value: {ltv*100:.0f}%")
    
    # If no specific factors but high probability
    if not risk_factors and probability > 0.5:
        risk_factors.append("Multiple minor factors combine to increase risk")
    
    if not risk_factors:
        risk_factors.append("Low risk profile detected")
    
    return risk_factors


def generate_recommendation(probability: float, risk_factors: List[str]) -> str:
    """Generate business recommendation"""
    if probability >= 0.7:
        return "🔴 HIGH RISK: Reject application or request additional collateral/guarantor"
    elif probability >= 0.4:
        return "🟡 MEDIUM RISK: Approve with caution - consider higher down payment or lower loan amount"
    else:
        return "🟢 LOW RISK: Approve application with standard terms"


def log_prediction(app_id: str, input_data: Dict, output: Dict):
    """Log prediction for monitoring"""
    log_entry = {
        'application_id': app_id,
        'timestamp': datetime.now().isoformat(),
        'input': input_data,
        'output': output
    }
    PREDICTION_LOG.append(log_entry)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Credit Risk Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "feature_engineer_loaded": FEATURE_ENGINEER is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Load results
    results_path = Path(__file__).parent.parent / 'models' / 'improved' / 'model_results.json'
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        xgboost_results = results.get('XGBoost', {})
    else:
        xgboost_results = {}
    
    total_features = len(FEATURE_ENGINEER.feature_names) if FEATURE_ENGINEER else 30
    
    return ModelInfo(
        model_name="XGBoost",
        version="1.0.0",
        accuracy=xgboost_results.get('accuracy', 0.9886),
        recall=xgboost_results.get('recall', 0.725),
        precision=xgboost_results.get('precision', 0.4574),
        f1_score=xgboost_results.get('f1_score', 0.5609),
        roc_auc=xgboost_results.get('roc_auc', 0.9654),
        threshold=THRESHOLD,
        total_features=total_features,
        last_trained="2025-11-14"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    loan_app: LoanApplication,
    background_tasks: BackgroundTasks
):
    """
    Predict default risk for a single loan application
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate application ID
        app_id = f"APP_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        # Preprocess input
        X = preprocess_input(loan_app)
        
        # Get prediction probability
        probability = float(MODEL.predict_proba(X)[0, 1])
        
        # Apply threshold
        prediction = "DEFAULT" if probability >= THRESHOLD else "NO_DEFAULT"
        
        # Calculate risk score (0-100)
        risk_score = probability * 100
        
        # Determine confidence
        if probability >= 0.8 or probability <= 0.2:
            confidence = "High"
        elif probability >= 0.6 or probability <= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Get risk factors
        risk_factors = calculate_risk_factors(loan_app, probability)
        
        # Generate recommendation
        recommendation = generate_recommendation(probability, risk_factors)
        
        # Create response
        response = PredictionResponse(
            application_id=app_id,
            prediction=prediction,
            default_probability=round(probability, 4),
            risk_score=round(risk_score, 2),
            confidence=confidence,
            recommendation=recommendation,
            risk_factors=risk_factors,
            timestamp=datetime.now().isoformat()
        )
        
        # Log prediction in background
        background_tasks.add_task(
            log_prediction,
            app_id,
            loan_app.dict(),
            response.dict()
        )
        
        logger.info(f"Prediction made: {app_id} - {prediction} ({probability:.4f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch: BatchLoanApplications,
    background_tasks: BackgroundTasks
):
    """
    Predict default risk for multiple loan applications
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        predictions = []
        high_risk = medium_risk = low_risk = 0
        
        for loan_app in batch.applications:
            # Preprocess
            X = preprocess_input(loan_app)
            
            # Predict
            probability = float(MODEL.predict_proba(X)[0, 1])
            prediction = "DEFAULT" if probability >= THRESHOLD else "NO_DEFAULT"
            
            # Risk categorization
            if probability >= 0.7:
                high_risk += 1
            elif probability >= 0.4:
                medium_risk += 1
            else:
                low_risk += 1
            
            # Create response
            app_id = f"APP_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            risk_score = probability * 100
            
            if probability >= 0.8 or probability <= 0.2:
                confidence = "High"
            elif probability >= 0.6 or probability <= 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            risk_factors = calculate_risk_factors(loan_app, probability)
            recommendation = generate_recommendation(probability, risk_factors)
            
            pred_response = PredictionResponse(
                application_id=app_id,
                prediction=prediction,
                default_probability=round(probability, 4),
                risk_score=round(risk_score, 2),
                confidence=confidence,
                recommendation=recommendation,
                risk_factors=risk_factors,
                timestamp=datetime.now().isoformat()
            )
            
            predictions.append(pred_response)
            
            # Log in background
            background_tasks.add_task(
                log_prediction,
                app_id,
                loan_app.dict(),
                pred_response.dict()
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Batch prediction: {len(predictions)} applications processed in {processing_time:.2f}s")
        
        return BatchPredictionResponse(
            total_applications=len(predictions),
            high_risk_count=high_risk,
            medium_risk_count=medium_risk,
            low_risk_count=low_risk,
            predictions=predictions,
            processing_time_seconds=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/stats")
async def get_statistics():
    """Get prediction statistics"""
    if not PREDICTION_LOG:
        return {
            "total_predictions": 0,
            "message": "No predictions made yet"
        }
    
    total = len(PREDICTION_LOG)
    defaults = sum(1 for p in PREDICTION_LOG if p['output']['prediction'] == 'DEFAULT')
    no_defaults = total - defaults
    
    avg_risk_score = np.mean([p['output']['risk_score'] for p in PREDICTION_LOG])
    
    return {
        "total_predictions": total,
        "default_predictions": defaults,
        "no_default_predictions": no_defaults,
        "default_rate": round(defaults / total * 100, 2) if total > 0 else 0,
        "average_risk_score": round(avg_risk_score, 2)
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "deployment_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
