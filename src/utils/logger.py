"""
Logging Utility
Centralized logging configuration for the credit risk agent
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = "credit_risk_agent",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "results/logs",
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with console and file handlers
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file name (None = auto-generate)
        log_dir: Directory for log files
        console_output: Enable console logging
        file_output: Enable file logging
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files to keep
        format_string: Custom log format
        
    Returns:
        Configured logger
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d')
            log_file = f"{name}_{timestamp}.log"
        
        log_filepath = log_path / log_file
        
        # Rotating file handler
        file_handler = RotatingFileHandler(
            log_filepath,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized: {name} (level: {log_level})")
    
    return logger


def get_logger(name: str = "credit_risk_agent") -> logging.Logger:
    """
    Get existing logger or create new one
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If no handlers, set up default logger
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class AgentLogger:
    """
    Custom logger for agent operations with structured logging
    """
    
    def __init__(self, name: str = "credit_risk_agent"):
        """Initialize agent logger"""
        self.logger = get_logger(name)
        self.session_id = None
    
    def set_session(self, session_id: str):
        """Set current session ID for logging"""
        self.session_id = session_id
    
    def _format_message(self, message: str) -> str:
        """Add session info to message"""
        if self.session_id:
            return f"[Session: {self.session_id}] {message}"
        return message
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(self._format_message(message), extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(self._format_message(message), extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(self._format_message(message), extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(self._format_message(message), extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(self._format_message(message), extra=kwargs)
    
    def log_query(self, query: str, response: str, metadata: dict = None):
        """Log agent query and response"""
        log_entry = {
            'type': 'query',
            'query': query,
            'response_length': len(response),
            'metadata': metadata or {}
        }
        self.info(f"Query processed: {query[:50]}...", **log_entry)
    
    def log_risk_assessment(
        self,
        risk_score: float,
        risk_category: str,
        action: str,
        customer_id: Optional[str] = None
    ):
        """Log risk assessment"""
        log_entry = {
            'type': 'risk_assessment',
            'risk_score': risk_score,
            'risk_category': risk_category,
            'action': action,
            'customer_id': customer_id
        }
        self.info(
            f"Risk assessment: {risk_category} (score: {risk_score})",
            **log_entry
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        self.error(
            f"Error in {context}: {str(error)}",
            exc_info=True
        )


def setup_production_logging(
    app_name: str = "credit_risk_agent",
    log_dir: str = "results/logs"
):
    """
    Set up production-grade logging with multiple handlers
    
    Args:
        app_name: Application name
        log_dir: Log directory
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Main application logger
    app_logger = setup_logger(
        name=app_name,
        log_level="INFO",
        log_file=f"{app_name}.log",
        log_dir=log_dir
    )
    
    # Error logger (separate file for errors only)
    error_logger = logging.getLogger(f"{app_name}.errors")
    error_logger.setLevel(logging.ERROR)
    
    error_handler = RotatingFileHandler(
        log_path / f"{app_name}_errors.log",
        maxBytes=10485760,
        backupCount=10
    )
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    error_logger.addHandler(error_handler)
    
    # Access logger (for API requests)
    access_logger = logging.getLogger(f"{app_name}.access")
    access_logger.setLevel(logging.INFO)
    
    access_handler = TimedRotatingFileHandler(
        log_path / f"{app_name}_access.log",
        when='midnight',
        interval=1,
        backupCount=30
    )
    access_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s'
    ))
    access_logger.addHandler(access_handler)
    
    return {
        'app': app_logger,
        'error': error_logger,
        'access': access_logger
    }


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Basic setup
    logger = setup_logger(
        name="test_logger",
        log_level="DEBUG",
        log_file="test.log"
    )
    
    # Test logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Agent logger
    agent_logger = AgentLogger("test_agent")
    agent_logger.set_session("session_123")
    
    agent_logger.info("Agent started")
    agent_logger.log_query(
        query="What are the lending criteria?",
        response="The lending criteria include...",
        metadata={'source': 'policy_db'}
    )
    agent_logger.log_risk_assessment(
        risk_score=15.2,
        risk_category="LOW",
        action="APPROVE",
        customer_id="CUST_001"
    )
    
    print("\nLogs written to: results/logs/")