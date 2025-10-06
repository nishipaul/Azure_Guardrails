"""
Optimized PII Detection Service
==============================

A high-performance, concurrent-ready PII detection service using a two-stage validation approach:
1. DetectPII - First level PII detection
2. GuardrailsPII - Second level comprehensive PII validation

Features:
- Concurrent processing for up to 100+ users
- Comprehensive logging with request tracking
- Optimized resource management
- Proper exception handling
- Performance monitoring
- Thread-safe operations

Author: Your Name
Date: October 2025
Version: 2.0.0
"""

import asyncio
import logging
import time
import uuid
import warnings
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading
from contextlib import contextmanager
import queue
import json
from guardrails import Guard
from guardrails.hub import DetectPII, GuardrailsPII




# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="guardrails")
warnings.filterwarnings("ignore", module="huggingface_hub")
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="pandas")

# Suppress specific library warnings and environment settings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizers warning
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Reduce transformers logging
os.environ["DATASETS_VERBOSITY"] = "error"  # Reduce datasets logging

# Additional logging configuration to suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("guardrails").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)








@dataclass
class PIICheckResult:
    """Result class for PII check operations."""
    request_id: str
    tenant_id: str
    text: str
    is_safe: bool
    stage_failed: Optional[str]  # "detect_pii", "guardrails_pii", or None
    error_message: Optional[str]
    processing_time: float
    timestamp: str
    detected_entities: List[str]




@dataclass
class TenantPIIConfig:
    """Configuration class for tenant-specific PII settings."""
    tenant_id: str
    tenant_name: str
    log_level: str = "INFO"
    log_file_prefix: str = ""
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    enable_performance_monitoring: bool = True
    guardrails_entities: List[str] = None
    model_name: str = "urchade/gliner_small-v2.1"  # DEAFULT MODEL - CAN BE SWITCHED TO OPENAI
    
    def __post_init__(self):
        if self.guardrails_entities is None:
            self.guardrails_entities = [
                "CREDIT_CARD", "CRYPTO", "DATE_TIME", "EMAIL_ADDRESS", "IBAN_CODE", "IP_ADDRESS", "NRP", "LOCATION", "PERSON", "PHONE_NUMBER", "MEDICAL_LICENSE",
                "URL", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", "US_PASSPORT", "US_SSN", "UK_NHS", "ES_NIF", "ES_NIE", "IT_FISCAL_CODE", "IT_DRIVER_LICENSE",
                "IT_VAT_CODE", "IT_PASSPORT", "IT_IDENTITY_CARD", "PL_PESEL", "SG_NRIC_FIN", "SG_UEN", "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE", "IN_PAN", "IN_AADHAAR",
                "IN_VEHICLE_REGISTRATION", "IN_VOTER", "IN_PASSPORT", "FI_PERSONAL_IDENTITY_CODE"
            ]
        # Set log file prefix with tenant name in azure_guardrails format
        if not self.log_file_prefix:
            self.log_file_prefix = f"azure_guardrails_pii_{self.tenant_name}"







@dataclass
class PIIServiceConfig:
    """Legacy configuration for backward compatibility."""
    max_concurrent_requests: int = 100
    request_timeout: int = 30  # seconds
    log_level: str = "INFO"
    enable_performance_monitoring: bool = True
    guardrails_entities: List[str] = None
    model_name: str = "urchade/gliner_small-v2.1"   # DEAFULT MODEL - CAN BE SWITCHED TO OPENAI
    
    def __post_init__(self):
        if self.guardrails_entities is None:
            self.guardrails_entities = [
                "CREDIT_CARD", "CRYPTO", "DATE_TIME", "EMAIL_ADDRESS", "IBAN_CODE", "IP_ADDRESS", "NRP", "LOCATION", "PERSON", "PHONE_NUMBER", "MEDICAL_LICENSE",
                "URL", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", "US_PASSPORT", "US_SSN", "UK_NHS", "ES_NIF", "ES_NIE", "IT_FISCAL_CODE", "IT_DRIVER_LICENSE",
                "IT_VAT_CODE", "IT_PASSPORT", "IT_IDENTITY_CARD", "PL_PESEL", "SG_NRIC_FIN", "SG_UEN", "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE", "IN_PAN", "IN_AADHAAR",
                "IN_VEHICLE_REGISTRATION", "IN_VOTER", "IN_PASSPORT", "FI_PERSONAL_IDENTITY_CODE"]








class OptimizedPIIDetectionService:
    """
    High-performance PII Detection Service optimized for concurrent processing.
    
    This service implements a two-stage PII detection workflow:
    1. Stage 1: DetectPII - Fast initial screening
    2. Stage 2: GuardrailsPII - Comprehensive validation
    
    Features:
    - Thread-safe operations
    - Request queuing and rate limiting
    - Comprehensive logging with request tracking
    - Performance monitoring
    - Resource pooling for optimal performance
    - Tenant-based configuration and logging
    """
    
    def __init__(self, tenant_config: Optional[TenantPIIConfig] = None, config: Optional[PIIServiceConfig] = None):
        """
        Initialize the PII Detection Service.
        
        Args:
            tenant_config: Tenant-specific configuration object (preferred)
            config: Legacy service configuration object (for backward compatibility)
        """
        # Use tenant config if provided, otherwise fall back to legacy config
        if tenant_config:
            self.tenant_config = tenant_config
            self.config = self._convert_tenant_to_service_config(tenant_config)
            self.tenant_id = tenant_config.tenant_id
            self.tenant_name = tenant_config.tenant_name
        else:
            self.config = config or PIIServiceConfig()
            self.tenant_config = None
            self.tenant_id = "Simpplr_Tenant"
            self.tenant_name = "Simpplr"
            
        self.logger = self._setup_logger()
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_requests,
            thread_name_prefix="PII-Worker"
        )
        
        # Request tracking
        self.active_requests = {}
        self.request_lock = threading.RLock()
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0,
            'concurrent_peak': 0
        }
        
        # Initialize guards with connection pooling
        self._initialize_guards()
        self.logger.info(f"PII Detection Service initialized for tenant '{self.tenant_name}' - Max concurrent: {self.config.max_concurrent_requests}")
    


    def _convert_tenant_to_service_config(self, tenant_config: TenantPIIConfig) -> PIIServiceConfig:
        """Convert tenant config to service config for backward compatibility."""
        return PIIServiceConfig(
            max_concurrent_requests=tenant_config.max_concurrent_requests,
            request_timeout=tenant_config.request_timeout,
            log_level=tenant_config.log_level,
            enable_performance_monitoring=tenant_config.enable_performance_monitoring,
            guardrails_entities=tenant_config.guardrails_entities,
            model_name=tenant_config.model_name
        )
    




    def _setup_logger(self) -> logging.Logger:
        """Set up comprehensive logging for the service."""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create logger with tenant-specific name
        logger_name = f"pii_detection.{self.tenant_name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Avoid adding multiple handlers
        if logger.handlers:
            return logger
        
        # File handler for detailed logs - using azure_guardrails format
        if self.tenant_config and self.tenant_config.log_file_prefix:
            log_filename = logs_dir / f"{self.tenant_config.log_file_prefix}_{datetime.now().strftime('%Y%m%d')}.log"
        else:
            # Fallback for legacy usage
            log_filename = logs_dir / f"azure_guardrail_pii_{self.tenant_name}_{datetime.now().strftime('%Y%m%d')}.log"
            
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Performance log handler
        perf_log_filename = logs_dir / f"azure_guardrail_pii_{self.tenant_name}_performance_{datetime.now().strftime('%Y%m%d')}.log"
        perf_handler = logging.FileHandler(perf_log_filename, mode='a', encoding='utf-8')
        perf_handler.setLevel(logging.INFO)
        
        # Formatters - matching azure_guardrails format
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] ---> %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s ---> %(message)s'
        )
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERF ---> %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        perf_handler.setFormatter(perf_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Add performance handler with filter
        perf_handler.addFilter(lambda record: 'PERF:' in record.getMessage())
        logger.addHandler(perf_handler)
        
        return logger





    
    def _initialize_guards(self):
        """Initialize GuardRails guards with optimized settings."""
        try:
            self.logger.info("Initializing GuardRails guards...")
            
            # Stage 1: DetectPII Guard (lightweight, fast)
            self.detect_pii_guard = Guard().use(
                DetectPII,
                on_fail="exception"
            )
            
            # Stage 2: GuardrailsPII Guard (comprehensive)
            pii_validator = GuardrailsPII(
                entities=self.config.guardrails_entities,
                model_name=self.config.model_name
            )
            self.guardrails_pii_guard = Guard().use(pii_validator, on_fail="exception")
            
            self.logger.info(f"Guards initialized successfully with entities: {self.config.guardrails_entities}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize guards: {str(e)}")
            raise




    
    @contextmanager
    def _track_request(self, request_id: str, text: str):
        """Context manager to track active requests."""
        with self.request_lock:
            self.active_requests[request_id] = {
                'start_time': time.time(),
                'text_length': len(text),
                'thread': threading.current_thread().name
            }
            
            # Update concurrent peak
            current_concurrent = len(self.active_requests)
            if current_concurrent > self.performance_stats['concurrent_peak']:
                self.performance_stats['concurrent_peak'] = current_concurrent
        
        try:
            yield
        finally:
            with self.request_lock:
                if request_id in self.active_requests:
                    del self.active_requests[request_id]




    
    def _update_performance_stats(self, processing_time: float, success: bool):
        """Update performance statistics."""
        with self.request_lock:
            self.performance_stats['total_requests'] += 1
            if success:
                self.performance_stats['successful_requests'] += 1
            else:
                self.performance_stats['failed_requests'] += 1
            
            # Update rolling average
            total = self.performance_stats['total_requests']
            current_avg = self.performance_stats['average_processing_time']
            self.performance_stats['average_processing_time'] = (
                (current_avg * (total - 1) + processing_time) / total
            )
    



    def _stage1_detect_pii(self, request_id: str, text: str) -> tuple[bool, Optional[str]]:
        """
        Stage 1: DetectPII validation - Fast initial screening.
        
        Returns:
            tuple: (passed, error_message)
        """
        self.logger.debug(f"[{request_id}] Stage 1: DetectPII check starting")
        
        try:
            # Run DetectPII validation
            self.detect_pii_guard.validate(text)
            self.logger.debug(f"[{request_id}] Stage 1: DetectPII PASSED")
            return True, None
            
        except Exception as e:
            error_msg = f"DetectPII validation failed: {str(e)}"
            self.logger.info(f"[{request_id}] Stage 1: DetectPII FAILED - {error_msg}")
            return False, error_msg
    



    def _stage2_guardrails_pii(self, request_id: str, text: str) -> tuple[bool, Optional[str], List[str]]:
        """
        Stage 2: GuardrailsPII validation - Comprehensive PII detection.
        
        Returns:
            tuple: (passed, error_message, detected_entities)
        """
        self.logger.debug(f"[{request_id}] Stage 2: GuardrailsPII check starting")
        
        try:
            # Run GuardrailsPII validation
            result = self.guardrails_pii_guard.validate(text)
            self.logger.debug(f"[{request_id}] Stage 2: GuardrailsPII PASSED")
            return True, None, []
            
        except Exception as e:
            error_msg = f"GuardrailsPII validation failed: {str(e)}"
            self.logger.info(f"[{request_id}] Stage 2: GuardrailsPII FAILED - {error_msg}")
            
            # Extract detected entities from error message
            detected_entities = self._extract_entities_from_error(str(e))
            
            return False, error_msg, detected_entities
    



    def _extract_entities_from_error(self, error_message: str) -> List[str]:
        """Extract detected PII entities from error message."""
        entities = []
        try:
            # Try to parse entities from GuardRails error message
            for entity_type in self.config.guardrails_entities:
                if entity_type.lower() in error_message.lower():
                    entities.append(entity_type)
        except Exception:
            pass
        return entities
    




    def _process_single_request(self, request_id: str, text: str) -> PIICheckResult:
        """
        Process a single PII check request through both stages.
        
        Args:
            request_id: Unique identifier for the request
            text: Text to check for PII
            
        Returns:
            PIICheckResult containing the processing results
        """
        start_time = time.time()
        
        try:
            with self._track_request(request_id, text):
                self.logger.info(f"[{request_id}] Starting PII check for tenant '{self.tenant_name}', text length: {len(text)}")
                
                # Stage 1: DetectPII
                stage1_passed, stage1_error = self._stage1_detect_pii(request_id, text)
                
                if not stage1_passed:
                    # Stage 1 failed - return immediately
                    processing_time = time.time() - start_time
                    self._update_performance_stats(processing_time, False)
                    
                    result = PIICheckResult(
                        request_id=request_id,
                        tenant_id=self.tenant_id,
                        text=text,
                        is_safe=False,
                        stage_failed="detect_pii",
                        error_message=stage1_error,
                        processing_time=processing_time,
                        timestamp=datetime.now().isoformat(),
                        detected_entities=[]
                    )
                    
                    self.logger.warning(f"[{request_id}] PII check FAILED at Stage 1 in {processing_time:.3f}s")
                    return result
                
                # Stage 2: GuardrailsPII (only if Stage 1 passed)
                stage2_passed, stage2_error, detected_entities = self._stage2_guardrails_pii(request_id, text)
                
                processing_time = time.time() - start_time
                
                if not stage2_passed:
                    # Stage 2 failed
                    self._update_performance_stats(processing_time, False)
                    
                    result = PIICheckResult(
                        request_id=request_id,
                        tenant_id=self.tenant_id,
                        text=text,
                        is_safe=False,
                        stage_failed="guardrails_pii",
                        error_message=stage2_error,
                        processing_time=processing_time,
                        timestamp=datetime.now().isoformat(),
                        detected_entities=detected_entities
                    )
                    
                    self.logger.warning(f"[{request_id}] PII check FAILED at Stage 2 in {processing_time:.3f}s")
                    return result
                
                # Both stages passed - text is SAFE
                self._update_performance_stats(processing_time, True)
                
                result = PIICheckResult(
                    request_id=request_id,
                    tenant_id=self.tenant_id,
                    text=text,
                    is_safe=True,
                    stage_failed=None,
                    error_message=None,
                    processing_time=processing_time,
                    timestamp=datetime.now().isoformat(),
                    detected_entities=[]
                )
                
                self.logger.info(f"[{request_id}] PII check PASSED - Text is SAFE in {processing_time:.3f}s")
                self.logger.info(f"PERF: Request {request_id} completed successfully in {processing_time:.3f}s")
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, False)
            
            error_msg = f"Unexpected error during PII processing: {str(e)}"
            self.logger.error(f"[{request_id}] {error_msg}")
            
            return PIICheckResult(
                request_id=request_id,
                tenant_id=self.tenant_id,
                text=text,
                is_safe=False,
                stage_failed="system_error",
                error_message=error_msg,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                detected_entities=[]
            )




    
    def check_pii_single(self, text: str, request_id: Optional[str] = None) -> PIICheckResult:
        """
        Check a single text for PII using the two-stage validation process.
        
        Args:
            text: Text to check for PII
            request_id: Optional custom request ID
            
        Returns:
            PIICheckResult containing the check results
        """
        if not request_id:
            request_id = str(uuid.uuid4())[:8]
        
        if not text or not isinstance(text, str):
            return PIICheckResult(
                request_id=request_id,
                text=text or "",
                is_safe=False,
                stage_failed="validation_error",
                error_message="Text must be a non-empty string",
                processing_time=0.0,
                timestamp=datetime.now().isoformat(),
                detected_entities=[]
            )
        
        return self._process_single_request(request_id, text)



    
    def check_pii_batch(self, texts: List[str], timeout: Optional[int] = None) -> List[PIICheckResult]:
        """
        Check multiple texts for PII concurrently.
        
        Args:
            texts: List of texts to check
            timeout: Optional timeout for the entire batch operation
            
        Returns:
            List of PIICheckResult objects
        """
        if not texts:
            return []
        
        timeout = timeout or self.config.request_timeout
        results = []
        
        self.logger.info(f"Starting batch PII check for {len(texts)} texts with timeout {timeout}s")
        
        # Create futures for concurrent processing
        futures = {}
        for i, text in enumerate(texts):
            request_id = f"batch_{int(time.time())}_{i:03d}"   # CAN BE REPLACED WITH TENANT NAME FOR EASY REQUEST ANALYSIS
            future = self.executor.submit(self._process_single_request, request_id, text)
            futures[future] = request_id
        
        # Collect results with timeout
        try:
            for future in as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    request_id = futures[future]
                    self.logger.error(f"Error processing request {request_id}: {str(e)}")
                    
                    # Create error result
                    error_result = PIICheckResult(
                        request_id=request_id,
                        text="",
                        is_safe=False,
                        stage_failed="processing_error",
                        error_message=str(e),
                        processing_time=0.0,
                        timestamp=datetime.now().isoformat(),
                        detected_entities=[]
                    )
                    results.append(error_result)
        
        except TimeoutError:
            self.logger.error(f"Batch processing timed out after {timeout}s")
            # Add timeout results for remaining requests
            for future in futures:
                if not future.done():
                    future.cancel()
                    request_id = futures[future]
                    timeout_result = PIICheckResult(
                        request_id=request_id,
                        text="",
                        is_safe=False,
                        stage_failed="timeout",
                        error_message=f"Request timed out after {timeout}s",
                        processing_time=timeout,
                        timestamp=datetime.now().isoformat(),
                        detected_entities=[]
                    )
                    results.append(timeout_result)
        
        self.logger.info(f"Batch PII check completed: {len(results)} results")
        return results




    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self.request_lock:
            stats = self.performance_stats.copy()
            stats['active_requests'] = len(self.active_requests)
            stats['success_rate'] = (
                stats['successful_requests'] / max(stats['total_requests'], 1) * 100
            )
            return stats




    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        try:
            # Quick health check
            test_result = self.check_pii_single("Health check test", "health_check")
            
            return {
                "status": "healthy" if test_result else "degraded",
                "timestamp": datetime.now().isoformat(),
                "performance_stats": self.get_performance_stats(),
                "guards_initialized": hasattr(self, 'detect_pii_guard') and hasattr(self, 'guardrails_pii_guard')
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    




    def shutdown(self):
        """Gracefully shutdown the service."""
        self.logger.info("Shutting down PII Detection Service...")

        # Wait for active requests to complete (compatible with older Python versions)
        try:
            self.executor.shutdown(wait=True)

            # Manual timeout handling for active requests
            start_time = time.time()
            while len(self.active_requests) > 0 and (time.time() - start_time) < 30:
                time.sleep(0.1)

        except Exception as e:
            self.logger.warning(f"Error during executor shutdown: {str(e)}")

        # Log final statistics
        final_stats = self.get_performance_stats()
        self.logger.info(f"Service shutdown complete. Final stats: {json.dumps(final_stats, indent=2)}")



    
   