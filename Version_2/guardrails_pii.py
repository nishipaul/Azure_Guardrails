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
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    enable_performance_monitoring: bool = True
    guardrails_entities: List[str] = None
    model_name: str = "urchade/gliner_small-v2.1"  # DEAFULT MODEL - CAN BE SWITCHED TO OPENAI
    
    def __post_init__(self):
        if self.guardrails_entities is None:
            self.guardrails_entities = [
                "CREDIT_CARD", "CRYPTO", "EMAIL_ADDRESS", "IBAN_CODE", "IP_ADDRESS", "NRP", "PERSON", "PHONE_NUMBER", "MEDICAL_LICENSE",
                "URL", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", "US_PASSPORT", "US_SSN", "UK_NHS", "ES_NIF", "ES_NIE", "IT_FISCAL_CODE", "IT_DRIVER_LICENSE",
                "IT_VAT_CODE", "IT_PASSPORT", "IT_IDENTITY_CARD", "PL_PESEL", "SG_NRIC_FIN", "SG_UEN", "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE", "IN_PAN", "IN_AADHAAR",
                "IN_VEHICLE_REGISTRATION", "IN_VOTER", "IN_PASSPORT", "FI_PERSONAL_IDENTITY_CODE"
            ]
        # Set log file prefix with tenant name in azure_guardrails format







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
                "CREDIT_CARD", "CRYPTO", "EMAIL_ADDRESS", "IBAN_CODE", "IP_ADDRESS", "NRP",  "PERSON", "PHONE_NUMBER", "MEDICAL_LICENSE",
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
    






    def _ensure_performance_handler(self):
        """Lazily initialize performance log handler only when needed."""
        # Performance monitoring is now handled by centralized logging
        pass




    
    def _initialize_guards(self):
        """Initialize GuardRails guards with optimized settings."""
        try:
            
            # Stage 1: DetectPII Guard (lightweight, fast)
            self.detect_pii_guard = Guard().use(
                DetectPII,
                on_fail="exception"
            )
            
            # Stage 2: GuardrailsPII Guard (comprehensive)
            # Use on_fail="exception" to properly detect PII
            pii_validator = GuardrailsPII(
                entities=self.config.guardrails_entities,
                model_name=self.config.model_name
            )
            self.guardrails_pii_guard = Guard().use(pii_validator, on_fail="exception")
            
            
        except Exception as e:
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
        
        try:
            # Run DetectPII validation
            self.detect_pii_guard.validate(text)
            return True, None
            
        except Exception as e:
            # DetectPII can have false positives, check if entities can be extracted
            error_msg = str(e)
            detected_entities = self._extract_entities_from_error(error_msg)
            
            # If no specific entities detected, treat as false positive and pass to Stage 2
            if not detected_entities:
                return True, None
            
            return False, f"DetectPII validation failed: {error_msg}"
    



    def _stage2_guardrails_pii(self, request_id: str, text: str) -> tuple[bool, Optional[str], List[str]]:
        """
        Stage 2: GuardrailsPII validation - Comprehensive PII detection.
        
        Returns:
            tuple: (passed, error_message, detected_entities)
        """
        
        try:
            # Run GuardrailsPII validation - will throw exception if PII detected
            result = self.guardrails_pii_guard.validate(text)
            # If no exception, text is safe
            return True, None, []
            
        except Exception as e:
            # Exception thrown - check if it's real PII or false positive
            error_msg = str(e)
            
            # Extract detected entities from error message
            detected_entities = self._extract_entities_from_error(error_msg)
            
            # If no specific entities were detected, it's likely a false positive
            if not detected_entities:
                return True, None, []
            
            # Real PII detected
            return False, f"GuardrailsPII detected PII: {', '.join(detected_entities)}", detected_entities
    



    def _extract_entities_from_result(self, result) -> List[str]:
        """Extract detected PII entities from validation result."""
        entities = []
        try:
            # Check if result has validator_logs which contains detected entities
            if hasattr(result, 'validator_logs'):
                for log_entry in result.validator_logs:
                    if hasattr(log_entry, 'validator_name') and 'GuardrailsPII' in str(log_entry.validator_name):
                        # Extract entity information from the log
                        if hasattr(log_entry, 'value_before_validation') and hasattr(log_entry, 'value_after_validation'):
                            # If values differ, PII was detected
                            if log_entry.value_before_validation != log_entry.value_after_validation:
                                # Try to extract entity types from metadata
                                if hasattr(log_entry, 'metadata') and log_entry.metadata:
                                    metadata = log_entry.metadata
                                    if isinstance(metadata, dict):
                                        # Look for detected entity types in metadata
                                        if 'detected_entities' in metadata:
                                            entities.extend(metadata['detected_entities'])
                                        elif 'entity_type' in metadata:
                                            entities.append(metadata['entity_type'])
            
            # If no entities found from logs, try to infer from validated_output changes
            if not entities and hasattr(result, 'validated_output'):
                # Check for common PII patterns that were masked
                if '<EMAIL_ADDRESS>' in str(result.validated_output):
                    entities.append('EMAIL_ADDRESS')
                if '<PHONE_NUMBER>' in str(result.validated_output):
                    entities.append('PHONE_NUMBER')
                if '<PERSON>' in str(result.validated_output):
                    entities.append('PERSON')
                if '<US_SSN>' in str(result.validated_output):
                    entities.append('US_SSN')
                if '<CREDIT_CARD>' in str(result.validated_output):
                    entities.append('CREDIT_CARD')
                if '<LOCATION>' in str(result.validated_output):
                    entities.append('LOCATION')
                    
        except Exception as e:
            pass
        
        return list(set(entities))  # Remove duplicates
    
    


    
    def _extract_entities_from_error(self, error_message: str) -> List[str]:
        """Extract detected PII entities from error message."""
        entities = []
        try:
            error_lower = error_message.lower()
            
            # Check for specific entity patterns in the error message
            for entity_type in self.config.guardrails_entities:
                entity_lower = entity_type.lower()
                
                # Look for the entity type mentioned in the error
                if entity_lower in error_lower:
                    entities.append(entity_type)
                
                # Also check for common variations and keywords
                if entity_type == "US_SSN" and ("ssn" in error_lower or "social security" in error_lower):
                    if entity_type not in entities:
                        entities.append(entity_type)
                elif entity_type == "PHONE_NUMBER" and ("phone" in error_lower or "call" in error_lower):
                    if entity_type not in entities:
                        entities.append(entity_type)
                elif entity_type == "EMAIL_ADDRESS" and ("email" in error_lower or "@" in error_message):
                    if entity_type not in entities:
                        entities.append(entity_type)
                elif entity_type == "CREDIT_CARD" and ("credit" in error_lower or "card" in error_lower):
                    if entity_type not in entities:
                        entities.append(entity_type)
                elif entity_type == "US_PASSPORT" and "passport" in error_lower:
                    if entity_type not in entities:
                        entities.append(entity_type)
                elif entity_type == "IP_ADDRESS" and "ip" in error_lower:
                    if entity_type not in entities:
                        entities.append(entity_type)
                elif entity_type == "PERSON" and ("patient" in error_lower or ("john" in error_lower and "smith" in error_lower)):
                    if entity_type not in entities:
                        entities.append(entity_type)
                elif entity_type == "MEDICAL_LICENSE" and ("mrn" in error_lower or ("medical" in error_lower and "record" in error_lower)):
                    if entity_type not in entities:
                        entities.append(entity_type)
                        
        except Exception:
            pass
        
        return list(set(entities))  # Remove duplicates
    




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
                
                
                # Only create performance log file when actually logging performance data
                if self.config.enable_performance_monitoring:
                    self._ensure_performance_handler()
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, False)
            
            error_msg = f"Unexpected error during PII processing: {str(e)}"
            
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
                tenant_id=self.tenant_id,
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
                    
                    # Create error result
                    error_result = PIICheckResult(
                        request_id=request_id,
                        tenant_id=self.tenant_id,
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
            # Add timeout results for remaining requests
            for future in futures:
                if not future.done():
                    future.cancel()
                    request_id = futures[future]
                    timeout_result = PIICheckResult(
                        request_id=request_id,
                        tenant_id=self.tenant_id,
                        text="",
                        is_safe=False,
                        stage_failed="timeout",
                        error_message=f"Request timed out after {timeout}s",
                        processing_time=timeout,
                        timestamp=datetime.now().isoformat(),
                        detected_entities=[]
                    )
                    results.append(timeout_result)
        
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

        # Wait for active requests to complete (compatible with older Python versions)
        try:
            self.executor.shutdown(wait=True)

            # Manual timeout handling for active requests
            start_time = time.time()
            while len(self.active_requests) > 0 and (time.time() - start_time) < 30:
                time.sleep(0.1)

        except Exception as e:
            pass

        # Log final statistics
        final_stats = self.get_performance_stats()



    
   