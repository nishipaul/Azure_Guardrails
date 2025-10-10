"""
Centralized Logging System for Guardrails PII Detection
=====================================================

A unified logging system that captures all functionality in JSON format
with detailed timing, metadata, and performance metrics.

Features:
- Single JSON log file per tenant per day
- Structured logging with timestamps
- Performance metrics (latency, processing time)
- Model information and configuration
- User tracking and request correlation
- Thread-safe operations
- Automatic log rotation by date

Author: AI Assistant
Date: January 2025
Version: 1.0.0
"""

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import uuid


@dataclass
class LogEntry:
    """Base structure for all log entries"""
    timestamp: str
    tenant_id: str
    functionality: str
    operation: str
    text: Optional[str] = None  # The text being processed
    user_name: Optional[str] = None
    request_id: Optional[str] = None
    processing_time_ms: Optional[float] = None
    status: str = "success"
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AzureGuardrailLog(LogEntry):
    """Log entry for Azure Guardrail operations"""
    functionality: str
    model_used: Optional[str] = None
    input_text_length: Optional[int] = None
    output_text_length: Optional[int] = None
    confidence_score: Optional[float] = None
    categories_checked: Optional[list] = None
    api_latency_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.functionality is None:
            self.functionality = "azure_guardrail"


@dataclass
class GuardrailPIILog(LogEntry):
    """Log entry for Guardrail PII operations"""
    functionality: str
    model_used: Optional[str] = None
    input_text_length: Optional[int] = None
    output_text_length: Optional[int] = None
    pii_entities_detected: Optional[list] = None
    pii_confidence_scores: Optional[Dict[str, float]] = None
    processing_stages: Optional[list] = None
    
    def __post_init__(self):
        if self.functionality is None:
            self.functionality = "guardrail_pii"


@dataclass
class PromptManagementLog(LogEntry):
    """Log entry for Prompt Management operations"""
    functionality: str
    prompt_name: Optional[str] = None
    prompt_label: Optional[str] = None
    prompt_version: Optional[str] = None
    model_used: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model_latency_ms: Optional[float] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    
    def __post_init__(self):
        if self.functionality is None:
            self.functionality = "prompt_management"


@dataclass
class SystemLog(LogEntry):
    """Log entry for System operations"""
    functionality: str
    operation_type: Optional[str] = None
    resource_usage: Optional[Dict[str, Any]] = None
    cache_status: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.functionality is None:
            self.functionality = "system"


class CentralizedLogger:
    """
    Centralized logging system for all Guardrails functionality
    """
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._log_cache = {}  # Cache for tenant log files
    
    def _get_log_file_path(self, tenant_id: str) -> Path:
        """Get the log file path for a specific tenant"""
        return self.logs_dir / f"guardrail_{tenant_id}.json"
    
    def _load_tenant_logs(self, tenant_id: str) -> Dict[str, Any]:
        """Load existing logs for the tenant"""
        log_file = self._get_log_file_path(tenant_id)
        
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted, start fresh
                return {}
        return {}
    
    def _save_logs(self, tenant_id: str, logs_data: Dict[str, Any]):
        """Save logs for the day"""
        log_file = self._get_log_file_path(tenant_id)
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving logs for tenant {tenant_id}: {e}")
    
    def _add_log_entry(self, tenant_id: str, log_entry: LogEntry):
        """Add a log entry to the tenant logs with new structure"""
        with self._lock:
            # Load existing logs
            logs_data = self._load_tenant_logs(tenant_id)

            # Get today's date key
            date_key = datetime.now().strftime('%Y-%m-%d')

            # Initialize date structure if not exists
            if date_key not in logs_data:
                logs_data[date_key] = {}

            # Use a fixed timestamp key for all operations of this query
            timestamp_key = f"query_{log_entry.request_id}"
            
            # Initialize timestamp structure if not exists
            if timestamp_key not in logs_data[date_key]:
                logs_data[date_key][timestamp_key] = {
                    "timestamp": log_entry.timestamp,
                    "text": self._truncate_text(log_entry.text) if log_entry.text else None,
                    "tenant_name": tenant_id
                }

            # Get or create call entry for this request_id
            request_id = log_entry.request_id or str(uuid.uuid4())

            # Determine which request ID block this log belongs to
            functionality = log_entry.functionality
            if functionality == "azure_guardrail" or functionality == "guardrail_pii":
                # Guardrails operations go to guardrails_requestid block
                # Use a fixed key for guardrails to ensure all guardrail operations go to the same block
                request_block_key = "GUARDRAIL_operations"
                # Determine if it's input or output guardrail based on operation
                if any(op in log_entry.operation for op in ["content_safety", "blocklist", "prompt_shield", "pii_validation", "query_allowed"]) and not any(op in log_entry.operation for op in ["output_", "groundedness"]):
                    section = "input_guardrails"
                else:
                    section = "output_guardrails"
            else:
                # Prompt management and system operations go to model_requestid block
                # Use a fixed key for model operations to ensure all model operations go to the same block
                request_block_key = "MODEL_REQUEST_operations"
                if functionality == "system":
                    section = "system_model"
                else:
                    section = functionality

            # Initialize the request block if it doesn't exist
            if request_block_key not in logs_data[date_key][timestamp_key]:
                if request_block_key.startswith("MODEL_REQUEST_"):
                    logs_data[date_key][timestamp_key][request_block_key] = {
                        "timestamp": log_entry.timestamp,
                        "request_id": request_id,
                        "input_guardrails": {},
                        "output_guardrails": {},
                        "prompt_management": {},
                        "system_model": {}
                    }
                else:  # GUARDRAIL_
                    logs_data[date_key][timestamp_key][request_block_key] = {
                        "timestamp": log_entry.timestamp,
                        "request_id": request_id,
                        "input_guardrails": {},
                        "output_guardrails": {},
                        "prompt_management": {},
                        "system_guardrail": {}
                    }

            # Initialize the section if it doesn't exist
            if section not in logs_data[date_key][timestamp_key][request_block_key]:
                logs_data[date_key][timestamp_key][request_block_key][section] = {}

            # Convert log entry to dict and add processing time
            log_dict = asdict(log_entry)
            if log_entry.processing_time_ms is None:
                log_dict["processing_time_ms"] = 0  # Default processing time
            
            logs_data[date_key][timestamp_key][request_block_key][section][log_entry.operation] = log_dict

            # Save updated logs
            self._save_logs(tenant_id, logs_data)
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to 200 characters if it's long"""
        if not text:
            return None
        if len(text) <= 200:
            return text
        return text[:200] + "..."
    
    def log_azure_guardrail(self, 
                           tenant_id: str,
                           operation: str,
                           user_name: Optional[str] = None,
                           request_id: Optional[str] = None,
                           processing_time_ms: Optional[float] = None,
                           status: str = "success",
                           error_message: Optional[str] = None,
                           model_used: Optional[str] = None,
                           input_text_length: Optional[int] = None,
                           output_text_length: Optional[int] = None,
                           confidence_score: Optional[float] = None,
                           categories_checked: Optional[list] = None,
                           api_latency_ms: Optional[float] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Log Azure Guardrail operation"""
        
        # Extract text from metadata if available
        text = metadata.get('text', None) if metadata else None
        
        log_entry = AzureGuardrailLog(
            timestamp=datetime.now().isoformat(),
            tenant_id=tenant_id,
            functionality="azure_guardrail",
            operation=operation,
            text=text,
            user_name=user_name,
            request_id=request_id or str(uuid.uuid4()),
            processing_time_ms=processing_time_ms or 0,
            status=status,
            error_message=error_message,
            model_used=model_used,
            input_text_length=input_text_length,
            output_text_length=output_text_length,
            confidence_score=confidence_score,
            categories_checked=categories_checked,
            api_latency_ms=api_latency_ms,
            metadata=metadata
        )
        
        self._add_log_entry(tenant_id, log_entry)
    
    def log_guardrail_pii(self,
                         tenant_id: str,
                         operation: str,
                         user_name: Optional[str] = None,
                         request_id: Optional[str] = None,
                         processing_time_ms: Optional[float] = None,
                         status: str = "success",
                         error_message: Optional[str] = None,
                         model_used: Optional[str] = None,
                         input_text_length: Optional[int] = None,
                         output_text_length: Optional[int] = None,
                         pii_entities_detected: Optional[list] = None,
                         pii_confidence_scores: Optional[Dict[str, float]] = None,
                         processing_stages: Optional[list] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """Log Guardrail PII operation"""
        
        # Extract text from metadata if available
        text = metadata.get('text', None) if metadata else None
        
        log_entry = GuardrailPIILog(
            timestamp=datetime.now().isoformat(),
            tenant_id=tenant_id,
            functionality="guardrail_pii",
            operation=operation,
            text=text,
            user_name=user_name,
            request_id=request_id or str(uuid.uuid4()),
            processing_time_ms=processing_time_ms or 0,
            status=status,
            error_message=error_message,
            model_used=model_used,
            input_text_length=input_text_length,
            output_text_length=output_text_length,
            pii_entities_detected=pii_entities_detected,
            pii_confidence_scores=pii_confidence_scores,
            processing_stages=processing_stages,
            metadata=metadata
        )
        
        self._add_log_entry(tenant_id, log_entry)
    
    def log_prompt_management(self,
                             tenant_id: str,
                             operation: str,
                             user_name: Optional[str] = None,
                             request_id: Optional[str] = None,
                             processing_time_ms: Optional[float] = None,
                             status: str = "success",
                             error_message: Optional[str] = None,
                             prompt_name: Optional[str] = None,
                             prompt_label: Optional[str] = None,
                             prompt_version: Optional[str] = None,
                             model_used: Optional[str] = None,
                             input_tokens: Optional[int] = None,
                             output_tokens: Optional[int] = None,
                             total_tokens: Optional[int] = None,
                             model_latency_ms: Optional[float] = None,
                             temperature: Optional[float] = None,
                             top_p: Optional[float] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """Log Prompt Management operation"""
        
        # Extract text from metadata if available
        text = metadata.get('text', None) if metadata else None
        
        log_entry = PromptManagementLog(
            timestamp=datetime.now().isoformat(),
            tenant_id=tenant_id,
            functionality="prompt_management",
            operation=operation,
            text=text,
            user_name=user_name,
            request_id=request_id or str(uuid.uuid4()),
            processing_time_ms=processing_time_ms or 0,
            status=status,
            error_message=error_message,
            prompt_name=prompt_name,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
            model_used=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model_latency_ms=model_latency_ms,
            temperature=temperature,
            top_p=top_p,
            metadata=metadata
        )
        
        self._add_log_entry(tenant_id, log_entry)
    
    def log_system(self,
                   tenant_id: str,
                   operation: str,
                   user_name: Optional[str] = None,
                   request_id: Optional[str] = None,
                   processing_time_ms: Optional[float] = None,
                   status: str = "success",
                   error_message: Optional[str] = None,
                   operation_type: Optional[str] = None,
                   resource_usage: Optional[Dict[str, Any]] = None,
                   cache_status: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   **kwargs):
        """Log System operation"""
        
        # Merge any additional kwargs into metadata
        if metadata is None:
            metadata = {}
        metadata.update(kwargs)
        
        # Extract text from metadata if available
        text = metadata.get('text', None) if metadata else None
        
        log_entry = SystemLog(
            timestamp=datetime.now().isoformat(),
            tenant_id=tenant_id,
            functionality="system",
            operation=operation,
            text=text,
            user_name=user_name,
            request_id=request_id or str(uuid.uuid4()),
            processing_time_ms=processing_time_ms or 0,
            status=status,
            error_message=error_message,
            operation_type=operation_type,
            resource_usage=resource_usage,
            cache_status=cache_status,
            metadata=metadata
        )
        
        self._add_log_entry(tenant_id, log_entry)
    
    @contextmanager
    def time_operation(self, tenant_id: str, functionality: str, operation: str, **kwargs):
        """Context manager to time operations and yield request_id"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            yield request_id
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            # Return processing time for manual logging if needed
            return processing_time
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            raise
    
    def get_logs_for_tenant(self, tenant_id: str, date: Optional[str] = None) -> Dict[str, Any]:
        """Get logs for a specific tenant and date"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        log_file = self.logs_dir / f"guardrail_{tenant_id}.json"
        
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs_data = json.load(f)
                    return logs_data.get(date, {})
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def get_logs_summary(self, tenant_id: str, date: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of logs for a tenant"""
        logs = self.get_logs_for_tenant(tenant_id, date)
        
        summary = {
            "date": date or datetime.now().strftime('%Y-%m-%d'),
            "tenant_id": tenant_id,
            "total_operations": 0,
            "functionality_counts": {},
            "average_processing_times": {},
            "error_counts": {},
            "success_rate": 0.0
        }
        
        total_ops = 0
        total_errors = 0
        
        for functionality, entries in logs.items():
            if isinstance(entries, list):
                count = len(entries)
                summary["functionality_counts"][functionality] = count
                total_ops += count
                
                # Calculate average processing times
                processing_times = [entry.get("processing_time_ms", 0) for entry in entries if entry.get("processing_time_ms")]
                if processing_times:
                    summary["average_processing_times"][functionality] = sum(processing_times) / len(processing_times)
                
                # Count errors
                errors = sum(1 for entry in entries if entry.get("status") == "error")
                summary["error_counts"][functionality] = errors
                total_errors += errors
        
        summary["total_operations"] = total_ops
        if total_ops > 0:
            summary["success_rate"] = ((total_ops - total_errors) / total_ops) * 100
        
        return summary


# Global logger instance
centralized_logger = CentralizedLogger()


# Convenience functions for easy usage
def log_azure_guardrail(**kwargs):
    """Convenience function for Azure Guardrail logging"""
    return centralized_logger.log_azure_guardrail(**kwargs)


def log_guardrail_pii(**kwargs):
    """Convenience function for Guardrail PII logging"""
    return centralized_logger.log_guardrail_pii(**kwargs)


def log_prompt_management(**kwargs):
    """Convenience function for Prompt Management logging"""
    return centralized_logger.log_prompt_management(**kwargs)


def log_system(**kwargs):
    """Convenience function for System logging"""
    return centralized_logger.log_system(**kwargs)


def time_operation(tenant_id: str, functionality: str, operation: str, **kwargs):
    """Convenience function for timing operations"""
    return centralized_logger.time_operation(tenant_id, functionality, operation, **kwargs)


def get_logs_summary(tenant_id: str, date: Optional[str] = None):
    """Convenience function for getting logs summary"""
    return centralized_logger.get_logs_summary(tenant_id, date)
