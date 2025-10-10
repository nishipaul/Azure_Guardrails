import os
import time
from azure_guardrails import AzureGuardrailFunctions, AzureBlocklistFunctions  # Using Azure Content Safety Services
from guardrails_pii import OptimizedPIIDetectionService, TenantPIIConfig, PIIServiceConfig    # Using GuardRails PII Detection
from typing import Dict, Any, List
from centralized_logger import centralized_logger, time_operation
import uuid
import threading
import warnings
warnings.filterwarnings("ignore")





def calculate_processing_time(start_time: float) -> float:
    """Calculate processing time in milliseconds"""
    return (time.time() - start_time) * 1000




# Global Azure Guardrail Functions cache - initialized once at startup - Azure Content Safety Client Setup
azure_guardrail_cache = {}
azure_cache_lock = threading.Lock()



def get_or_create_azure_guardrail_client(tenant_id):
    """Get Azure Guardrail client from cache or create if not exists (thread-safe)"""
    with azure_cache_lock:
        if tenant_id not in azure_guardrail_cache:
            print(f"Initializing Azure Guardrail client for: {tenant_id}")
            azure_guardrail_cache[tenant_id] = AzureGuardrailFunctions(initialize_tenants=[tenant_id])
            print(f"Azure Guardrail client cached for: {tenant_id}")
        else:
            print(f"Using cached Azure Guardrail client for: {tenant_id}")
        return azure_guardrail_cache[tenant_id]




# Global Azure Blocklist Functions cache - initialized once at startup - Azure Blocklist Client Setup
azure_blocklist_cache = {}
azure_blocklist_lock = threading.Lock()


def get_or_create_azure_blocklist_client(tenant_id):
    """Get Azure Blocklist client from cache or create if not exists (thread-safe)"""
    with azure_blocklist_lock:
        if tenant_id not in azure_blocklist_cache:
            print(f"Initializing Azure Blocklist client for: {tenant_id}")
            azure_blocklist_cache[tenant_id] = AzureBlocklistFunctions(initialize_tenants=[tenant_id])
            print(f"Azure Blocklist client cached for: {tenant_id}")
        else:
            print(f"Using cached Azure Blocklist client for: {tenant_id}")
        return azure_blocklist_cache[tenant_id]







# PII FUNCTION FOR SINGLE TEXT
def validate_pii_for_single_text(user_input: str, service: OptimizedPIIDetectionService) -> dict:
    """
    Validate user input for PII using the OptimizedPIIDetectionService using single text check for PII,
    and gracefully shut down all services after validation.

    Args:
        user_input (str): Input text from user.
        service (OptimizedPIIDetectionService): The main PII detection service instance.

    Returns:
        dict: Validation result with status, message, and timing details.
    """
    try:
        # Stage 1: Validate user input ---
        result = service.check_pii_single(user_input)

        if result.is_safe:
            response = {
                "status": "SAFE",
                "message": "Input is safe to process",
                "processing_time": result.processing_time
            }
        else:
            response = {
                "status": "UNSAFE",
                "message": "Input contains PII and cannot be processed",
                "stage_failed": result.stage_failed,
                "detected_entities": result.detected_entities,
                "processing_time": result.processing_time
            }

        return response

    except Exception as e:
        # Catch and log any unexpected error
        return {
            "status": "error",
            "message": f"Validation failed due to an unexpected error: {str(e)}"
        }

    finally:
        # Stage 2: Graceful shutdown of services
        try:
            service.shutdown()
        except Exception as e:
            print(f"Warning: Failed to shut down OptimizedPIIDetectionService")

        if service is not None:
            try:
                service.shutdown()
            except Exception as e:
                print(f"Warning: Failed to shut down TenantService")









# INPUT GUARDRAIL FUNCTION
# CHECKING IN SEQUENCE OF CONTENT SAFETY, BLOCKLIST, PROMPT SHIELD, AND PII VALIDATION
def process_user_query(user_query: str, pii_service: str, tenant_id: str = None, request_id: str = None) -> Dict[str, Any]:
    """
    Process user input through Input Guardrails:
    1. PII Validation
    2. Content Safety
    3. Blocklist
    4. Prompt Shield
    """
    # Use the provided request_id 
    guardrail_request_id = request_id # or str(uuid.uuid4())
    
    # Use cached Azure Guardrail client
    client = get_or_create_azure_guardrail_client(tenant_id)
    
    if not tenant_id:
        print("Pass the name for the tenant to continue")
        return
        
    if tenant_id:
        all_tenant_id = client.get_available_tenants()
        if tenant_id not in all_tenant_id:
            print(f"Tenant {tenant_id} not found in the list of available tenants")
            return

        # Stage 1: Content Safety
        try:
            cs_start_time = time.time()
            cs_result = client.analyze_text_content_safety(tenant_id=tenant_id, text=user_query)
            cs_processing_time = calculate_processing_time(cs_start_time)

            analysis = cs_result.get("analysis_results", {})
            if any(v.get("severity", 0) > 0 for v in analysis.values()):
                # Log blocked content with flagged text
                flagged_categories = [cat for cat, result in analysis.items() if result.get("severity", 0) > 0]
                flagged_text = user_query[:500] if len(user_query) > 500 else user_query
                
                centralized_logger.log_azure_guardrail(tenant_id=tenant_id, operation="content_safety_blocked", request_id=guardrail_request_id, processing_time_ms=cs_processing_time,
                                                       input_text_length=len(user_query), categories_checked=list(analysis.keys()), 
                                                       metadata={
                                                                    "text": user_query,
                                                                    "analysis_results": analysis,
                                                                    "flagged_categories": flagged_categories,
                                                                    "flagged_text": flagged_text
                                                                }
                                                        )
                return {"status": "blocked", "reason": "Content Safety flagged text.", "details": cs_result}
            else:
                centralized_logger.log_azure_guardrail(
                    tenant_id=tenant_id, operation="content_safety_passed", request_id=guardrail_request_id,
                    processing_time_ms=cs_processing_time, input_text_length=len(user_query), categories_checked=list(analysis.keys()), metadata={"text": user_query}
                    )

        except Exception as e:
            cs_processing_time = calculate_processing_time(cs_start_time)
            centralized_logger.log_azure_guardrail(tenant_id=tenant_id, operation="content_safety_error", request_id=guardrail_request_id,
                processing_time_ms=cs_processing_time, input_text_length=len(user_query), status="error", error_message=str(e), metadata={"text": user_query}
                )

            return {"status": "error", "reason": f"Content Safety check failed: {str(e)}"}


    # Stage 2: Blocklist
    try:
        bl_start_time = time.time()
        blocklist_client = get_or_create_azure_blocklist_client(tenant_id)
        blocklists = blocklist_client.get_all_blocklists(tenant_id=tenant_id)
        
        if blocklists and blocklists.get("blocklists"):
            for bl in blocklists["blocklists"]:
                bl_name = bl.get("name")
                bl_result = client.analyze_text_with_blocklist(
                    tenant_id=tenant_id,
                    text=user_query,  
                    blocklist_names=[bl_name]
                )

                bl_processing_time = calculate_processing_time(bl_start_time)
                # Check for blocklist matches
                if bl_result.get("blocklist_matches"):
                    # Log blocked content with flagged text
                    flagged_text = user_query[:500] if len(user_query) > 500 else user_query
                    
                    centralized_logger.log_azure_guardrail(
                        tenant_id=tenant_id,
                        operation="blocklist_blocked",
                        request_id=guardrail_request_id,
                        processing_time_ms=bl_processing_time,
                        input_text_length=len(user_query),
                        metadata={
                            "text": user_query,
                            "blocklist_name": bl_name,
                            "blocklist_matches": bl_result.get("blocklist_matches"),
                            "flagged_text": flagged_text
                        }
                    )
                    return {
                        "status": "blocked",
                        "reason": f"Matched blocklist '{bl_name}'.",
                        "details": bl_result
                    }

                # Check content safety results inside blocklist
                cs_results = bl_result.get("content_safety_results", {})
                if any(v.get("severity", 0) > 0 for v in cs_results.values()):
                    # Log blocked content with flagged text
                    flagged_text = user_query[:500] if len(user_query) > 500 else user_query
                    flagged_categories = [cat for cat, result in cs_results.items() if result.get("severity", 0) > 0]
                    
                    bl_processing_time = calculate_processing_time(bl_start_time)
                    centralized_logger.log_azure_guardrail(
                        tenant_id=tenant_id,
                        operation="blocklist_content_safety_blocked",
                        request_id=guardrail_request_id,
                        processing_time_ms=bl_processing_time,
                        input_text_length=len(user_query),
                        metadata={
                            "text": user_query,
                            "blocklist_name": bl_name,
                            "content_safety_results": cs_results,
                            "flagged_categories": flagged_categories,
                            "flagged_text": flagged_text
                        }
                    )
                    return {
                        "status": "blocked",
                        "reason": f"Content Safety flagged text in blocklist '{bl_name}'.",
                        "details": bl_result
                    }

    except Exception as e:
        bl_processing_time = calculate_processing_time(bl_start_time)
        centralized_logger.log_azure_guardrail(
            tenant_id=tenant_id,
            operation="blocklist_error",
            request_id=guardrail_request_id,
            processing_time_ms=bl_processing_time,
            input_text_length=len(user_query),
            status="error",
            error_message=str(e),
            metadata={"text": user_query}
        )
        return {"status": "error", "reason": f"Blocklist check failed: {str(e)}"}
    
    bl_processing_time = calculate_processing_time(bl_start_time)
    # Log that blocklist check passed
    centralized_logger.log_azure_guardrail(
        tenant_id=tenant_id,
        operation="blocklist_passed",
        request_id=guardrail_request_id,
        processing_time_ms=bl_processing_time,
        input_text_length=len(user_query),
        metadata={"text": user_query}
    )

    # Stage 3: Prompt Shield
    try:
        ps_start_time = time.time()
        ps_result = client.shield_prompt(
            tenant_id=tenant_id,
            user_prompt=user_query  
        )
        ps_processing_time = calculate_processing_time(ps_start_time)
        attack_detected = ps_result.get("userPromptAnalysis", {}).get("attackDetected", False)
        if attack_detected:
            # Log blocked content with flagged text
            flagged_text = user_query[:500] if len(user_query) > 500 else user_query
            
            centralized_logger.log_azure_guardrail(
                tenant_id=tenant_id,
                operation="prompt_shield_blocked",
                request_id=guardrail_request_id,
                processing_time_ms=ps_processing_time,
                input_text_length=len(user_query),
                metadata={
                    "text": user_query,
                    "prompt_shield_result": ps_result,
                    "flagged_text": flagged_text
                }
            )
            return {
                "status": "blocked",
                "reason": "Prompt Shield flagged input.",
                "details": ps_result
            }
        else:
            # Log passed content
            centralized_logger.log_azure_guardrail(
                tenant_id=tenant_id,
                operation="prompt_shield_passed",
                request_id=guardrail_request_id,
                processing_time_ms=ps_processing_time,
                input_text_length=len(user_query),
                metadata={"text": user_query}
            )
    except Exception as e:
        return {"status": "error", "reason": f"Prompt Shield check failed: {str(e)}"}



    # Stage 4: PII Validation
    try:
        pii_start_time = time.time()
        pii_check = validate_pii_for_single_text(user_query, pii_service)
        pii_processing_time = calculate_processing_time(pii_start_time)

        if pii_check.get("status") == "UNSAFE":
            # Log blocked content with flagged text
            flagged_text = user_query[:500] if len(user_query) > 500 else user_query
            
            centralized_logger.log_guardrail_pii(
                tenant_id=tenant_id,
                operation="pii_validation_blocked",
                request_id=guardrail_request_id,
                processing_time_ms=pii_processing_time,
                input_text_length=len(user_query),
                pii_entities_detected=pii_check.get("detected_entities", []),
                metadata={
                    "text": user_query,
                    "pii_check_result": pii_check,
                    "flagged_text": flagged_text
                }
            )
            return {
                "status": "blocked",
                "reason": "PII detected in input.",
                "details": pii_check
            }
        elif pii_check.get("status") == "error":
            centralized_logger.log_guardrail_pii(
                tenant_id=tenant_id,
                operation="pii_validation_error",
                request_id=guardrail_request_id,
                processing_time_ms=pii_processing_time,
                input_text_length=len(user_query),
                status="error",
                error_message=pii_check.get("error_message", "Unknown PII validation error"),
                metadata={"text": user_query}
            )
            return {
                "status": "error",
                "reason": "PII validation failed.",
                "details": pii_check
            }
        else:
            # Log passed content
            centralized_logger.log_guardrail_pii(
                tenant_id=tenant_id,
                operation="pii_validation_passed",
                request_id=guardrail_request_id,
                processing_time_ms=pii_processing_time,
                input_text_length=len(user_query),
                metadata={"text": user_query}
            )
    except Exception as e:
        return {"status": "error", "reason": f"PII validation check failed: {str(e)}"}

    # Log that the query passed all checks
    centralized_logger.log_system(
        tenant_id=tenant_id,
        operation="query_allowed",
        request_id=guardrail_request_id,
        processing_time_ms=0,  # This is a summary log, not a timed operation
        input_text_length=len(user_query),
        metadata={
            "text": user_query,
            "message": "Query passed all checks",
            "stages_passed": ["content_safety", "blocklist", "prompt_shield", "pii_validation"]
        }
    )

    return {"status": "allowed", "user_query": user_query, "message": "Query passed all checks."}













# MODEL RESPONSE GUARDRAIL FUNCTION
# CHECKING IN SEQUENCE OF CONTENT SAFETY, BLOCKLIST, AND GROUNDEDNESS
def process_model_response(user_query: str, model_response: str, source_text: str, tenant_id: str = None, request_id: str = None) -> Dict[str, Any]:
    """
    Process model response through Output Guardrails:
    1. Content Safety
    2. Blocklist
    3. Groundedness
    """
    # Use the provided request_id or generate one
    guardrail_request_id = request_id or str(uuid.uuid4())
    
    # Use cached Azure Guardrail client
    client = get_or_create_azure_guardrail_client(tenant_id)
    
    if not tenant_id:
        print("Pass the name for the tenant to continue")
        return
        
    all_tenant_id = client.get_available_tenants()
    
    if tenant_id not in all_tenant_id:
        print(f"Tenant {tenant_id} not found in the list of available tenants")
        return

    # Stage 1: Content Safety
    try:
        cs_start_time = time.time()
        cs_result = client.analyze_text_content_safety(
            tenant_id=tenant_id,
            text=model_response
        )
        cs_processing_time = calculate_processing_time(cs_start_time)
        analysis = cs_result.get("analysis_results", {})
        if any(v.get("severity", 0) > 0 for v in analysis.values()):
            # Log blocked content
            centralized_logger.log_azure_guardrail(
                tenant_id=tenant_id,
                operation="output_content_safety_blocked",
                request_id=guardrail_request_id,
                processing_time_ms=cs_processing_time,
                input_text_length=len(model_response),
                categories_checked=list(analysis.keys()),
                metadata={
                    "text": model_response,
                    "analysis_results": analysis
                }
            )
            return {"status": "blocked", "reason": "Content Safety flagged response.", "details": cs_result}
        else:
            # Log passed content
            centralized_logger.log_azure_guardrail(
                tenant_id=tenant_id,
                operation="output_content_safety_passed",
                request_id=guardrail_request_id,
                processing_time_ms=cs_processing_time,
                input_text_length=len(model_response),
                categories_checked=list(analysis.keys()),
                metadata={"text": model_response}
            )
    except Exception as e:
        cs_processing_time = calculate_processing_time(cs_start_time)
        centralized_logger.log_azure_guardrail(
            tenant_id=tenant_id,
            operation="output_content_safety_error",
            request_id=guardrail_request_id,
            processing_time_ms=cs_processing_time,
            input_text_length=len(model_response),
            status="error",
            error_message=str(e),
            metadata={"text": model_response}
        )
        return {"status": "error", "reason": f"Content Safety check failed: {str(e)}"}

    # Stage 2: Groundedness
    if source_text is None:
        # Log that groundedness is not applicable
        centralized_logger.log_azure_guardrail(
            tenant_id=tenant_id,
            operation="output_groundedness_not_applicable",
            request_id=guardrail_request_id,
            processing_time_ms=0,  # No processing time for not applicable case
            input_text_length=len(model_response),
            metadata={"text": model_response, "reason": "source_text is None"}
        )
    else:
        try:
            gr_start_time = time.time()
            groundedness_result = client.detect_groundedness(
                tenant_id=tenant_id,
                query=user_query,
                source_text=source_text,
                text=model_response
            )
            gr_processing_time = calculate_processing_time(gr_start_time)
            if groundedness_result.get("ungroundedDetected", False):
                # Log blocked content
                centralized_logger.log_azure_guardrail(
                    tenant_id=tenant_id,
                    operation="output_groundedness_blocked",
                    request_id=guardrail_request_id,
                    processing_time_ms=gr_processing_time,
                    input_text_length=len(model_response),
                    metadata={
                        "text": model_response,
                        "groundedness_result": groundedness_result
                    }
                )
                return {"status": "blocked", "reason": "Ungrounded response detected.", "details": groundedness_result}
            else:
                # Log passed content
                centralized_logger.log_azure_guardrail(
                    tenant_id=tenant_id,
                    operation="output_groundedness_passed",
                    request_id=guardrail_request_id,
                    processing_time_ms=gr_processing_time,
                    input_text_length=len(model_response),
                    metadata={"text": model_response}
                )
        except Exception as e:
            gr_processing_time = calculate_processing_time(gr_start_time)
            centralized_logger.log_azure_guardrail(
                tenant_id=tenant_id,
                operation="output_groundedness_error",
                request_id=guardrail_request_id,
                processing_time_ms=gr_processing_time,
                input_text_length=len(model_response),
                status="error",
                error_message=str(e),
                metadata={"text": model_response}
            )
            return {"status": "error", "reason": f"Groundedness check failed: {str(e)}"}

    # Stage 3: Blocklist
    try:
        bl_start_time = time.time()
        blocklist_client = get_or_create_azure_blocklist_client(tenant_id)
        blocklists = blocklist_client.get_all_blocklists(tenant_id=tenant_id)
        bl_processing_time = calculate_processing_time(bl_start_time)
        if blocklists and blocklists.get("blocklists"):
            for bl in blocklists["blocklists"]:
                bl_name = bl.get("name")
                bl_result = client.analyze_text_with_blocklist(
                    tenant_id=tenant_id,
                    text=model_response,
                    blocklist_names=[bl_name]
                )

                # Check blocklist matches
                if bl_result.get("blocklist_matches"):
                    # Log blocked content
                    centralized_logger.log_azure_guardrail(
                        tenant_id=tenant_id,
                        operation="output_blocklist_blocked",
                        request_id=guardrail_request_id,
                        processing_time_ms=bl_processing_time,
                        input_text_length=len(model_response),
                        metadata={
                            "text": model_response,
                            "blocklist_name": bl_name,
                            "blocklist_matches": bl_result.get("blocklist_matches")
                        }
                    )
                    return {"status": "blocked", "reason": f"Matched blocklist '{bl_name}'.", "details": bl_result}

                # Check content safety results inside blocklist
                cs_blocklist = bl_result.get("content_safety_results", {})
                if any(v.get("severity", 0) > 0 for v in cs_blocklist.values()):
                    # Log blocked content
                    centralized_logger.log_azure_guardrail(
                        tenant_id=tenant_id,
                        operation="output_blocklist_content_safety_blocked",
                        request_id=guardrail_request_id,
                        processing_time_ms=bl_processing_time,
                        input_text_length=len(model_response),
                        metadata={
                            "text": model_response,
                            "blocklist_name": bl_name,
                            "content_safety_results": cs_blocklist
                        }
                    )
                    return {"status": "blocked", "reason": f"Content Safety flagged text in blocklist '{bl_name}'.", "details": bl_result}
        
        # Log passed content
        centralized_logger.log_azure_guardrail(
            tenant_id=tenant_id,
            operation="output_blocklist_passed",
            request_id=guardrail_request_id,
            processing_time_ms=bl_processing_time,
            input_text_length=len(model_response),
            metadata={"text": model_response}
        )
    except Exception as e:
        bl_processing_time = calculate_processing_time(bl_start_time)
        centralized_logger.log_azure_guardrail(
            tenant_id=tenant_id,
            operation="output_blocklist_error",
            request_id=guardrail_request_id,
            processing_time_ms=bl_processing_time,
            input_text_length=len(model_response),
            status="error",
            error_message=str(e),
            metadata={"text": model_response}
        )
        return {"status": "error", "reason": f"Blocklist check failed: {str(e)}"}

    return {"status": "allowed", "model_response": model_response, "message": "Response passed all checks."}


