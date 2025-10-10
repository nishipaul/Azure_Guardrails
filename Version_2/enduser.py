# For loading the environment keys

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from datetime import datetime
from typing import Optional

from langfuse import Langfuse
from langchain_core.prompts import ChatPromptTemplate

from guardrails_main_function import *

# Import centralized logging system
from centralized_logger import centralized_logger, time_operation

# FastAPI imports 
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import time

load_dotenv()


def calculate_processing_time(start_time: float) -> float:
    """Calculate processing time in milliseconds"""
    return (time.time() - start_time) * 1000



# Initialize FastAPI app
app = FastAPI(title="Guardrails PII Detection API", description="API for PII detection and LLM query processing")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class QueryRequest(BaseModel):
    username: str
    query: str
    label: str = "latest"

class QueryResponse(BaseModel):
    result: str
    status: str = "success"

# Creating the client that will connect with the project via keys and host
langfuse_client = Langfuse(
    secret_key = os.getenv("LANGFUSE_SECRET_KEY"),
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY"),
    host = os.getenv("LANGFUSE_HOST")
)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")





# Global tenant service cache - initialized once at startup - This is for Guardrails PII
tenant_services_cache = {}



# Thread locks for thread-safe caching
import threading
tenant_cache_lock = threading.Lock()



def create_config_for_pii_detection(tenant_name):
    # Create tenant-specific configuration
    tenant_config = TenantPIIConfig(
        tenant_id=f"{tenant_name}_001",
        tenant_name=tenant_name,
        log_level="INFO",
        max_concurrent_requests=20,
        enable_performance_monitoring=True,
        # Customize entities to detect
        guardrails_entities=[
            "CREDIT_CARD", "CRYPTO", "EMAIL_ADDRESS", "IBAN_CODE", "IP_ADDRESS", "NRP", "MEDICAL_LICENSE",
            "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", "US_PASSPORT", "US_SSN", "UK_NHS", "ES_NIF", "ES_NIE", "IT_FISCAL_CODE", "IT_DRIVER_LICENSE",
            "IT_VAT_CODE", "IT_PASSPORT", "IT_IDENTITY_CARD", "PL_PESEL", "SG_NRIC_FIN", "SG_UEN", "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE", "IN_PAN", "IN_AADHAAR",
            "IN_VEHICLE_REGISTRATION", "IN_VOTER", "IN_PASSPORT", "FI_PERSONAL_IDENTITY_CODE"
        ]
    )

    # Initialize service with tenant config
    tenant_service = OptimizedPIIDetectionService(tenant_config=tenant_config)

    return tenant_service





def get_or_create_tenant_service(tenant_name):
    """Get tenant service from cache or create if not exists (thread-safe)"""
    with tenant_cache_lock:
        if tenant_name not in tenant_services_cache:
            print(f"Initializing tenant service for: {tenant_name}")
            tenant_services_cache[tenant_name] = create_config_for_pii_detection(tenant_name)
            print(f" Tenant service cached for: {tenant_name}")
        else:
            print(f" Using cached tenant service for: {tenant_name}")
        return tenant_services_cache[tenant_name]





def get_prompt_and_call_llm(user_name, user_query, label="latest", request_id=None):
    """Get prompt from Langfuse and call LLM"""
    start_time = time.time()
    try:
        # Log Langfuse prompt fetching - processing time is 0 as it is not a timed operation
        centralized_logger.log_prompt_management(tenant_id=user_name, operation="langfuse_prompt_fetch", request_id=request_id, processing_time_ms=0, 
                                                 prompt_name=user_name, prompt_label=label, metadata={"text": user_query})

        langfuse_prompt = langfuse_client.get_prompt(name=user_name, label=label)
        prompt_messages = langfuse_prompt.get_langchain_prompt()
        langchain_prompt = ChatPromptTemplate.from_messages(prompt_messages)
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, top_p=0.1, openai_api_key=OPENAI_KEY)  # Model can be changed as per user need

        llm_chain = langchain_prompt | llm
        user_data = {"assistant_name": "simtant", "user_name": user_name, "user_input": user_query}  # Variable based on prompt set up and is flexible to change as per user need

        response = llm_chain.invoke(input=user_data)

        processing_time = calculate_processing_time(start_time)
        # Log successful LLM response
        centralized_logger.log_prompt_management(tenant_id=user_name, operation="llm_response_generated", request_id=request_id, processing_time_ms=processing_time,
                                                 prompt_name=user_name, prompt_label=label, model_used="gpt-4o-mini", input_tokens=len(user_query.split()),
                                                 output_tokens=len(response.content.split()), total_tokens=len(user_query.split()) + len(response.content.split()),
                                                 temperature=0.2, top_p=0.1, 
                                                 metadata={"text": user_query,"response_preview": response.content[:200] + "..." if len(response.content) > 200 else response.content}
                                                 )

        return response.content
        
    except Exception as e:
        processing_time = calculate_processing_time(start_time)
        # Log LLM error
        centralized_logger.log_prompt_management(tenant_id=user_name, operation="langfuse_call_error", request_id=request_id, processing_time_ms=processing_time, prompt_name=user_name, 
                                                 prompt_label=label, model_used="gpt-4o-mini", status="error", error_message=str(e), metadata={"text": user_query}
                                                 )
        raise





def receive_user_query(user_name, user_query, label="latest"):
    """Process user query through PII detection and LLM"""
    # Create single request ID for the entire flow
    import uuid
    request_id = str(uuid.uuid4())
    
    # Log query initiation
    centralized_logger.log_system(tenant_id=user_name, operation="query_initiated_from_user", request_id=request_id, processing_time_ms=0, metadata={"text": user_query, "label": label})
    
    try:
        start_time = time.time()
        # Use cached tenant service
        tenant_service = get_or_create_tenant_service(user_name)
        
        # Process through all types of validation (Input Guardrails)
        result = process_user_query(user_query=user_query, pii_service=tenant_service, tenant_id=user_name, request_id=request_id)
        
        if result is None or result.get("status") == "allowed":
            # Input Guardrails validation passed, proceed to LLM
            llm_result = get_prompt_and_call_llm(user_name, user_query, label, request_id)
            
            # Validate model response (Output Guardrails)
            output_response = process_model_response(user_query=user_query, model_response=llm_result, source_text=None, tenant_id=user_name, request_id=request_id)
            
            processing_time = calculate_processing_time(start_time)
            if output_response["status"] == "allowed":
                # Log successful completion
                centralized_logger.log_system(tenant_id=user_name, operation="query_completed_successfully", request_id=request_id,
                                              processing_time_ms=processing_time, metadata={"text": user_query, "response_length": len(llm_result)}
                                              )
                return llm_result
            else:
                # Log blocked response
                centralized_logger.log_system(tenant_id=user_name, operation="model_response_blocked_by_output_guardrails", request_id=request_id, processing_time_ms=processing_time,
                                              status="blocked", error_message=output_response["reason"], metadata={"text": user_query, "response_length": len(llm_result)})
                return output_response["reason"]
        else:
            processing_time = calculate_processing_time(start_time)
            # Log blocked query
            centralized_logger.log_system(
                tenant_id=user_name, operation="query_blocked_by_input_guardrails", request_id=request_id, processing_time_ms=processing_time, status="blocked",
                error_message=result["reason"],metadata={"text": user_query}
                )
            return result["reason"]
    except Exception as e:
        processing_time = calculate_processing_time(start_time)
        # Log any errors
        centralized_logger.log_system(tenant_id=user_name, operation="query_error", request_id=request_id, processing_time_ms=processing_time, status="error", error_message=str(e), metadata={"text": user_query})
        return f"Error processing query: {str(e)}"




# FastAPI Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process user query through PII detection and LLM"""
    try:
        result = receive_user_query(user_name=request.username, user_query=request.query, label=request.label)
        return QueryResponse(result=result, status="success")
    except Exception as e:
        return QueryResponse(result=f"Error: {str(e)}", status="error")

@app.post("/query-form", response_class=HTMLResponse)
async def process_query_form(request: Request, username: str = Form(...), query: str = Form(...), label: str = Form(...)):
    """Process query from HTML form and return result page"""
    try:
        result = receive_user_query(user_name=username, user_query=query, label=label)
        return templates.TemplateResponse("result.html", {
            "request": request,
            "username": username,
            "label": label,
            "query": query,
            "result": result,
            "status": "success"
        })
    except Exception as e:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "username": username,
            "label": label,
            "query": query,
            "result": f"Error: {str(e)}",
            "status": "error"
        })

@app.get("/cache-status")
async def get_cache_status():
    """Get information about cached services"""
    return {
        "cached_tenants": list(tenant_services_cache.keys()),
        "total_cached_tenants": len(tenant_services_cache),
        "cached_azure_clients": list(azure_guardrail_cache.keys()),
        "total_cached_azure_clients": len(azure_guardrail_cache)
    }

@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches"""
    with tenant_cache_lock:
        tenant_services_cache.clear()
    with azure_cache_lock:
        azure_guardrail_cache.clear()
    return {"message": "All caches cleared successfully"}

@app.get("/logs/{tenant_id}")
async def get_logs(tenant_id: str, date: Optional[str] = None):
    """Get logs for a specific tenant"""
    try:
        logs = centralized_logger.get_logs_for_tenant(tenant_id, date)
        return {"tenant_id": tenant_id, "date": date or datetime.now().strftime('%Y-%m-%d'), "logs": logs}
    except Exception as e:
        return {"error": str(e)}

@app.get("/logs/{tenant_id}/summary")
async def get_logs_summary(tenant_id: str, date: Optional[str] = None):
    """Get logs summary for a specific tenant"""
    try:
        summary = centralized_logger.get_logs_summary(tenant_id, date)
        return summary
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting Guardrails PII Detection FastAPI Application")
    print("Tenant services will be cached for optimal performance")
    uvicorn.run(app, host="0.0.0.0", port=8000)