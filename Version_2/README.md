# 🛡️ Guardrails Prompt E2E System

A comprehensive end-to-end system for implementing AI guardrails with PII detection, content safety, and prompt management using Azure Content Safety, Langfuse, and FastAPI.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [File Structure](#file-structure)
- [Performance Optimizations](#performance-optimizations)
- [Logging](#logging)

## Overview

This system provides a complete guardrails solution for AI applications, implementing multiple layers of protection including:

- **Input Guardrails**: PII detection, content safety, blocklist filtering, and prompt shield
- **Output Guardrails**: Content safety validation, groundedness checking, and blocklist verification
- **Prompt Management**: Integration with Langfuse for dynamic prompt templates
- **Centralized Logging**: Comprehensive logging system with structured JSON output
- **High Performance**: Optimized for 100+ concurrent users with caching and thread safety

## Features

### Guardrails Protection
- **PII Detection**: Advanced PII entity detection using spaCy and custom models
- **Content Safety**: Azure Content Safety API integration for harmful content detection
- **Blocklist Management**: Custom blocklist filtering with Azure Blocklist API
- **Prompt Shield**: Protection against prompt injection attacks
- **Groundedness Detection**: Ensures responses are grounded in provided source material

### Performance Features
- **Thread-Safe Caching**: Optimized client caching for high concurrency
- **Single Request ID**: Complete request traceability across all operations
- **Parallel Processing**: Support for 100+ concurrent users
- **Memory Efficient**: Shared caches reduce memory footprint

### Monitoring & Logging
- **Centralized Logging**: Structured JSON logging with tenant-specific files
- **Performance Metrics**: Processing time tracking for all operations
- **Cache Monitoring**: Real-time cache status and management
- **Request Tracing**: Complete request lifecycle tracking

## Architecture

User Query -> Input Guardrail Checks (Azure Content Safety + Azure Blocklist + Azure Prompt Shield + Guardrails AI PII) -> Prompt Fetch from Langfuse -> Model Calling -> Output Guardrail Checks (Azure Content Safety + Azure Groundedness (optional) + Azure Blocklist Check)

## Installation

### Prerequisites
- Python 3.12 (Will not work with 3.13 or higher as of September 2025)
- Azure Content Safety API access
- Langfuse account
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   ```

2. **Create virtual environment**
   ```bash
   python -m venv azure_guardrails_venv
   source azure_guardrails_venv/bin/activate  # On Windows: azure_guardrails_venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file with the following variables:
   ```env
   # Azure Content Safety
   AZURE_CONTENT_SAFETY_ENDPOINT=your_azure_endpoint
   AZURE_CONTENT_SAFETY_KEY=your_azure_key
   
   # Langfuse
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_HOST=https://cloud.langfuse.com
   
   # OpenAI
   OPENAI_API_KEY=your_openai_api_key
   
   # Tenant Configurations (example)
   TENANT_amazon_ENDPOINT=your_amazon_endpoint
   TENANT_amazon_KEY=your_amazon_key
   TENANT_allinsale_ENDPOINT=your_allinsale_endpoint
   TENANT_allinsale_KEY=your_allinsale_key
   ```

## Configuration

### Tenant Configuration
The system supports multiple tenants. Each tenant requires:
- Azure Content Safety endpoint and key
- PII detection configuration
- Custom blocklist settings

Example tenant configuration:
```python
TENANT_amazon_ENDPOINT=https://amazon-contentsafety.cognitiveservices.azure.com/
TENANT_amazon_KEY=your_amazon_azure_key
TENANT_amazon_BLOCKLIST_NAMES=amazon_blocklist
```

### PII Detection Configuration
Customize PII entities to detect:
```python
guardrails_entities=[
    "CREDIT_CARD", "EMAIL_ADDRESS", "IP_ADDRESS", 
    "US_SSN", "US_PASSPORT", "PHONE_NUMBER"
]
```

## Usage

### Web Interface
1. **Start the server**
   ```bash
   uvicorn enduser:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the web interface**
   Open your browser and navigate to `http://localhost:8000`

3. **Submit queries**
   - Enter username (tenant identifier)
   - Select label (prompt version)
   - Enter your query
   - Click "Process Query"

### Programmatic Usage

```python
from enduser import receive_user_query

# Process a user query
result = receive_user_query(
    user_name="amazon",
    user_query="What are your delivery options?",
    label="latest"
)

print(result)
```

### Direct Function Usage

```python
from guardrails_main_function import process_user_query, process_model_response
from guardrails_pii import OptimizedPIIDetectionService

# Initialize PII service
pii_service = OptimizedPIIDetectionService(tenant_config)

# Process input guardrails
input_result = process_user_query(
    user_query="My email is john@example.com",
    pii_service=pii_service,
    tenant_id="amazon",
    request_id="unique-request-id"
)

# Process output guardrails
output_result = process_model_response(
    user_query="What is your refund policy?",
    model_response="Our refund policy allows returns within 30 days...",
    source_text=None,
    tenant_id="amazon",
    request_id="unique-request-id"
)
```

## API Endpoints

### Main Endpoints
- `GET /` - Web interface for query submission
- `POST /query-form` - Process user queries through web form
- `POST /query` - Process queries via API

### Management Endpoints
- `GET /cache-status` - View cache status and statistics
- `POST /clear-cache` - Clear all caches
- `GET /logs/{tenant_id}` - Retrieve logs for specific tenant

### Example API Usage
```bash
# Submit query via API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "amazon",
    "query": "What are your shipping options?",
    "label": "latest"
  }'

# Check cache status
curl -X GET "http://localhost:8000/cache-status"

# Clear caches
curl -X POST "http://localhost:8000/clear-cache"
```

## File Structure

```
Guardrails_Prompt_E2E/
├── 📄 enduser.py                    # FastAPI web application and main entry point
├── 📄 guardrails_main_function.py  # Core guardrails processing logic
├── 📄 guardrails_pii.py            # PII detection service implementation
├── 📄 azure_guardrails.py          # Azure Content Safety API integration
├── 📄 centralized_logger.py        # Centralized logging system
├── 📄 requirements.txt             # Python dependencies
├── 📄 create_langfuse_prompts.ipynb # Jupyter notebook for prompt creation
├── 📁 templates/                   # HTML templates for web interface
│   ├── index.html                  # Main query form
│   └── result.html                 # Results display
```

## Performance Optimizations

### Caching Strategy
- **Azure Client Caching**: Clients are initialized once per tenant and reused
- **PII Service Caching**: PII detection services are cached per tenant
- **Thread-Safe Operations**: All caches use locks for concurrent access

### Request Optimization
- **Single Request ID**: One ID flows through entire request lifecycle
- **Parallel Processing**: Multiple guardrails run concurrently where possible
- **Memory Efficiency**: Shared resources reduce memory footprint

### Scalability Features
- **100+ Concurrent Users**: Optimized for high concurrency
- **Horizontal Scaling**: Stateless design allows multiple instances
- **Resource Management**: Automatic cleanup and resource optimization

## Logging

### Log Structure
Logs are stored in JSON format with the following structure:
```json
{
  "2025-01-08": {
    "query_request_id": {
            "timestamp": "2025-01-08T17:39:04.871298",
            "text": "user query text",
            "tenant_name": "amazon",
            "MODEL_REQUEST_operations": {
            "timestamp": "2025-01-08T17:39:04.871298",
            "request_id": "uuid-here",
            "prompt_management": { ... },
            "system_model": { ... }
            },
            "GUARDRAIL_operations": {
            "timestamp": "2025-01-08T17:39:11.149777",
            "request_id": "uuid-here",
            "input_guardrails": { ... },
            "output_guardrails": { ... },
            "system_guardrail": { ... }
            }
        }
    }
}
```

### Log Categories
- **Input Guardrails**: Content safety, blocklist, prompt shield, PII validation
- **Output Guardrails**: Content safety, groundedness, blocklist verification
- **Prompt Management**: Langfuse operations, LLM calls
- **System Operations**: Request lifecycle, performance metrics

### Log File Naming
- Format: `guardrail_{tenant_name}.json`
- Location: `logs/` directory
- Rotation: Daily rotation with date in content

## Development

### Adding New Guardrails
1. Create new detection function in `guardrails_main_function.py`
2. Add logging calls using `centralized_logger`
3. Integrate into `process_user_query` or `process_model_response`
4. Update configuration as needed

### Adding New Tenants
1. Add tenant configuration to `.env` file
2. Create tenant-specific PII configuration
3. Test with sample queries
4. Monitor logs for proper operation

### Customizing PII Detection
Modify the `guardrails_entities` list in `create_config_for_pii_detection()`:
```python
guardrails_entities=[
    "CREDIT_CARD", "EMAIL_ADDRESS", "IP_ADDRESS",
    "CUSTOM_ENTITY"  # Add your custom entities
]
```

## Error Handling

The system includes comprehensive error handling:
- **Graceful Degradation**: System continues operation if individual components fail
- **Detailed Error Logging**: All errors are logged with context and stack traces
- **User-Friendly Messages**: Errors are presented in a user-friendly format
- **Recovery Mechanisms**: Automatic retry and fallback strategies

## Monitoring

### Performance Metrics
- Processing time for each operation
- Cache hit/miss ratios
- Error rates and types
- Request throughput

### Health Checks
- Azure API connectivity
- Langfuse service status
- Cache health
- Log file integrity


## Scope of Improvement
- Reducing Latency
- Parallel async operation building for guardrails