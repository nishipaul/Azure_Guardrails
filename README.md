# About Azure Guardrails

The existing files of "azure_guardrails.py" checks for content safety and gives the flexibility to add list of words that needs to be blocked. This DOES NOT cover PII, which will be uploaded separately (done using Presidio by Microsoft).

#### How to run the file
1. Create .env file where you should have two values mandatorily -
     a. TENANT_ABC_KEY= {API KEY GENERATED FOR THE TENANT}
     b. TENANT_ABC_ENDPOINT="https://tenant-abc.cognitiveservices.azure.com"
   Here ABC is the name of the tenant. If you are new then please read this to know how to generate - https://simpplr.atlassian.net/wiki/spaces/DS/pages/3954311181/Azure+Guardrails+-+Content+Safety+and+Azure+AI+Foundry
2. Pip install all the dependencies in the requirements.txt file.
3. Open the file of azure_guardrail_enduser.ipynb which is calling the functions from azure_guardrails.py and run the cells to view the output
4. For step by step example, follow the file - azure_guardrail_usage_example.ipynb





# About PII using GuardrailAI with Microsoft Presidio - Optimized PII Detection Service

A high-performance, production-ready PII detection service using GuardRails AI with a two-stage validation approach optimized for concurrent processing.

## Features

- **Two-Stage Validation**: DetectPII → GuardrailsPII for comprehensive detection
- **High Concurrency**: Supports 100+ concurrent users with optimized resource management
- **Comprehensive Logging**: Detailed request tracking, performance monitoring, and error logging
- **Thread-Safe**: Fully thread-safe operations with proper synchronization
- **Performance Monitoring**: Real-time statistics and health checks
- **Error Handling**: Robust error handling with detailed error messages
- **Resource Management**: Optimized memory usage and connection pooling
- **Model for Guardrail**: Model used is - urchade/gliner_small-v2.1 (Default and Can be replaced)




## About Gliner model

The GLiNER Small-v2.1 model, developed by Urchade, is a Named Entity Recognition (NER) model that utilizes a bidirectional transformer encoder similar to BERT. This model is designed to identify a wide range of entity types in text, offering a practical alternative to traditional NER models that are limited to predefined entities.

* **File Size**: Approximately **611 MB**
* **Parameters**: Originally reported as **166 million parameters**.
* **Fine-Tuned Version**: A fine-tuned variant has **50 million parameters**.
* **Inference Memory**: Typically requires **~6 GB of system memory** during inference.
* **Inference Speed**: Offers **~60 ms per inference**, making it one of the faster models in its category
* **Use Case**: Suitable for resource-constrained environments where traditional large models may be impractical


## 🏗️ Architecture

### Two-Stage PII Detection Workflow

```
Text Input
    ↓
Stage 1: DetectPII (Fast screening)
    ↓
[FAIL] → Return Exception
    ↓
[PASS] → Stage 2: GuardrailsPII (Comprehensive check)
    ↓
[FAIL] → Return Exception
    ↓
[PASS] → Return "SAFE"
```

### Concurrency Design

- **ThreadPoolExecutor**: Manages worker threads for concurrent processing
- **Request Tracking**: Monitors active requests and performance metrics
- **Resource Pooling**: Optimized guard initialization and reuse
- **Queue Management**: Handles request queuing and rate limiting

## Installation

```bash
# Install requirements
pip install -r requirements.txt

# Install GuardRails hub (for PII validators)
pip install guardrails-ai[hub]
```




## Quick Start

### Service Configuration

```python
from guardrails_pii import OptimizedPIIDetectionService, PIIServiceConfig

# Custom configuration for high concurrency
config = PIIServiceConfig(
    max_concurrent_requests=200,  # Support 200 concurrent users
    request_timeout=30,
    log_level="INFO",
    guardrails_entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"]
)

service = OptimizedPIIDetectionService(config)

# Single text check
result = service.check_pii_single("Test message")

# Batch processing
texts = ["Message 1", "Message 2", "Message 3"]
results = service.check_pii_batch(texts)

service.shutdown()
```

### Concurrent Processing

```python
import threading
from concurrent.futures import ThreadPoolExecutor

def process_user_messages(user_id, messages):
    """Process messages for a single user."""
    service = OptimizedPIIDetectionService()
    results = []
    
    for message in messages:
        result = service.check_pii_single(message, f"user_{user_id}")
        results.append(result)
    
    return results

# Simulate 100 concurrent users
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = []
    
    for user_id in range(100):
        messages = [f"User {user_id} message {i}" for i in range(5)]
        future = executor.submit(process_user_messages, user_id, messages)
        futures.append(future)
    
    # Collect results
    all_results = []
    for future in futures:
        user_results = future.result()
        all_results.extend(user_results)

print(f"Processed {len(all_results)} messages from 100 concurrent users")
```

## Performance Monitoring

### Performance Statistics

```python
service = OptimizedPIIDetectionService()

# Process some requests...
service.check_pii_batch(["test1", "test2", "test3"])

# Get performance statistics
stats = service.get_performance_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Average processing time: {stats['average_processing_time']:.3f}s")
print(f"Peak concurrent requests: {stats['concurrent_peak']}")
```

### Health Monitoring

```python
# Health check
health = service.get_health_status()
print(f"Service status: {health['status']}")  # healthy, degraded, unhealthy
print(f"Guards initialized: {health['guards_initialized']}")
```





## API Reference

### PIICheckResult

Result object returned by PII checks:

```python
@dataclass
class PIICheckResult:
    request_id: str           # Unique request identifier
    text: str                 # Original text
    is_safe: bool             # True if no PII detected
    stage_failed: str         # "detect_pii", "guardrails_pii", or None
    error_message: str        # Error description if failed
    processing_time: float    # Processing time in seconds
    timestamp: str            # ISO timestamp
    detected_entities: List[str]  # List of detected PII entity types
```





### OptimizedPIIDetectionService

Main service class:

#### Methods

- `check_pii_single(text, request_id=None)` → `PIICheckResult`
- `check_pii_batch(texts, timeout=None)` → `List[PIICheckResult]`
- `get_performance_stats()` → `Dict[str, Any]`
- `get_health_status()` → `Dict[str, Any]`
- `shutdown()` → `None`




### Configuration

```python
@dataclass
class PIIServiceConfig:
    max_concurrent_requests: int = 100      # Max concurrent processing
    request_timeout: int = 30               # Request timeout in seconds
    log_level: str = "INFO"                 # Logging level
    enable_performance_monitoring: bool = True
    guardrails_entities: List[str] = None   # PII entities to detect
    model_name: str = "urchade/gliner_small-v2.1"  # GuardRails model
```




## Logging

The service creates detailed logs in the `logs/` directory:

- `pii_service_YYYYMMDD.log` - Main service logs
- `pii_performance_YYYYMMDD.log` - Performance metrics

### Log Levels

- **DEBUG**: Detailed request processing information
- **INFO**: General operation information and results
- **WARNING**: Performance issues and failed requests
- **ERROR**: System errors and exceptions

### Sample Log Entries

```
2025-10-03 10:30:15 - pii_detection_service - INFO - [PII-Worker-1] - [abc123] Starting PII check for text length: 45
2025-10-03 10:30:15 - pii_detection_service - DEBUG - [PII-Worker-1] - [abc123] Stage 1: DetectPII check starting
2025-10-03 10:30:15 - pii_detection_service - INFO - [PII-Worker-1] - [abc123] Stage 1: DetectPII FAILED - DetectPII validation failed: PII detected
2025-10-03 10:30:15 - pii_detection_service - WARNING - [PII-Worker-1] - [abc123] PII check FAILED at Stage 1 in 0.123s
2025-10-03 10:30:15 - PERF - Request abc123 failed at stage detect_pii in 0.123s
```





### Performance Tuning

```python
# High-performance configuration
config = PIIServiceConfig(
    max_concurrent_requests=200,
    request_timeout=45,
    log_level="WARNING",  # Reduce log overhead
    enable_performance_monitoring=True
)
```

### Monitoring Setup

```python
# Health check endpoint (for load balancers)
def health_check():
    service = get_pii_service()
    health = service.get_health_status()
    return health['status'] == 'healthy'

# Performance monitoring
def get_metrics():
    service = get_pii_service()
    return service.get_performance_stats()
```



## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce `max_concurrent_requests`
   - Lower log level to WARNING or ERROR
   - Monitor with `get_performance_stats()`

2. **Slow Processing**
   - Check `average_processing_time` in stats
   - Verify GuardRails model is loaded correctly
   - Consider using faster model variant

3. **Thread Pool Exhaustion**
   - Monitor `concurrent_peak` in stats
   - Increase `max_concurrent_requests` if needed
   - Implement request queuing in your application





## Performance Benchmarks for PII

Based on testing with the default configuration:

- **Single Request**: ~0.1-0.3 seconds
- **Batch Processing**: Current testing - ~50-100 requests/second
- **Memory Usage**: ~200-500MB (depending on model)
- **Concurrent Users**: Tested up to 200 simultaneous users

##  Security Considerations

- **Input Validation**: All inputs are validated before processing
- **Resource Limits**: Built-in protection against resource exhaustion
- **Error Handling**: No sensitive data in error messages
- **Logging**: PII data is not logged (only metadata)

## Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Review performance statistics with `get_performance_stats()`
3. Test with debug logging enabled
4. Verify GuardRails dependencies are installed correctly

