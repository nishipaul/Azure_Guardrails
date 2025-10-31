# Output Guardrails - Azure Content Safety

This directory contains output guardrail modules for analyzing generated text after processing. The modules provide PII detection and groundedness checking using Azure services.

## Overview

The Output Guardrails system provides comprehensive analysis of generated/response text through multiple checks:

1. **PII Detection** - Identifies Personally Identifiable Information in generated text
2. **Groundedness Check** - Verifies if generated content is grounded in source documents (requires source documents)

## Files Structure

```
Output_Guardrails/
├── __init__.py              # Package initialization
├── main_output.py           # Main orchestrator for all output checks
├── pii_check.py            # PII detection and modification module
├── pii_modify.py           # PII redaction module
└── groundedness_check.py   # Groundedness detection module
```

## Requirements

```bash
pip install azure-ai-textanalytics azure-core requests
```

### Environment Variables

Set the following environment variables:
```bash
export endpoint="your_azure_content_safety_endpoint"
export subscription_key="your_azure_subscription_key"
```

## Quick Start

### Using the Main Orchestrator (Recommended)

```python
import os
from main_output import OutputGuardrailOrchestrator

# Initialize orchestrator
orchestrator = OutputGuardrailOrchestrator(
    endpoint=os.getenv("endpoint"),
    subscription_key=os.getenv("subscription_key")
)

# Analyze generated text with source documents
result = orchestrator.analyze(
    generated_text="Generated response text",
    source_documents=["Source document 1", "Source document 2"],
    execution_mode='parallel'
)

print(result)
```

### Using Individual Modules

```python
from pii_check import PIIChecker
from groundedness_check import GroundednessChecker

# Individual module usage
pii_checker = PIIChecker(endpoint, key)
result = pii_checker.analyze_pii("Generated text")

groundedness_checker = GroundednessChecker(endpoint, key)
result = groundedness_checker.check_groundedness(
    "Generated text", 
    ["Source doc 1", "Source doc 2"]
)
```

## Module Documentation

### 1. main_output.py

**OutputGuardrailOrchestrator** - Main orchestrator class that runs all output guardrail checks.

#### Methods

##### `__init__(endpoint, subscription_key, ...)`
Initialize the orchestrator with Azure credentials.

**Parameters:**
- `endpoint` (str): Azure endpoint URL
- `subscription_key` (str): Azure subscription key
- `groundedness_api_version` (str, optional): API version for groundedness (default: "2024-09-15-preview")
- `domain` (DomainType, optional): Domain type for groundedness - `GENERIC` or `MEDICAL` (default: `GENERIC`)
- `task_type` (TaskType, optional): Task type - `SUMMARIZATION` or `QNA` (default: `SUMMARIZATION`)

##### `analyze(generated_text, source_documents=None, functions=None, ...)`
Synchronous method to analyze generated text.

**Parameters:**
- `generated_text` (str): Generated/response text to analyze
- `source_documents` (List[str], optional): Source documents for groundedness check. If None or empty, groundedness_check is automatically skipped
- `functions` (List[str], optional): Functions to execute. Options: `'pii_check'`, `'groundedness_check'`. Default: all (or only `pii_check` if no documents)
- `execution_mode` (str): `'parallel'` or `'sequential'`. Default: `'parallel'`
- `enable_pii_modify` (bool): Enable PII redaction. Default: False
- `keep_categories` (List[str], optional): PII categories to keep when blocking
- `content_text` (str, optional): Optional content text for groundedness (if None, uses generated_text)
- `reasoning` (bool): Whether to include reasoning in groundedness. Default: False

**Returns:**
```python
{
    'query_timestamp': {
        'timestamp': '2024-01-01T12:34:56.789012',
        'pii_check': {...},
        'groundedness_check': {...},  # Only if source_documents provided
        'summary': {
            'output_check_passed': True/False,
            'failed_functions': []
        }
    }
}
```

##### `analyze_async(generated_text, source_documents=None, ...)`
Asynchronous version of `analyze()` method. Use `await` in async contexts or Jupyter notebooks.

**Example:**
```python
# In Jupyter notebook or async function
result = await orchestrator.analyze_async(
    generated_text="Generated response",
    source_documents=["Doc 1", "Doc 2"],
    execution_mode='parallel'
)
```

#### Usage Examples

**Basic usage (with source documents):**
```python
result = orchestrator.analyze(
    generated_text="The main topic is artificial intelligence...",
    source_documents=["Document about AI...", "Additional context..."]
)
```

**Without source documents (only PII check):**
```python
result = orchestrator.analyze(
    generated_text="Generated response text"
    # groundedness_check automatically skipped
)
```

**Selective functions:**
```python
result = orchestrator.analyze(
    generated_text="Response text",
    source_documents=["Doc 1"],
    functions=['pii_check'],  # Only PII check
    execution_mode='sequential'
)
```

**With PII redaction:**
```python
result = orchestrator.analyze(
    generated_text="Contact me at john@example.com or 555-5555",
    enable_pii_modify=True,
    keep_categories=['Address']  # Keep Address, redact others
)
```

**Async usage:**
```python
# In Jupyter notebook
result = await orchestrator.analyze_async(
    generated_text="Generated response",
    source_documents=["Source doc"],
    execution_mode='parallel'
)
```

---

### 2. pii_check.py

**PIIChecker** - Detects Personally Identifiable Information (PII) in generated text.

#### Methods

##### `analyze_pii(text, language='en', enable_pii_modify=False)`
Detect all PII entities in text.

**Parameters:**
- `text` (str): Generated text to analyze
- `language` (str): Language code. Default: 'en'
- `enable_pii_modify` (bool): Enable PII redaction. Default: False

**Returns:**
```python
{
    'pii_check': {
        'input_text': 'text',
        'entities': [
            {
                'entity': '917234567890',
                'category': 'PhoneNumber',
                'confidence_score': 0.8,
                'offset': 24,
                'length': 12
            }
        ],
        'pii_detected': True,
        'pii_detected_count': 2,
        'time_taken': 0.32
    },
    'pii_modify': {
        'enabled': False  # or True with redacted_text if enabled
    }
}
```

##### `analyze_pii_block_categories(text, keep_categories, language='en', enable_pii_modify=False)`
Detect PII but keep only specified categories.

**Parameters:**
- `text` (str): Generated text to analyze
- `keep_categories` (List[str]): Categories to keep (others are filtered out)
- `language` (str): Language code. Default: 'en'
- `enable_pii_modify` (bool): Enable PII redaction. Default: False

**Example:**
```python
from pii_check import PIIChecker

checker = PIIChecker(endpoint, key)

# Detect all PII
result = checker.analyze_pii("Email: john@example.com, Phone: 555-5555")

# Block specific categories, keep Address
result = checker.analyze_pii_block_categories(
    "123 Main St, Phone: 555-5555",
    keep_categories=["Email", "PhoneNumber"]
)
```

---

### 3. pii_modify.py

**PiiModifier** - Redacts PII from generated text.

#### Methods

##### `redact(text, language='en', redact_categories=None)`
Redact PII entities from text.

**Parameters:**
- `text` (str): Generated text to redact
- `language` (str): Language code. Default: 'en'
- `redact_categories` (List[str], optional): Specific categories to redact. Default: None (all)

**Returns:**
```python
{
    'redacted_text': 'Email: ********, Phone: ********',
    'time_taken': 0.31
}
```

**Example:**
```python
from pii_modify import PiiModifier

modifier = PiiModifier(endpoint, key)
result = modifier.redact("Email: john@example.com, Phone: 555-5555")
print(result['redacted_text'])
```

---

### 4. groundedness_check.py

**GroundednessChecker** - Checks if generated text is grounded in source documents.

#### Methods

##### `__init__(endpoint, subscription_key, ...)`
Initialize the groundedness checker.

**Parameters:**
- `endpoint` (str): Azure Content Safety endpoint
- `subscription_key` (str): Azure subscription key
- `api_version` (str, optional): API version (default: "2024-09-15-preview")
- `domain` (DomainType, optional): `GENERIC` or `MEDICAL` (default: `GENERIC`)
- `task_type` (TaskType, optional): `SUMMARIZATION` or `QNA` (default: `SUMMARIZATION`)

##### `check_groundedness(user_query, documents, content_text=None, reasoning=False)`
Check if generated text is grounded in the provided documents.

**Parameters:**
- `user_query` (str): The generated text/query to check for groundedness
- `documents` (List[str]): List of source document strings to use as grounding sources
- `content_text` (str, optional): Optional content/text to analyze (if None, uses user_query)
- `reasoning` (bool): Whether to include reasoning. Default: False

**Returns:**
```python
{
    'groundedness_check': {
        'input_text': 'generated text',
        'groundedness_detected': True/False,  # True if ungrounded detected
        'time_taken': 1.23
    }
}
```

**Important Notes:**
- `groundedness_detected=True` means ungrounded content was detected (bad)
- `groundedness_detected=False` means content is grounded (good) OR error occurred

**Example:**
```python
from groundedness_check import GroundednessChecker, DomainType, TaskType

checker = GroundednessChecker(
    endpoint, 
    key,
    domain=DomainType.GENERIC,
    task_type=TaskType.SUMMARIZATION
)

result = checker.check_groundedness(
    "The main topic is artificial intelligence...",
    documents=["Document about AI...", "Additional context..."]
)
```

#### Enums

**DomainType:**
- `GENERIC` - Generic domain (default)
- `MEDICAL` - Medical domain

**TaskType:**
- `SUMMARIZATION` - Summarization task (default)
- `QNA` - Question and Answer task

---

## Result Summary

The orchestrator returns a summary indicating which checks passed or failed:

```python
{
    'summary': {
        'output_check_passed': False,
        'failed_functions': ['pii_check', 'groundedness_check']
    }
}
```

### Detection Flags

Functions fail if these conditions are met:

- **pii_check**: `pii_detected == True` (PII entities found)
- **groundedness_check**: `groundedness_detected == False` (when enabled, ungrounded content detected or check failed)

**Note on Groundedness**: The orchestrator will mark groundedness as failed if:
- `groundedness_detected` is `False` (meaning not grounded) when the check was enabled
- `groundedness_detected` is `True` (meaning ungrounded detected)

## Execution Modes

### Parallel Execution (Default)
All selected functions run concurrently for better performance:
```python
result = orchestrator.analyze(
    generated_text="Response",
    source_documents=["Doc"],
    execution_mode='parallel'
)
```

### Sequential Execution
Functions run one after another:
```python
result = orchestrator.analyze(
    generated_text="Response",
    source_documents=["Doc"],
    execution_mode='sequential'
)
```

### Async Execution
Non-blocking execution for async applications:
```python
# Jupyter notebook or async function
result = await orchestrator.analyze_async(
    generated_text="Response",
    source_documents=["Doc"],
    execution_mode='parallel'
)
```

## Automatic Function Skipping

The orchestrator automatically skips `groundedness_check` if:
- `source_documents` is `None`
- `source_documents` is an empty list
- No documents are provided

In such cases, only PII check will run.

## Error Handling

All modules include error handling:
- API errors return default safe values
- Exceptions are caught and included in results
- Failed functions are listed in `failed_functions`
- Groundedness errors default to assuming content is grounded (safe default)

## Notes

- **Jupyter Notebooks**: Use `await` directly instead of `asyncio.run()`
- **Timing**: All operations include `time_taken` in results
- **Groundedness**: Requires source documents to function
- **PII Categories**: Available categories depend on Azure Text Analytics language support
- **Groundedness Detection**: The check verifies if generated content is properly grounded in source documents

## License

Copyright (c) Microsoft. All rights reserved.

