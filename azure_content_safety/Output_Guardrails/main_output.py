import os
from typing import Dict, Any, List, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import asyncio
from datetime import datetime

from .pii_check import PIIChecker
from .groundedness_check import GroundednessChecker, DomainType, TaskType


class OutputGuardrailOrchestrator:
    """
    Orchestrator class to run output guardrail checks (PII, groundedness)
    with options for parallel/sequential execution and selective function execution.
    """
    
    # Available function keys
    FUNCTIONS = {
        'pii_check': 'pii_check',
        'groundedness_check': 'groundedness_check'
    }
    
    # Default execution order
    DEFAULT_ORDER = ['pii_check', 'groundedness_check']
    
    def __init__(
        self,
        endpoint: str,
        subscription_key: str,
        groundedness_api_version: str = "2024-09-15-preview",
        domain: DomainType = DomainType.GENERIC,
        task_type: TaskType = TaskType.SUMMARIZATION,
    ):
        """
        Initialize the orchestrator with endpoint and subscription key.
        
        Args:
            endpoint: Azure endpoint (used for all services)
            subscription_key: Azure subscription key (used for all services)
            groundedness_api_version: Groundedness API version
            domain: Domain type for groundedness (default: GENERIC)
            task_type: Task type for groundedness (default: SUMMARIZATION)
        """
        self.groundedness_api_version = groundedness_api_version
        self.domain = domain
        self.task_type = task_type
        
        # Initialize all clients with endpoint and key
        self.pii_checker = PIIChecker(endpoint, subscription_key)
        self.groundedness_checker = GroundednessChecker(
            endpoint, subscription_key, groundedness_api_version, domain, task_type
        )
    
    def _run_pii_check(
        self,
        text: str,
        enable_pii_modify: bool = False,
        keep_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run PII detection."""
        if keep_categories is not None:
            return self.pii_checker.analyze_pii_block_categories(
                text, keep_categories=keep_categories, enable_pii_modify=enable_pii_modify
            )
        return self.pii_checker.analyze_pii(text, enable_pii_modify=enable_pii_modify)
    
    def _run_groundedness_check(
        self,
        text: str,
        documents: List[str],
        content_text: Optional[str] = None,
        reasoning: bool = False,
    ) -> Dict[str, Any]:
        """Run groundedness detection."""
        return self.groundedness_checker.check_groundedness(
            text, documents, content_text, reasoning
        )
    
    def _get_function_call(self, func_name: str):
        """Get the function callable by name."""
        func_map = {
            'pii_check': self._run_pii_check,
            'groundedness_check': self._run_groundedness_check,
        }
        return func_map.get(func_name)
    
    def analyze(
        self,
        generated_text: str,
        source_documents: Optional[List[str]] = None,
        functions: Optional[List[str]] = None,
        execution_mode: Literal['parallel', 'sequential'] = 'parallel',
        # PII options
        enable_pii_modify: bool = False,
        keep_categories: Optional[List[str]] = None,
        # Groundedness options
        content_text: Optional[str] = None,
        reasoning: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze generated text through selected output guardrail functions.
        
        Args:
            generated_text: The generated text to analyze
            source_documents: List of source documents for groundedness check (optional).
                           If None, groundedness_check will be skipped.
            functions: List of function names to execute (default: all).
                      Valid options: 'pii_check', 'groundedness_check'
            execution_mode: 'parallel' or 'sequential' (default: 'parallel')
            enable_pii_modify: Enable PII redaction (default: False)
            keep_categories: Categories to keep when blocking PII (default: None, uses analyze_pii)
            content_text: Optional content text for groundedness (default: None, uses generated_text)
            reasoning: Whether to include reasoning in groundedness (default: False)
        
        Returns:
            Dictionary containing results from all executed functions, plus summary with pass/fail status,
            all wrapped in query_timestamp
        """
        # Auto-exclude groundedness if no source documents provided
        if source_documents is None or len(source_documents) == 0:
            if functions is None:
                functions = ['pii_check']
            elif 'groundedness_check' in functions:
                functions = [f for f in functions if f != 'groundedness_check']
        
        # Validate and set default functions
        if functions is None:
            functions = self.DEFAULT_ORDER.copy()
        else:
            # Validate function names
            invalid = [f for f in functions if f not in self.FUNCTIONS]
            if invalid:
                raise ValueError(f"Invalid function names: {invalid}. Valid options: {list(self.FUNCTIONS.keys())}")
            # Maintain order based on DEFAULT_ORDER
            functions = [f for f in self.DEFAULT_ORDER if f in functions]
        
        if not functions:
            raise ValueError("At least one function must be selected")
        
        # Prepare function calls with their arguments
        func_calls = []
        for func_name in functions:
            func = self._get_function_call(func_name)
            if func_name == 'pii_check':
                args = [generated_text]
                kwargs = {'enable_pii_modify': enable_pii_modify}
                if keep_categories is not None:
                    kwargs['keep_categories'] = keep_categories
                func_calls.append((func_name, func, args, kwargs))
            elif func_name == 'groundedness_check':
                if source_documents is None or len(source_documents) == 0:
                    continue  # Skip if no documents
                args = [generated_text, source_documents]
                kwargs = {}
                if content_text is not None:
                    kwargs['content_text'] = content_text
                if reasoning:
                    kwargs['reasoning'] = reasoning
                func_calls.append((func_name, func, args, kwargs))
        
        if not func_calls:
            raise ValueError("No valid functions to execute")
        
        # Execute functions
        if execution_mode == 'parallel':
            results = self._execute_parallel(func_calls)
        else:
            results = self._execute_sequential(func_calls)
        
        # Determine which functions failed based on detection flags
        failed_functions = []
        
        # Check for errors first
        for func_name, result in results.items():
            if isinstance(result, dict):
                func_result = result.get(func_name, {})
                if 'error' in func_result:
                    failed_functions.append(func_name)
        
        # Check detection flags
        for func_name, result in results.items():
            if isinstance(result, dict):
                # Check pii_check
                if 'pii_check' in result:
                    pii_result = result.get('pii_check', {})
                    if pii_result.get('pii_detected', False):
                        if 'pii_check' not in failed_functions:
                            failed_functions.append('pii_check')
                
                # Check groundedness_check
                # If groundedness was enabled and executed, check the result
                if 'groundedness_check' in result:
                    groundedness_result = result.get('groundedness_check', {})
                    groundedness_detected = groundedness_result.get('groundedness_detected', False)
                    # If groundedness check was enabled and groundedness_detected is False, fail
                    if groundedness_detected is False:
                        if 'groundedness_check' not in failed_functions:
                            failed_functions.append('groundedness_check')
                    elif groundedness_detected is True:
                        # Also fail if True (ungrounded detected)
                        if 'groundedness_check' not in failed_functions:
                            failed_functions.append('groundedness_check')
        
        output_check_passed = len(failed_functions) == 0
        
        # Combine results into single dictionary
        combined_result = {}
        for func_name, result in results.items():
            combined_result.update(result)
        
        # Add summary
        combined_result['summary'] = {
            'output_check_passed': output_check_passed,
            'failed_functions': failed_functions
        }
        
        # Wrap everything in query_timestamp with timestamp value
        query_timestamp_value = datetime.now().isoformat()
        return {
            'query_timestamp': {
                'timestamp': query_timestamp_value,
                **combined_result
            }
        }
    
    def _execute_parallel(self, func_calls: List[tuple]) -> Dict[str, Dict[str, Any]]:
        """Execute functions in parallel using ThreadPoolExecutor."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(func_calls)) as executor:
            future_to_func = {
                executor.submit(func, *args, **kwargs): func_name
                for func_name, func, args, kwargs in func_calls
            }
            
            for future in as_completed(future_to_func):
                func_name = future_to_func[future]
                try:
                    result = future.result()
                    results[func_name] = result
                except Exception as e:
                    # On error, return error info in the result structure
                    results[func_name] = {
                        func_name: {
                            'error': str(e),
                            'input_text': func_calls[0][2][0] if func_calls else ''
                        }
                    }
        
        return results
    
    def _execute_sequential(self, func_calls: List[tuple]) -> Dict[str, Dict[str, Any]]:
        """Execute functions sequentially."""
        results = {}
        
        for func_name, func, args, kwargs in func_calls:
            try:
                result = func(*args, **kwargs)
                results[func_name] = result
            except Exception as e:
                # On error, return error info in the result structure
                results[func_name] = {
                    func_name: {
                        'error': str(e),
                        'input_text': args[0] if args else ''
                    }
                }
        
        return results
    
    async def analyze_async(
        self,
        generated_text: str,
        source_documents: Optional[List[str]] = None,
        functions: Optional[List[str]] = None,
        execution_mode: Literal['parallel', 'sequential'] = 'parallel',
        # PII options
        enable_pii_modify: bool = False,
        keep_categories: Optional[List[str]] = None,
        # Groundedness options
        content_text: Optional[str] = None,
        reasoning: bool = False,
    ) -> Dict[str, Any]:
        """
        Async version: Analyze generated text through selected output guardrail functions.
        
        Args:
            generated_text: The generated text to analyze
            source_documents: List of source documents for groundedness check (optional).
                           If None, groundedness_check will be skipped.
            functions: List of function names to execute (default: all).
                      Valid options: 'pii_check', 'groundedness_check'
            execution_mode: 'parallel' or 'sequential' (default: 'parallel')
            enable_pii_modify: Enable PII redaction (default: False)
            keep_categories: Categories to keep when blocking PII (default: None, uses analyze_pii)
            content_text: Optional content text for groundedness (default: None, uses generated_text)
            reasoning: Whether to include reasoning in groundedness (default: False)
        
        Returns:
            Dictionary containing results from all executed functions, plus summary with pass/fail status,
            all wrapped in query_timestamp
        """
        # Auto-exclude groundedness if no source documents provided
        if source_documents is None or len(source_documents) == 0:
            if functions is None:
                functions = ['pii_check']
            elif 'groundedness_check' in functions:
                functions = [f for f in functions if f != 'groundedness_check']
        
        # Validate and set default functions
        if functions is None:
            functions = self.DEFAULT_ORDER.copy()
        else:
            # Validate function names
            invalid = [f for f in functions if f not in self.FUNCTIONS]
            if invalid:
                raise ValueError(f"Invalid function names: {invalid}. Valid options: {list(self.FUNCTIONS.keys())}")
            # Maintain order based on DEFAULT_ORDER
            functions = [f for f in self.DEFAULT_ORDER if f in functions]
        
        if not functions:
            raise ValueError("At least one function must be selected")
        
        # Prepare function calls with their arguments
        func_calls = []
        for func_name in functions:
            func = self._get_function_call(func_name)
            if func_name == 'pii_check':
                args = [generated_text]
                kwargs = {'enable_pii_modify': enable_pii_modify}
                if keep_categories is not None:
                    kwargs['keep_categories'] = keep_categories
                func_calls.append((func_name, func, args, kwargs))
            elif func_name == 'groundedness_check':
                if source_documents is None or len(source_documents) == 0:
                    continue  # Skip if no documents
                args = [generated_text, source_documents]
                kwargs = {}
                if content_text is not None:
                    kwargs['content_text'] = content_text
                if reasoning:
                    kwargs['reasoning'] = reasoning
                func_calls.append((func_name, func, args, kwargs))
        
        if not func_calls:
            raise ValueError("No valid functions to execute")
        
        # Execute functions asynchronously
        if execution_mode == 'parallel':
            results = await self._execute_parallel_async(func_calls)
        else:
            results = await self._execute_sequential_async(func_calls)
        
        # Determine which functions failed based on detection flags
        failed_functions = []
        
        # Check for errors first
        for func_name, result in results.items():
            if isinstance(result, dict):
                func_result = result.get(func_name, {})
                if 'error' in func_result:
                    failed_functions.append(func_name)
        
        # Check detection flags
        for func_name, result in results.items():
            if isinstance(result, dict):
                # Check pii_check
                if 'pii_check' in result:
                    pii_result = result.get('pii_check', {})
                    if pii_result.get('pii_detected', False):
                        if 'pii_check' not in failed_functions:
                            failed_functions.append('pii_check')
                
                # Check groundedness_check
                # If groundedness was enabled and executed, check the result
                if 'groundedness_check' in result:
                    groundedness_result = result.get('groundedness_check', {})
                    groundedness_detected = groundedness_result.get('groundedness_detected', False)
                    # If groundedness check was enabled and groundedness_detected is False, fail
                    if groundedness_detected is False:
                        if 'groundedness_check' not in failed_functions:
                            failed_functions.append('groundedness_check')
                    elif groundedness_detected is True:
                        # Also fail if True (ungrounded detected)
                        if 'groundedness_check' not in failed_functions:
                            failed_functions.append('groundedness_check')
        
        output_check_passed = len(failed_functions) == 0
        
        # Combine results into single dictionary
        combined_result = {}
        for func_name, result in results.items():
            combined_result.update(result)
        
        # Add summary
        combined_result['summary'] = {
            'output_check_passed': output_check_passed,
            'failed_functions': failed_functions
        }
        
        # Wrap everything in query_timestamp with timestamp value
        query_timestamp_value = datetime.now().isoformat()
        return {
            'query_timestamp': {
                'timestamp': query_timestamp_value,
                **combined_result
            }
        }
    
    async def _execute_parallel_async(self, func_calls: List[tuple]) -> Dict[str, Dict[str, Any]]:
        """Execute functions in parallel asynchronously using asyncio.gather."""
        async def run_func(func_name: str, func, args: List, kwargs: Dict) -> tuple:
            try:
                # Run synchronous function in thread pool
                if hasattr(asyncio, 'to_thread'):
                    result = await asyncio.to_thread(func, *args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                return (func_name, result)
            except Exception as e:
                return (func_name, {
                    func_name: {
                        'error': str(e),
                        'input_text': args[0] if args else ''
                    }
                })
        
        # Create tasks for all functions
        tasks = [run_func(func_name, func, args, kwargs) for func_name, func, args, kwargs in func_calls]
        
        # Execute all tasks in parallel
        results_list = await asyncio.gather(*tasks)
        
        # Convert to dictionary
        results = {func_name: result for func_name, result in results_list}
        return results
    
    async def _execute_sequential_async(self, func_calls: List[tuple]) -> Dict[str, Dict[str, Any]]:
        """Execute functions sequentially asynchronously."""
        results = {}
        
        for func_name, func, args, kwargs in func_calls:
            try:
                # Run synchronous function in thread pool
                if hasattr(asyncio, 'to_thread'):
                    result = await asyncio.to_thread(func, *args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                results[func_name] = result
            except Exception as e:
                results[func_name] = {
                    func_name: {
                        'error': str(e),
                        'input_text': args[0] if args else ''
                    }
                }
        
        return results


def create_orchestrator_from_env() -> OutputGuardrailOrchestrator:
    """
    Convenience function to create orchestrator from environment variables.
    
    Expected environment variables:
        - endpoint or CONTENT_SAFETY_ENDPOINT
        - subscription_key or CONTENT_SAFETY_KEY
    """
    endpoint = os.getenv("endpoint") or os.getenv("CONTENT_SAFETY_ENDPOINT")
    subscription_key = os.getenv("subscription_key") or os.getenv("CONTENT_SAFETY_KEY")
    
    return OutputGuardrailOrchestrator(endpoint=endpoint, subscription_key=subscription_key)

