import os
from typing import Dict, Any, List, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import asyncio
from datetime import datetime

from .content_safety import ContentSafety
from .sentiment_check import SentimentAnalyzer
from .pii_check import PIIChecker
from .prompt_injection import PromptInjectionDetector


class InputGuardrailOrchestrator:
    """
    Orchestrator class to run input guardrail checks (content safety, sentiment, PII, prompt injection)
    with options for parallel/sequential execution and selective function execution.
    """
    
    # Available function keys
    FUNCTIONS = {
        'content_safety': 'content_safety',
        'sentiment_check': 'sentiment_check',
        'pii_check': 'pii_check',
        'prompt_injection': 'prompt_injection'
    }
    
    # Default execution order
    DEFAULT_ORDER = ['content_safety', 'sentiment_check', 'pii_check', 'prompt_injection']
    
    def __init__(
        self,
        endpoint: str,
        subscription_key: str,
        content_safety_api_version: str = "2024-09-01",
        prompt_injection_api_version: str = "2024-09-01",
    ):
        """
        Initialize the orchestrator with endpoint and subscription key.
        
        Args:
            endpoint: Azure endpoint (used for all services)
            subscription_key: Azure subscription key (used for all services)
            content_safety_api_version: Content Safety API version
            prompt_injection_api_version: Prompt Injection API version
        """
        self.content_safety_api_version = content_safety_api_version
        self.prompt_injection_api_version = prompt_injection_api_version
        
        # Initialize all clients with endpoint and key
        self.content_safety = ContentSafety(endpoint, subscription_key, self.content_safety_api_version)
        self.sentiment_analyzer = SentimentAnalyzer(endpoint, subscription_key)
        self.pii_checker = PIIChecker(endpoint, subscription_key)
        self.prompt_injection_detector = PromptInjectionDetector(
            endpoint, subscription_key, self.prompt_injection_api_version
        )
    
    def _run_content_safety(self, query: str, threshold: int = 2) -> Dict[str, Any]:
        """Run content safety analysis."""
        return self.content_safety.analyze_text(query, threshold=threshold)
    
    def _run_sentiment_check(self, query: str) -> Dict[str, Any]:
        """Run sentiment analysis."""
        return self.sentiment_analyzer.analyze_sentiment(query)
    
    def _run_pii_check(
        self,
        query: str,
        enable_pii_modify: bool = False,
        keep_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run PII detection."""
        if keep_categories is not None:
            return self.pii_checker.analyze_pii_block_categories(
                query, keep_categories=keep_categories, enable_pii_modify=enable_pii_modify
            )
        return self.pii_checker.analyze_pii(query, enable_pii_modify=enable_pii_modify)
    
    def _run_prompt_injection(self, query: str) -> Dict[str, Any]:
        """Run prompt injection detection."""
        return self.prompt_injection_detector.analyze(query)
    
    def _get_function_call(self, func_name: str):
        """Get the function callable by name."""
        func_map = {
            'content_safety': self._run_content_safety,
            'sentiment_check': self._run_sentiment_check,
            'pii_check': self._run_pii_check,
            'prompt_injection': self._run_prompt_injection,
        }
        return func_map.get(func_name)
    
    def analyze(
        self,
        query: str,
        functions: Optional[List[str]] = None,
        execution_mode: Literal['parallel', 'sequential'] = 'parallel',
        # Content safety options
        content_safety_threshold: int = 2,
        # PII Other options
        enable_pii_modify: bool = False,
        keep_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze input query through selected guardrail functions.
        
        Args:
            query: User input text to analyze
            functions: List of function names to execute (default: all).
                      Valid options: 'content_safety', 'sentiment_check', 'pii_check', 'prompt_injection'
            execution_mode: 'parallel' or 'sequential' (default: 'parallel')
            content_safety_threshold: Threshold for content safety (2, 4, or 6, default: 2)
            enable_pii_modify: Enable PII redaction (default: False)
            keep_categories: Categories to keep when blocking PII (default: None, uses analyze_pii)
        
        Returns:
            Dictionary containing results from all executed functions, plus summary with pass/fail status
        """
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
            if func_name == 'content_safety':
                func_calls.append((func_name, func, [query, content_safety_threshold], {}))
            elif func_name == 'pii_check':
                args = [query]
                kwargs = {'enable_pii_modify': enable_pii_modify}
                if keep_categories is not None:
                    kwargs['keep_categories'] = keep_categories
                func_calls.append((func_name, func, args, kwargs))
            else:
                func_calls.append((func_name, func, [query], {}))
        
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
                # Check sentiment_analysis (func_name is 'sentiment_check' but result key is 'sentiment_analysis')
                if 'sentiment_analysis' in result:
                    sentiment_result = result.get('sentiment_analysis', {})
                    if sentiment_result.get('negative_sentiment_detected', False):
                        if 'sentiment_analysis' not in failed_functions:
                            failed_functions.append('sentiment_analysis')
                
                # Check content_safety
                if 'content_safety' in result:
                    content_safety_result = result.get('content_safety', {})
                    if content_safety_result.get('content_safety_detected', False):
                        if 'content_safety' not in failed_functions:
                            failed_functions.append('content_safety')
                
                # Check pii_check
                if 'pii_check' in result:
                    pii_result = result.get('pii_check', {})
                    if pii_result.get('pii_detected', False):
                        if 'pii_check' not in failed_functions:
                            failed_functions.append('pii_check')
                
                # Check prompt_injection
                if 'prompt_injection' in result:
                    prompt_result = result.get('prompt_injection', {})
                    if prompt_result.get('detected', False):
                        if 'prompt_injection' not in failed_functions:
                            failed_functions.append('prompt_injection')
        
        all_passed = len(failed_functions) == 0
        
        # Combine results into single dictionary
        combined_result = {}
        for func_name, result in results.items():
            combined_result.update(result)
        
        # Add summary
        combined_result['summary'] = {
            'all_passed': all_passed,
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
    
    async def analyze_async(
        self,
        query: str,
        functions: Optional[List[str]] = None,
        execution_mode: Literal['parallel', 'sequential'] = 'parallel',
        # Content safety options
        content_safety_threshold: int = 2,
        # PII Other options
        enable_pii_modify: bool = False,
        keep_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Async version: Analyze input query through selected guardrail functions.
        
        Args:
            query: User input text to analyze
            functions: List of function names to execute (default: all).
                      Valid options: 'content_safety', 'sentiment_check', 'pii_check', 'prompt_injection'
            execution_mode: 'parallel' or 'sequential' (default: 'parallel')
            content_safety_threshold: Threshold for content safety (2, 4, or 6, default: 2)
            enable_pii_modify: Enable PII redaction (default: False)
            keep_categories: Categories to keep when blocking PII (default: None, uses analyze_pii)
        
        Returns:
            Dictionary containing results from all executed functions, plus summary with pass/fail status
        """
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
            if func_name == 'content_safety':
                func_calls.append((func_name, func, [query, content_safety_threshold], {}))
            elif func_name == 'pii_check':
                args = [query]
                kwargs = {'enable_pii_modify': enable_pii_modify}
                if keep_categories is not None:
                    kwargs['keep_categories'] = keep_categories
                func_calls.append((func_name, func, args, kwargs))
            else:
                func_calls.append((func_name, func, [query], {}))
        
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
                # Check sentiment_analysis (func_name is 'sentiment_check' but result key is 'sentiment_analysis')
                if 'sentiment_analysis' in result:
                    sentiment_result = result.get('sentiment_analysis', {})
                    if sentiment_result.get('negative_sentiment_detected', False):
                        if 'sentiment_analysis' not in failed_functions:
                            failed_functions.append('sentiment_analysis')
                
                # Check content_safety
                if 'content_safety' in result:
                    content_safety_result = result.get('content_safety', {})
                    if content_safety_result.get('content_safety_detected', False):
                        if 'content_safety' not in failed_functions:
                            failed_functions.append('content_safety')
                
                # Check pii_check
                if 'pii_check' in result:
                    pii_result = result.get('pii_check', {})
                    if pii_result.get('pii_detected', False):
                        if 'pii_check' not in failed_functions:
                            failed_functions.append('pii_check')
                
                # Check prompt_injection
                if 'prompt_injection' in result:
                    prompt_result = result.get('prompt_injection', {})
                    if prompt_result.get('detected', False):
                        if 'prompt_injection' not in failed_functions:
                            failed_functions.append('prompt_injection')
        
        all_passed = len(failed_functions) == 0
        
        # Combine results into single dictionary
        combined_result = {}
        for func_name, result in results.items():
            combined_result.update(result)
        
        # Add summary
        combined_result['summary'] = {
            'all_passed': all_passed,
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


def create_orchestrator_from_env() -> InputGuardrailOrchestrator:
    """
    Convenience function to create orchestrator from environment variables.
    
    Expected environment variables:
        - endpoint or CONTENT_SAFETY_ENDPOINT
        - subscription_key or CONTENT_SAFETY_KEY
    """
    endpoint = os.getenv("endpoint") or os.getenv("CONTENT_SAFETY_ENDPOINT")
    subscription_key = os.getenv("subscription_key") or os.getenv("CONTENT_SAFETY_KEY")
    
    return InputGuardrailOrchestrator(endpoint=endpoint, subscription_key=subscription_key)

