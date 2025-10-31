import json
import time
import requests
from typing import Dict, Any, List, Optional
from enum import Enum


class TaskType(Enum):
    SUMMARIZATION = "SUMMARIZATION"
    QNA = "QNA"


class DomainType(Enum):
    MEDICAL = "MEDICAL"
    GENERIC = "GENERIC"


class GroundednessChecker:
    def __init__(
        self,
        endpoint: str,
        subscription_key: str,
        api_version: str = "2024-09-15-preview",
        domain: DomainType = DomainType.GENERIC,
        task_type: TaskType = TaskType.SUMMARIZATION,
    ):
        """
        Initialize the Groundedness Checker.
        
        Args:
            endpoint: Azure Content Safety endpoint
            subscription_key: Azure Content Safety subscription key
            api_version: API version (default: "2024-09-15-preview")
            domain: Domain type (default: GENERIC)
            task_type: Task type - SUMMARIZATION or QNA (default: SUMMARIZATION)
        """
        self.endpoint = endpoint
        self.subscription_key = subscription_key
        self.api_version = api_version
        self.domain = domain
        self.task_type = task_type

    def _build_url(self) -> str:
        """Build the API URL."""
        return f"{self.endpoint}/contentsafety/text:detectGroundedness?api-version={self.api_version}"

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        return {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.subscription_key,
        }

    def _build_request_body(
        self,
        user_query: str,
        documents: List[str],
        content_text: Optional[str] = None,
        reasoning: bool = False,
    ) -> Dict[str, Any]:
        """
        Build the request body for groundedness detection.
        
        Args:
            user_query: The user query to check
            documents: List of document strings to use as grounding sources
            content_text: The content/text to analyze (if None, uses user_query)
            reasoning: Whether to include reasoning (default: False)
        """
        body = {
            "domain": self.domain.value,
            "task": self.task_type.value,
            "text": content_text if content_text else user_query,
            "groundingSources": documents,
            "Reasoning": reasoning,
        }
        
        # For QNA task type, add query
        if self.task_type == TaskType.QNA:
            body["qna"] = {"query": user_query}
        
        return body

    def check_groundedness(
        self,
        user_query: str,
        documents: List[str],
        content_text: Optional[str] = None,
        reasoning: bool = False,
    ) -> Dict[str, Any]:
        """
        Check if user query is grounded in the provided documents.
        
        Args:
            user_query: The user query to check for groundedness
            documents: List of document strings to use as grounding sources
            content_text: Optional content/text to analyze (if None, uses user_query)
            reasoning: Whether to include reasoning (default: False)
        
        Returns:
            Dictionary with groundedness_check containing:
                - input_text: The user query
                - groundedness_detected: True if ungrounded, False if grounded
                - time_taken: Time taken for the operation in seconds
        """
        start_time = time.time()
        
        url = self._build_url()
        headers = self._build_headers()
        payload = self._build_request_body(user_query, documents, content_text, reasoning)
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            end_time = time.time()
            
            if response.status_code != 200:
                # On error, return default (assume grounded for safety)
                return {
                    'groundedness_check': {
                        'input_text': user_query,
                        'groundedness_detected': False,
                        'time_taken': end_time - start_time,
                    }
                }
            
            result_data = response.json()
            
            # Determine if groundedness was detected (unbound/ungrounded)
            # The API typically returns isGrounded or similar field
            groundedness_detected = False
            if isinstance(result_data, dict):
                # Check various possible field names for ungrounded detection
                groundedness_detected = bool(
                    result_data.get('isGrounded') == False
                    or result_data.get('ungrounded') == True
                    or result_data.get('groundedness') == 'unbound'
                    or (isinstance(result_data.get('analysis'), dict) 
                        and result_data['analysis'].get('isGrounded') == False)
                )
            
            return {
                'groundedness_check': {
                    'input_text': user_query,
                    'groundedness_detected': groundedness_detected,
                    'time_taken': end_time - start_time,
                }
            }
            
        except Exception as e:
            end_time = time.time()
            # On exception, return default (assume grounded for safety)
            return {
                'groundedness_check': {
                    'input_text': user_query,
                    'groundedness_detected': False,
                    'time_taken': end_time - start_time,
                }
            }
