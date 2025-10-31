import json
import time
import requests
from typing import Dict, Any


class PromptInjectionDetector:
    def __init__(self, endpoint: str, subscription_key: str, api_version: str = "2024-09-01") -> None:
        self.endpoint = endpoint
        self.subscription_key = subscription_key
        self.api_version = api_version

    def _build_url(self) -> str:
        return f"{self.endpoint}/contentsafety/text:shieldPrompt?api-version={self.api_version}"

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.subscription_key,
        }

    def _body(self, user_input: str) -> Dict[str, Any]:
        return {
            "userPrompt": user_input,
            "documents": []
        }

    def analyze(self, user_input: str) -> Dict[str, Any]:
        start_time = time.time()

        url = self._build_url()
        payload = json.dumps(self._body(user_input))
        resp = requests.post(url, headers=self._headers(), data=payload)
        data = {}
        try:
            data = resp.json()
        except Exception:
            data = {}

        if resp.status_code != 200:
            end_time = time.time()
            return {
                'prompt_injection': {
                    'input_text': user_input,
                    'detected': False,
                    'time_taken': end_time - start_time,
                }
            }

        # Heuristic over potential response fields for prompt-injection detection
        detected = False
        if isinstance(data, dict):
            detected = bool(
                data.get('attackDetected')
                or data.get('promptInjectionDetected')
                or data.get('isPromptInjection')
                or (isinstance(data.get('analysis'), dict) and data['analysis'].get('promptInjectionDetected'))
            )

        end_time = time.time()
        return {
            'prompt_injection': {
                'input_text': user_input,
                'detected': bool(detected),
                'time_taken': end_time - start_time,
            }
        }
