
import json
import os
import time
import requests
from typing import List, Dict


class DetectionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        return f"DetectionError(code={self.code}, message={self.message})"


class ContentSafety(object):
    def __init__(self, endpoint: str, subscription_key: str, api_version: str) -> None:
        self.endpoint = endpoint
        self.subscription_key = subscription_key
        self.api_version = api_version

    def _build_url(self) -> str:
        return f"{self.endpoint}/contentsafety/text:analyze?api-version={self.api_version}"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/json",
        }

    def _build_request_body(self, text: str) -> Dict:
        return {"text": text}

    def detect_text(self, text: str) -> Dict:
        url = self._build_url()
        headers = self._build_headers()
        payload = json.dumps(self._build_request_body(text))

        response = requests.post(url, headers=headers, data=payload)
        res_content = response.json()

        if response.status_code != 200:
            raise DetectionError(
                res_content.get("error", {}).get("code", "UnknownError"),
                res_content.get("error", {}).get("message", "Unknown error"),
            )

        return res_content

    def analyze_text(self, input_text: str, threshold: int = 2) -> Dict:
        if threshold not in (2, 4, 6):
            raise ValueError("threshold must be one of 2, 4, 6")

        start_time = time.time()
        detection_result = self.detect_text(input_text)
        end_time = time.time()

        categories_analysis: List[Dict] = detection_result.get("categoriesAnalysis", [])
        categories: Dict[str, int] = {}
        detected_categories: List[str] = []
        
        for item in categories_analysis:
            name = item.get("category")
            severity = item.get("severity")
            if name is None or severity is None:
                continue
            categories[name] = severity
            if severity > 0:
                detected_categories.append(name)
        
        result = {
            "content_safety": {
                "input_text": input_text,
                "categories": categories,
                "time_taken": end_time - start_time,
            }
        }
        
        if detected_categories:
            result["content_safety"]["content_safety_detected"] = True
            result["content_safety"]["detected_categories"] = detected_categories
        
        return result


    

