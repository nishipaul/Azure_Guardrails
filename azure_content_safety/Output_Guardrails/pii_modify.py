from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from typing import Dict, Any, List, Optional
import time


class PiiModifier:
    def __init__(self, endpoint: str, subscription_key: str):
        self.client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(subscription_key)
        )

    def redact(
        self,
        text: str,
        language: str = "en",
        redact_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        response = self.client.recognize_pii_entities(
            [text], language=language, categories_filter=redact_categories
        )
        doc = response[0]
        end_time = time.time()

        if doc.is_error:
            raise ValueError(f"Error in PII redaction: {doc.error}")

        return {
            'redacted_text': getattr(doc, 'redacted_text', None),
            'time_taken': end_time - start_time
        }

