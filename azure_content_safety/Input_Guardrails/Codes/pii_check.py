from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from typing import Dict, Any, List, Optional
import time
from .pii_modify import PiiModifier


class PIIChecker:
    def __init__(self, endpoint: str, subscription_key: str):
        self._endpoint = endpoint
        self._subscription_key = subscription_key
        self.client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(subscription_key)
        )

    def analyze_pii(
        self,
        text: str,
        language: str = "en",
        enable_pii_modify: bool = False,
    ) -> Dict[str, Any]:
        start_time = time.time()
        response = self.client.recognize_pii_entities([text], language=language)
        doc = response[0]
        end_time = time.time()

        if doc.is_error:
            raise ValueError(f"Error in PII analysis: {doc.error}")

        entities: List[Dict[str, Any]] = []
        for entity in doc.entities:
            entities.append({
                'entity': entity.text,
                'category': str(entity.category),
                'confidence_score': entity.confidence_score,
                'offset': entity.offset,
                'length': entity.length
            })

        pii_detected_count = len(entities)
        pii_detected = pii_detected_count > 0

        result = {
            'pii_check': {
                'input_text': text,
                'entities': entities,
                'pii_detected': pii_detected,
                'pii_detected_count': pii_detected_count,
                'time_taken': end_time - start_time
            }
        }

        # Optional redaction via PiiModifier (separate class/file)
        if enable_pii_modify:
            modifier = PiiModifier(self._endpoint, self._subscription_key)
            redaction = modifier.redact(text, language=language)
            result['pii_modify'] = {
                'enabled': True,
                'redacted_text': redaction.get('redacted_text'),
                'time_taken': redaction.get('time_taken'),
            }
        else:
            result['pii_modify'] = {'enabled': False}

        return result

    def analyze_pii_block_categories(
        self,
        text: str,
        keep_categories: Optional[List[str]],
        language: str = "en",
        enable_pii_modify: bool = False,
    ) -> Dict[str, Any]:

        start_time = time.time()
        response = self.client.recognize_pii_entities([text], language=language)
        doc = response[0]
        end_time = time.time()

        if doc.is_error:
            raise ValueError(f"Error in PII analysis: {doc.error}")

        keep_set = {c.lower() for c in (keep_categories or [])}

        entities_all: List[Dict[str, Any]] = []
        for entity in doc.entities:
            entities_all.append({
                'entity': entity.text,
                'category': str(entity.category),
                'confidence_score': entity.confidence_score,
                'offset': entity.offset,
                'length': entity.length
            })

        # Keep entities that are in the list
        entities: List[Dict[str, Any]] = [e for e in entities_all if e['category'].lower() not in keep_set]

        pii_detected_count = len(entities)
        pii_detected = pii_detected_count > 0

        result = {
            'pii_check': {
                'input_text': text,
                'entities': entities,
                'pii_detected': pii_detected,
                'pii_detected_count': pii_detected_count,
                'time_taken': end_time - start_time
            }
        }

        if enable_pii_modify:
            modifier = PiiModifier(self._endpoint, self._subscription_key)
            redact_categories = sorted({e['category'] for e in entities_all if e['category'].lower() not in keep_set})
            redaction = modifier.redact(text, language=language, redact_categories=redact_categories)
            result['pii_modify'] = {
                'enabled': True,
                'redacted_text': redaction.get('redacted_text'),
                'time_taken': redaction.get('time_taken'),
            }
        else:
            result['pii_modify'] = {'enabled': False}

        return result


