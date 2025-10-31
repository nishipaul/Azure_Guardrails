from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from typing import Dict, Any
import time


class SentimentAnalyzer:
    def __init__(self, endpoint: str, subscription_key: str):
        """
        Initialize the Sentiment Analyzer.
        
        Args:
            endpoint (str): The Azure Text Analytics endpoint URL.
            subscription_key (str): The Azure subscription key.
        """
        self.client = TextAnalyticsClient(endpoint=endpoint,  credential=AzureKeyCredential(subscription_key))


        
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text and return confidence scores.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            Dict[str, float]: Dictionary containing sentiment scores: 'positive', 'neutral', 'negative', 'overall_sentiment'
        """
        try:
            start_time = time.time()
            result = self.client.analyze_sentiment([text], show_opinion_mining=False)
            doc = result[0]
            end_time = time.time()
            
            if doc.is_error:
                raise ValueError(f"Error analyzing sentiment: {doc.error}")
            
            # Extract confidence scores
            overall_sentiment = doc.sentiment
            negative_sentiment_detected = (overall_sentiment.lower() == 'negative') if overall_sentiment else False
            
            result_dict = {
                'sentiment_analysis': {
                    'input_text': text,
                    'sentiment_categories': {
                        'positive': doc.confidence_scores.positive,
                        'neutral': doc.confidence_scores.neutral,
                        'negative': doc.confidence_scores.negative,
                        'overall_sentiment': overall_sentiment,
                    },
                    'time_taken': end_time - start_time,
                }
            }
            
            if negative_sentiment_detected:
                result_dict['sentiment_analysis']['negative_sentiment_detected'] = True
            
            return result_dict
            
        except Exception as e:
            raise ValueError(f"Failed to analyze sentiment: {str(e)}")

