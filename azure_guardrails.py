# IMPORTING ALL THE REQUIRED LIBRARIES
# BLOCK START

import os
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from azure.ai.contentsafety import ContentSafetyClient, BlocklistClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import (
    AnalyzeTextOptions, 
    TextCategory, 
    TextBlocklist, 
    TextBlocklistItem, 
    AddOrUpdateTextBlocklistItemsOptions,
    RemoveTextBlocklistItemsOptions
)
from azure.core.exceptions import HttpResponseError

# BLOCK END



# Load environment variables
load_dotenv()






# CREATING DATACLASS BLUEPRINT FOR TENANT CONFIGURATION
# BLOCK START

@dataclass
class TenantConfig:
    """Configuration class for tenant-specific settings."""
    tenant_id: str
    endpoint: str
    api_key: str
    log_level: str = "INFO"
    log_file_prefix: str = "azure_guardrails"

# BLOCK END





# CREATING AZURE GUARDRAILS CLIENT CLASS WITH PRIVATE METHODS AND INITIALIZATION LOGIC
"""
    METHODS:
    - __init__
    - _load_tenant_configs_from_env
    - _initialize_tenant
    - _setup_tenant_logger
    - _validate_tenant
    - get_available_tenants
"""
# BLOCK START

class AzureGuardrailsClient:
    """
    Multi-tenant Azure Content Safety Client
    """

    # CONSTRUCTOR
    def __init__(self, auto_initialize: bool = False, initialize_tenants: Optional[List[str]] = None):
        """
        Initialize the Azure Guardrails Client with tenant configurations.
        
        Args:
            auto_initialize: If True, initializes all available tenants immediately. If False, uses lazy initialization (default).
            initialize_tenants: List of specific tenant IDs to initialize immediately. Only used if auto_initialize is False.
        """
        self.tenant_configs = self._load_tenant_configs_from_env()
        self.tenant_clients = {}
        self.tenant_loggers = {}
        
        # Initialize based on parameters
        if auto_initialize:
            # Initialize all tenants immediately (legacy behavior)
            for tenant_id, config in self.tenant_configs.items():
                self._initialize_tenant(tenant_id, config)
        elif initialize_tenants:
            # Initialize only specified tenants
            for tenant_id in initialize_tenants:
                if tenant_id in self.tenant_configs:
                    self._initialize_tenant(tenant_id, self.tenant_configs[tenant_id])
                else:
                    print(f"Warning: Tenant '{tenant_id}' not found in configurations. Available: {list(self.tenant_configs.keys())}")



    

    # LOAD TENANT CONFIGS FROM ENVIRONMENT VARIABLES
    def _load_tenant_configs_from_env(self) -> Dict[str, TenantConfig]:
        """
        Load tenant configurations from environment variables.
        Returns:
            Dictionary of tenant configurations
        """
        configs = {}
        
        # Look for tenant environment variables
        for key, value in os.environ.items():
            if key.startswith("TENANT_") and key.endswith("_ENDPOINT"):
                # Extract tenant ID from environment variable name
                tenant_id = key.replace("TENANT_", "").replace("_ENDPOINT", "").lower()
                
                endpoint = value
                api_key_var = f"TENANT_{tenant_id.upper()}_KEY"
                log_level_var = f"TENANT_{tenant_id.upper()}_LOG_LEVEL"
                
                if api_key_var in os.environ:
                    configs[tenant_id] = TenantConfig(
                        tenant_id=tenant_id,
                        endpoint=endpoint,
                        api_key=os.environ[api_key_var],
                        log_level=os.environ.get(log_level_var, "INFO"),
                        log_file_prefix=f"azure_guardrails_{tenant_id}"
                    )
        
        if not configs:
            # Fallback to legacy single tenant configuration
            print("No tenant configurations found in environment variables")
        
        return configs





    # INITIALIZE TENANT
    def _initialize_tenant(self, tenant_id: str, config: TenantConfig):
        """
        Initialize Azure clients and logger for a specific tenant.
        
        Args:
            tenant_id: Unique identifier for the tenant
            config: Tenant configuration object
        """
        try:
            # Initialize Azure clients
            credential = AzureKeyCredential(config.api_key)
            content_safety_client = ContentSafetyClient(endpoint=config.endpoint, credential=credential)
            blocklist_client = BlocklistClient(endpoint=config.endpoint, credential=credential)
            
            self.tenant_clients[tenant_id] = {
                'content_safety': content_safety_client,
                'blocklist': blocklist_client,
                'config': config
            }
            
            # Initialize tenant-specific logger
            self.tenant_loggers[tenant_id] = self._setup_tenant_logger(tenant_id, config)
            self.tenant_loggers[tenant_id].info(f"Successfully initialized tenant: {tenant_id}")
            
        except Exception as e:
            print(f"Failed to initialize tenant {tenant_id}: {str(e)}")
            raise



    # SET UP TENANT LOGGER - GENERATE LOGS FOR EACH TENANT
    def _setup_tenant_logger(self, tenant_id: str, config: TenantConfig) -> logging.Logger:
        """
        Set up a dedicated logger for a specific tenant.
        
        Args:
            tenant_id: Unique identifier for the tenant
            config: Tenant configuration object
            
        Returns:
            Configured logger instance
        """
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create tenant-specific logger
        logger_name = f"azure_guardrails.{tenant_id}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Avoid adding multiple handlers to the same logger
        if logger.handlers:
            return logger
        
        # Create file handler for tenant-specific logs
        log_filename = logs_dir / f"{config.log_file_prefix}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, config.log_level.upper()))
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] --->  %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s --->  %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger




    # CHECK IF TENANT IS CONFIGURED
    def _validate_tenant(self, tenant_id: str) -> Tuple[Dict[str, Any], logging.Logger]:
        """
        Validate tenant exists and return client and logger.
        
        Args:
            tenant_id: Tenant identifier to validate
            
        Returns:
            Tuple of (tenant_client_dict, logger)
            
        Raises:
            ValueError: If tenant is not configured
        """
        if tenant_id not in self.tenant_clients:
            raise ValueError(f"Tenant '{tenant_id}' is not configured. Available tenants: {list(self.tenant_clients.keys())}")
        
        return self.tenant_clients[tenant_id], self.tenant_loggers[tenant_id]



    
    # LOOK INTO THE AVAILABLE INITIALIZED TENANTS
    def get_available_tenants(self) -> List[str]:
        """
        Get list of configured tenants.
        
        Returns:
            List of available tenant IDs
        """
        return list(self.tenant_clients.keys())
    
# BLOCK END 





# CREATING AZURE GUARDRAILS FUNCTIONS CLASS WHICH HAS THE CALLABLE FUNCTIONS FOR THE AZURE GUARDRAILS CLIENT
"""
    METHODS:
    - __init__
    - detect_groundedness
    - shield_prompt
    - analyze_text_content_safety
    - analyze_text_with_blocklist
"""
# BLOCK START

class AzureGuardrailFunctions(AzureGuardrailsClient):
    def __init__(self, auto_initialize: bool = False, initialize_tenants: Optional[List[str]] = None):
        if not auto_initialize:
            initialize_tenants = [name.lower() for name in initialize_tenants]
        super().__init__(auto_initialize, initialize_tenants)


    # GROUNDEDNESS CHECKER ON RESPONSE DATA
    def detect_groundedness(self, tenant_id: str, query: str, source_text: str, text: str) -> Dict[str, Any]:
        """
        Detect if the provided text is grounded in the given source material.
        
        Groundedness detection helps identify whether generated content is factually
        consistent with the provided source material.
        
        Args:
            tenant_id: Identifier for the tenant
            query: The query or question being asked
            source_text: Source material to check groundedness against  
            
        Returns:
            Dictionary containing groundedness detection results
            
        Raises:
            ValueError: If tenant is not configured
            HttpResponseError: If Azure API call fails
        """
        tenant_client, logger = self._validate_tenant(tenant_id)
        
        logger.info(f"Starting groundedness detection for query: {query[:100]}...")
        
        try:
            url = f"{tenant_client['config'].endpoint}/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview"
            
            headers = {
                'Ocp-Apim-Subscription-Key': tenant_client['config'].api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                "domain": "Generic",
                "task": "QnA", 
                "qna": {
                    "query": query
                },
                "text": text,
                "groundingSources": [source_text],
            }
            
            logger.debug(f"Sending groundedness detection request: {json.dumps(data, indent=2)}")
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Groundedness detection completed successfully")
            logger.debug(f"Groundedness result: {json.dumps(result, indent=2)}")
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Groundedness detection request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in groundedness detection: {str(e)}")
            raise

    


    # PROMPT SHIELD METHODS - TO AVOID INJECTION ATTACKS AND JAIL BREAKING PROMPTS
    def shield_prompt(self, tenant_id: str, user_prompt: str) -> Dict[str, Any]:
        """
        Analyze user prompts for potential security risks and injection attacks.
        
        Prompt Shield helps identify potentially harmful prompts that might be trying
        to manipulate AI systems through injection attacks or other malicious techniques.
        
        Args:
            tenant_id: Identifier for the tenant
            user_prompt: User prompt to analyze for security risks
            
        Returns:
            Dictionary containing prompt shield analysis results
            
        Raises:
            ValueError: If tenant is not configured
            HttpResponseError: If Azure API call fails
        """
        tenant_client, logger = self._validate_tenant(tenant_id)
        
        logger.info(f"Starting prompt shield analysis for prompt: {user_prompt[:100]}...")
        
        try:
            url = f"{tenant_client['config'].endpoint}/contentsafety/text:shieldPrompt?api-version=2024-02-15-preview"
            
            headers = {
                'Ocp-Apim-Subscription-Key': tenant_client['config'].api_key,
                'Content-Type': 'application/json'
            }
            
            data = {
                "userPrompt": user_prompt
            }
            
            logger.debug(f"Sending prompt shield request for prompt length: {len(user_prompt)} characters")
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Prompt shield analysis completed successfully")
            logger.debug(f"Prompt shield result: {json.dumps(result, indent=2)}")
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Prompt shield request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in prompt shield analysis: {str(e)}")
            raise
    



    # CONTENT SAFETY TEXT ANALYSIS METHODS   
    def analyze_text_content_safety(self, tenant_id: str, text: str) -> Dict[str, Any]:
        """
        Analyze text content for safety violations across multiple categories.
        This method evaluates text for potential issues in these categories: Hate speech, Self-harm content, Sexual content and Violence
        
        Args:
            tenant_id: Identifier for the tenant
            text: Text content to analyze
            
        Returns:
            Dictionary containing analysis results with severity levels for each category
            
        Raises:
            ValueError: If tenant is not configured
            HttpResponseError: If Azure API call fails
        """
        tenant_client, logger = self._validate_tenant(tenant_id)
        
        logger.info(f"Starting content safety analysis for text: {text[:100]}...")
        
        try:
            # Construct analysis request
            request = AnalyzeTextOptions(text=text)
            # Perform analysis using Azure Content Safety client
            response = tenant_client['content_safety'].analyze_text(request)
            
            # Extract results for each category
            results = {}
            categories = [TextCategory.HATE, TextCategory.SELF_HARM, TextCategory.SEXUAL, TextCategory.VIOLENCE]
            
            for category in categories:
                try:
                    category_result = next((item for item in response.categories_analysis if item.category == category), None)
                    if category_result:
                        category_name = category.value.lower()
                        results[category_name] = {
                            'severity': category_result.severity,
                            'category': category_name
                        }
                        logger.debug(f"{category_name.title()} severity: {category_result.severity}")
                except StopIteration:
                    logger.warning(f"No results found for category: {category}")
            
            logger.info(f"Content safety analysis completed. Found {len(results)} category results")
            return {
                'text_length': len(text),
                'analysis_results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except HttpResponseError as e:
            logger.error(f"Content safety analysis failed - Error code: {e.error.code if e.error else 'Unknown'}")
            logger.error(f"Error message: {e.error.message if e.error else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in content safety analysis: {str(e)}")
            raise




    # ANALYZE USER TEXT WITH BLOCKLISTS DEFINED BY THE USER
    def analyze_text_with_blocklist(self, tenant_id: str, text: str, blocklist_names: List[str], halt_on_blocklist_hit: bool = False) -> Dict[str, Any]:
        """
        Analyze text content against custom blocklists in addition to content safety analysis.
        
        This method combines content safety analysis with custom blocklist checking.
        Note: After editing your blocklist, it usually takes effect in 5 minutes.
        
        Args:
            tenant_id: Identifier for the tenant
            text: Text content to analyze
            blocklist_names: List of blocklist names to check against
            halt_on_blocklist_hit: Whether to halt processing on first blocklist match
            
        Returns:
            Dictionary containing analysis results including blocklist matches
            
        Raises:
            ValueError: If tenant is not configured
            HttpResponseError: If Azure API call fails
        """
        tenant_client, logger = self._validate_tenant(tenant_id)
        
        logger.info(f"Starting text analysis with blocklist for text: {text[:100]}...")
        logger.info(f"Using blocklists: {', '.join(blocklist_names)}")
        
        try:
            # Perform analysis with blocklist checking
            analysis_result = tenant_client['content_safety'].analyze_text(
                AnalyzeTextOptions(
                    text=text, 
                    blocklist_names=blocklist_names, 
                    halt_on_blocklist_hit=halt_on_blocklist_hit
                )
            )
            
            results = {
                'text_length': len(text),
                'blocklist_matches': [],
                'content_safety_results': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Check for blocklist matches
            if analysis_result and analysis_result.blocklists_match:
                logger.info(f"Found {len(analysis_result.blocklists_match)} blocklist matches")
                for match_result in analysis_result.blocklists_match:
                    match_info = {
                        'blocklist_name': match_result.blocklist_name,
                        'blocklist_item_id': match_result.blocklist_item_id,
                        'blocklist_item_text': match_result.blocklist_item_text
                    }
                    results['blocklist_matches'].append(match_info)
                    logger.debug(f"Blocklist match: {match_info}")
            else:
                logger.info("No blocklist matches found")
            
            # Extract content safety results if available
            if analysis_result and analysis_result.categories_analysis:
                categories = [TextCategory.HATE, TextCategory.SELF_HARM, TextCategory.SEXUAL, TextCategory.VIOLENCE]
                for category in categories:
                    category_result = next((item for item in analysis_result.categories_analysis if item.category == category), None)
                    if category_result:
                        category_name = category.value.lower()
                        results['content_safety_results'][category_name] = {
                            'severity': category_result.severity,
                            'category': category_name
                        }
            
            logger.info("Text analysis with blocklist completed successfully")
            return results
            
        except HttpResponseError as e:
            logger.error(f"Text analysis with blocklist failed - Error code: {e.error.code if e.error else 'Unknown'}")
            logger.error(f"Error message: {e.error.message if e.error else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in text analysis with blocklist: {str(e)}")
            raise
    

# BLOCK END





# CREATING AZURE BLOCKLIST FUNCTIONS CLASS WHICH HAS THE CALLABLE FUNCTIONS FOR THE AZURE BLOCKLIST CLIENT
"""
    METHODS:
    - __init__
    - create_or_update_blocklist
    - add_blocklist_items
    - get_all_blocklists
    - get_blocklist_with_items
    - remove_blocklist_item_by_text
    - delete_blocklist
"""
# BLOCK START

class AzureBlocklistFunctions(AzureGuardrailFunctions):
    def __init__(self, auto_initialize: bool = False, initialize_tenants: Optional[List[str]] = None):
        if not auto_initialize:
            initialize_tenants = [name.lower() for name in initialize_tenants]
        super().__init__(auto_initialize, initialize_tenants)



    # CREATE OR UPDATE BLOCKLIST AS PER TENANT DEMANDS
    def create_or_update_blocklist(self, tenant_id: str, blocklist_name: str, blocklist_description: str) -> Dict[str, Any]:
        """
        Create a new blocklist or update an existing one.
        
        Args:
            tenant_id: Identifier for the tenant
            blocklist_name: Name of the blocklist to create or update
            blocklist_description: Description of the blocklist purpose
            
        Returns:
            Dictionary containing the created/updated blocklist information
            
        Raises:
            ValueError: If tenant is not configured
            HttpResponseError: If Azure API call fails
        """
        tenant_client, logger = self._validate_tenant(tenant_id)
        
        logger.info(f"Creating or updating blocklist: {blocklist_name}")
        
        try:
            blocklist = tenant_client['blocklist'].create_or_update_text_blocklist(
                blocklist_name=blocklist_name,
                options=TextBlocklist(blocklist_name=blocklist_name, description=blocklist_description)
            )
            
            result = {
                'blocklist_name': blocklist.blocklist_name,
                'description': blocklist.description,
                'timestamp': datetime.now().isoformat(),
                'operation': 'create_or_update'
            }
            
            logger.info(f"Successfully created/updated blocklist: {blocklist.blocklist_name}")
            logger.debug(f"Blocklist details: {result}")
            
            return result
            
        except HttpResponseError as e:
            logger.error(f"Create or update blocklist failed - Error code: {e.error.code if e.error else 'Unknown'}")
            logger.error(f"Error message: {e.error.message if e.error else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating/updating blocklist: {str(e)}")
            raise
    



    # ADD ITEMS TO AN EXISTING BLOCKLIST
    def add_blocklist_items(self, tenant_id: str, blocklist_name: str, items: List[str]) -> Dict[str, Any]:
        """
        Add items to an existing blocklist.
        
        Note: The maximum length of a blocklist item is 128 characters.
        
        Args:
            tenant_id: Identifier for the tenant
            blocklist_name: Name of the blocklist to add items to
            items: List of text items to add to the blocklist
            
        Returns:
            Dictionary containing information about the added items
            
        Raises:
            ValueError: If tenant is not configured or items exceed character limit
            HttpResponseError: If Azure API call fails
        """
        tenant_client, logger = self._validate_tenant(tenant_id)
        
        logger.info(f"Adding {len(items)} items to blocklist: {blocklist_name}")
        
        # Validate item lengths
        invalid_items = [item for item in items if len(item) > 128]
        if invalid_items:
            error_msg = f"Items exceed 128 character limit: {invalid_items[:3]}..."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Convert strings to TextBlocklistItem objects
            blocklist_items = [TextBlocklistItem(text=item) for item in items]
            
            result = tenant_client['blocklist'].add_or_update_blocklist_items(
                blocklist_name=blocklist_name,
                options=AddOrUpdateTextBlocklistItemsOptions(blocklist_items=blocklist_items)
            )
            
            added_items = []
            for blocklist_item in result.blocklist_items:
                item_info = {
                    'blocklist_item_id': blocklist_item.blocklist_item_id,
                    'text': blocklist_item.text,
                    'description': blocklist_item.description
                }
                added_items.append(item_info)
                logger.debug(f"Added item: {item_info}")
            
            response = {
                'blocklist_name': blocklist_name,
                'added_items_count': len(added_items),
                'added_items': added_items,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully added {len(added_items)} items to blocklist")
            return response
            
        except HttpResponseError as e:
            logger.error(f"Add blocklist items failed - Error code: {e.error.code if e.error else 'Unknown'}")
            logger.error(f"Error message: {e.error.message if e.error else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error adding blocklist items: {str(e)}")
            raise



    # GET ALL BLOCKLIST NAMES FOR TENANT - THIS IS NOT SAME AS GETTING THE BLOCKLIST ITEMS
    def get_all_blocklists(self, tenant_id: str) -> Dict[str, Any]:
        """
        Retrieve all existing blocklists for a tenant.
        
        Args:
            tenant_id: Identifier for the tenant
            
        Returns:
            Dictionary containing all blocklists and their information
            
        Raises:
            ValueError: If tenant is not configured
            HttpResponseError: If Azure API call fails
        """
        tenant_client, logger = self._validate_tenant(tenant_id)
        
        logger.info("Retrieving all blocklists")
        
        try:
            blocklists = tenant_client['blocklist'].list_text_blocklists()
            
            blocklist_info = []
            for blocklist in blocklists:
                info = {
                    'name': blocklist.blocklist_name,
                    'description': blocklist.description
                }
                blocklist_info.append(info)
                logger.debug(f"Found blocklist: {info}")
            
            result = {
                'blocklists_count': len(blocklist_info),
                'blocklists': blocklist_info,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully retrieved {len(blocklist_info)} blocklists")
            return result
            
        except HttpResponseError as e:
            logger.error(f"List blocklists failed - Error code: {e.error.code if e.error else 'Unknown'}")
            logger.error(f"Error message: {e.error.message if e.error else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing blocklists: {str(e)}")
            raise
    


    # GET ALL THE BLOCKLIST ITEMS FOR A PARTICULAR BLOCKLIST NAME
    def get_blocklist_with_items(self, tenant_id: str, blocklist_name: str) -> Dict[str, Any]:
        """
        Retrieve blocklist details along with all its items.
        
        Args:
            tenant_id: Identifier for the tenant
            blocklist_name: Name of the blocklist to retrieve
            
        Returns:
            Dictionary containing blocklist details and its items
            
        Raises:
            ValueError: If tenant is not configured
            HttpResponseError: If Azure API call fails
        """
        tenant_client, logger = self._validate_tenant(tenant_id)
        
        logger.info(f"Retrieving blocklist and items for: {blocklist_name}")
        
        try:
            # Get blocklist details
            blocklist = tenant_client['blocklist'].get_text_blocklist(blocklist_name=blocklist_name)
            blocklist_info = {
                'name': blocklist.blocklist_name,
                'description': blocklist.description
            }
            logger.info(f"Successfully retrieved blocklist: {blocklist_name}")
            logger.debug(f"Blocklist details: {blocklist_info}")
            
            # Get blocklist items
            blocklist_items = tenant_client['blocklist'].list_text_blocklist_items(blocklist_name=blocklist_name)
            items_info = []
            for blocklist_item in blocklist_items:
                item_info = {
                    'blocklist_item_id': blocklist_item.blocklist_item_id,
                    'text': blocklist_item.text,
                    'description': blocklist_item.description
                }
                items_info.append(item_info)
                logger.debug(f"Found item: {item_info}")
            
            logger.info(f"Successfully retrieved {len(items_info)} items from blocklist")
            
            # Final result combining both
            result = {
                'blocklist': blocklist_info,
                'items_count': len(items_info),
                'items': items_info,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except HttpResponseError as e:
            logger.error(f"Get blocklist/items failed - Error code: {e.error.code if e.error else 'Unknown'}")
            logger.error(f"Error message: {e.error.message if e.error else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error retrieving blocklist and items: {str(e)}")
            raise

 


    


    # REMOVE A SPECIFIC ITEM FROM A BLOCKLIST
    def remove_blocklist_item_by_text(self, tenant_id: str, blocklist_name: str, blocklist_item_text: str) -> Dict[str, Any]:
        """
        Remove a specific item from a blocklist by its text (instead of ID).
        
        Args:
            tenant_id: Identifier for the tenant
            blocklist_name: Name of the blocklist to remove item from
            blocklist_item_text: Text of the item to remove
            
        Returns:
            Dictionary containing removal confirmation
            
        Raises:
            ValueError: If tenant is not configured or item not found
            HttpResponseError: If Azure API call fails
        """
        tenant_client, logger = self._validate_tenant(tenant_id)
        
        logger.info(f"Attempting to remove item with text '{blocklist_item_text}' from blocklist: {blocklist_name}")
        
        try:
            # Step 1: Fetch all items in the blocklist
            blocklist_items = tenant_client['blocklist'].list_text_blocklist_items(blocklist_name=blocklist_name)
            
            # Step 2: Find the matching item by text
            matching_item = None
            for item in blocklist_items:
                if item.text.strip().lower() == blocklist_item_text.strip().lower():
                    matching_item = item
                    break
            
            if not matching_item:
                logger.error(f"No blocklist item found with text: {blocklist_item_text}")
                raise ValueError(f"Blocklist item with text '{blocklist_item_text}' not found in '{blocklist_name}'")
            
            # Step 3: Remove item by its ID
            tenant_client['blocklist'].remove_blocklist_items(
                blocklist_name=blocklist_name,
                options=RemoveTextBlocklistItemsOptions(blocklist_item_ids=[matching_item.blocklist_item_id])
            )
            
            result = {
                'blocklist_name': blocklist_name,
                'removed_item_text': blocklist_item_text,
                'removed_item_id': matching_item.blocklist_item_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            logger.info(f"Successfully removed item '{blocklist_item_text}' (ID: {matching_item.blocklist_item_id}) from blocklist")
            return result
            
        except HttpResponseError as e:
            logger.error(f"Remove blocklist item failed - Error code: {e.error.code if e.error else 'Unknown'}")
            logger.error(f"Error message: {e.error.message if e.error else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error removing blocklist item: {str(e)}")
            raise

           
    




    # DELETE AN ENTIRE BLOCKLIST AND ALL ITS ITEMS
    def delete_blocklist(self, tenant_id: str, blocklist_name: str) -> Dict[str, Any]:
        """
        Delete an entire blocklist and all its items.
        
        Args:
            tenant_id: Identifier for the tenant
            blocklist_name: Name of the blocklist to delete
            
        Returns:
            Dictionary containing deletion confirmation
            
        Raises:
            ValueError: If tenant is not configured
            HttpResponseError: If Azure API call fails
        """
        tenant_client, logger = self._validate_tenant(tenant_id)
        
        logger.info(f"Deleting blocklist: {blocklist_name}")
        
        try:
            tenant_client['blocklist'].delete_text_blocklist(blocklist_name=blocklist_name)
            
            result = {
                'deleted_blocklist': blocklist_name,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
            logger.info(f"Successfully deleted blocklist: {blocklist_name}")
            return result
            
        except HttpResponseError as e:
            logger.error(f"Delete blocklist failed - Error code: {e.error.code if e.error else 'Unknown'}")
            logger.error(f"Error message: {e.error.message if e.error else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting blocklist: {str(e)}")
            raise




# BLOCK END
