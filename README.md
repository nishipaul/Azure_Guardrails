## About Azure Guardrails

The existing files of "azure_guardrails.py" checks for content safety and gives the flexibility to add list of words that needs to be blocked. This DOES NOT cover PII, which will be uploaded separately (done using Presidio by Microsoft).

#### How to run the file
1. Create .env file where you should have two values mandatorily -
     a. TENANT_ABC_KEY= {API KEY GENERATED FOR THE TENANT}
     b. TENANT_ABC_ENDPOINT="https://tenant-abc.cognitiveservices.azure.com"
   Here ABC is the name of the tenant. If you are new then please read this to know how to generate - https://simpplr.atlassian.net/wiki/spaces/DS/pages/3954311181/Azure+Guardrails+-+Content+Safety+and+Azure+AI+Foundry
2. Pip install all the dependencies in the requirements.txt file.
3. Open the file of azure_guardrail_enduser.ipynb which is calling the functions from azure_guardrails.py and run the cells to view the output
4. For step by step example, follow the file - azure_guardrail_usage_example.ipynb
