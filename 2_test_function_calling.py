import os
import json 
import openai 
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import Vector  
from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    PrioritizedFields,  
    SemanticField,  
    SearchField,  
    SemanticSettings,  
    VectorSearch,  
    HnswVectorSearchAlgorithmConfiguration,
)

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]

deployment_id='gpt-35-turbo'

messages = [{"role": "user", "content": "Help me find a good lasagna recipe."}]

functions = [
    {
        "name": "query_recipes",
        "description": "Retrieve recipes from the Azure Cognitive Search index",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query string to search for recipes",
                },
                "ingredients_filter": {
                    "type": "string",
                    "description": "The odata filter to apply for the ingredients field. Only actual ingredient names should be used in this filter. If you're not sure something is an ingredient, don't include this filter. Example: ingredients/any(i: i eq 'salt' or o eq 'pepper')",
                },
                "time_filter": {
                    "type": "string",
                    "description": "The odata filter to apply for the total_time field. If a user asks for a quick or easy recipe, you should filter down to recipes that will take less than 30 minutes.Example: total_time lt 25",
                }
            },
            "required": ["query"],
        },
    }
]

response = openai.ChatCompletion.create(
    deployment_id='gpt-35-turbo',
    messages = messages,
    functions=functions,
    temperature=0.2,
    function_call="auto",
)

print(response['choices'][0]['message'])
