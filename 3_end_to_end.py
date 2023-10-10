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

# load config values
with open(r'config.json') as config_file:
    config_details = json.load(config_file)

# configure env variables for acs
service_endpoint = config_details["SEARCH_SERVICE_ENDPOINT"]
index_name = config_details["SEARCH_INDEX_NAME"]
key = config_details["SEARCH_ADMIN_KEY"]
credential = AzureKeyCredential(key)

# create the acs client to issue queries
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

# create the index client
index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]

# function to generate embeddings for title and content fields, and to query embeddings
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text, engine="text-embedding-ada-002")
    embeddings = response['data'][0]['embedding']
    return embeddings

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
                    "description": "The odata filter to apply for the ingredients field. Only actual ingredient names should be used in this filter. If you're not sure something is an ingredient, don't include this filter. Example: ingredients/any(i: i eq 'salt' or i eq 'pepper')",
                },
                "time_filter": {
                    "type": "string",
                    "description": "The odata filter to apply for the total_time field. If a user asks for a quick or easy recipe, you should filter down to recipes that will take less than 30 minutes. Example: total_time lt 25",
                }
            },
            "required": ["query"],
        },
    }
]

# define function to call acs

def query_recipes(query, ingredients_filter=None, time_filter=None):
    filter = ""
    if ingredients_filter and time_filter:
        filter = f"{time_filter} and {ingredients_filter}"
    elif ingredients_filter:
        filter = ingredients_filter
    elif time_filter:
        filter = time_filter

    results = search_client.search(
        query_type="semantic",
        query_language="en-us",
        semantic_configuration_name="my-semantic-config",
        search_text=query,
        vectors=[Vector(value=generate_embeddings(query), k=3, fields="recipe_vector")],
        filter=filter,
        select=["recipe_id", "recipe", "recipe_category", "recipe_name", "description"],
    )

    n = 1
    recipes_for_prompt = ""
    for result in results:
        recipes_for_prompt += f"Recipe {result['recipe_id']}: {result['recipe_name']}: {result['description']}\n "
        n += 1

    return recipes_for_prompt

# end to end flow

def run_conversation(messages, functions, available_functions, deployment_id):

    # send the conversation and available functions to GPT
    response = openai.ChatCompletion.create(
        deployment_id=deployment_id,
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature=0.2
    )
    response_message = response["choices"][0]["message"]

    # check if the model wants to call a function
    if response_message.get("function_call"):
        print("Recommended function call:")
        print(response_message.get("function_call"))
        print()

        # call the function and handle potential errors
        function_name = response_message["function_call"]["name"]

        # verify function exists
        if function_name not in available_functions:
            return "Function" + function_name + "does not exist"
        function_to_call = available_functions[function_name]

        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(**function_args)

        print("Output of function call:")
        print(function_response)
        print()

        # send info on function call and function response to the model
        messages.append(
            {
                "role": response_message["role"],
                "function_call": {
                    "name": response_message["function_call"]["name"],
                    "arguments": response_message["function_call"]["arguments"],
                },
                "content": None
            }
        )

        # adding function response to messages
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )   # extend conversation with function response

        print("Messages in second request:")
        for message in messages:
            print(message)
        print()

        second_response = openai.ChatCompletion.create(
            messages = messages,
            deployment_id=deployment_id
        )

        return second_response
    else:
        return response
    
system_message = """Assistant is a large language model designed to help users find and create recipes.

You have access to an Azure Cognitive Search index with hundreds of recipes. You can search for recipes by name, ingredient, or cuisine.

You are designed to be an interactive assistant, so you can ask users clarifying questions to help them find the right recipe. It's better to give more detailed queries to the search index rather than vague one.
"""

messages = [{"role": "system", "content": system_message},
            {"role": "user", "content": "I want to make a pasta dish that takes less than 60 minutes to make."}]

deployment_name='gpt-35-turbo'

available_functions = {'query_recipes': query_recipes}

result = run_conversation(messages, functions, available_functions, deployment_name)

print("Final response:")
print(result['choices'][0]['message']['content'])