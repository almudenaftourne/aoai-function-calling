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

deployment_name = config_details['DEPLOYMENT_NAME'] # You need to use the 0613 version of gpt-35-turbo or gpt-4

# Create a search index
fields = [
    SimpleField(name="recipe_id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
    SearchableField(name="recipe_category", type=SearchFieldDataType.String, filterable=True, analyzer_name="en.microsoft"),    
    SearchableField(name="recipe_name", type=SearchFieldDataType.String, facetable=True, analyzer_name="en.microsoft"),
    SearchableField(name="ingredients", collection=True, type=SearchFieldDataType.String, facetable=True, filterable=True),
    SearchableField(name="recipe", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
    SearchableField(name="description", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
    SimpleField(name="total_time", type=SearchFieldDataType.Int32, filterable=True, facetable=True),
    SearchField(name="recipe_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_configuration="my-vector-config")
]

vector_search = VectorSearch(
    algorithm_configurations=[
        HnswVectorSearchAlgorithmConfiguration(
            name="my-vector-config",
            kind="hnsw"
        )
    ]
)

# Semantic Configuration to leverage Bing family of ML models for re-ranking (L2)
semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=PrioritizedFields(
        title_field=None,
        prioritized_keywords_fields=[],
        prioritized_content_fields=[SemanticField(field_name="recipe")]
    ))
semantic_settings = SemanticSettings(configurations=[semantic_config])

# Create the search index with the semantic settings
index = SearchIndex(name=index_name, fields=fields, 
                    vector_search=vector_search, semantic_settings=semantic_settings)
result = index_client.delete_index(index)
print(f' {index_name} deleted')
result = index_client.create_index(index)
print(f' {result.name} created')

# function to generate embeddings for title and content fields, and to query embeddings
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text, engine="text-embedding-ada-002")
    embeddings = response['data'][0]['embedding']
    return embeddings

batch_size = 100
counter = 0
documents = []
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)

with open("recipes_final.jsonl", "r") as j_in:
    for line in j_in:
        counter += 1 
        json_recipe = json.loads(line)
        json_recipe['total_time'] = int(json_recipe['total_time'].split(' ')[0])
        json_recipe['recipe_vector'] = generate_embeddings(json_recipe['recipe'])
        json_recipe["@search.action"] = "upload"
        documents.append(json_recipe)
        if counter % batch_size == 0:
            # Load content into index
            result = search_client.upload_documents(documents)  
            print(f"Uploaded {len(documents)} documents") 
            documents = []
            
            
if documents != []:
    # Load content into index
    result = search_client.upload_documents(documents)  
    print(f"Uploaded {len(documents)} documents") 
