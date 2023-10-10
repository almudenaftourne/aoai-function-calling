import openai
import json
import os
from dotenv import load_dotenv

# load env variables
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]

deployment_name = 'gpt-35-turbo'

# call the model with the user query and the set of defined functions in the functions parameter
# the model can choose if it calls a function. If so, the content will be a stringfied JSON object
# the function call that should be made and arguments are location in: response[choices][0][function_call]

def get_function_call(messages, function_call = "auto"):
    # define the functions to use
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    ]

    # call the model with the user query (messages) and the functions defined in the functions parameter
    response = openai.ChatCompletion.create(
        deployment_id = deployment_name,
        messages = messages,
        functions = functions,
        function_call = function_call
    )

    return response

first_message = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

# 'auto' : Let the model decide what function to call
print("Let the model decide what function to call:")
print(get_function_call(first_message, "auto")["choices"][0]['message'])

# 'none' : Don't call any function 
print("Don't call any function:")
print(get_function_call(first_message, "none")["choices"][0]['message'])

# force a specific function call
print("Force a specific function call:")
print(get_function_call(first_message, function_call={"name": "get_current_weather"})["choices"][0]['message'])


