import openai
import json
import os
from dotenv import load_dotenv
import pytz
from datetime import datetime
import pandas as pd

# load env variables
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_version = os.environ["OPENAI_API_VERSION"]

deployment_id = 'gpt-35-turbo'

# function 1 get current time

def get_current_time(location):
    try:
        # get the timezone for the city
        timezone = pytz.timezone(location)

        # get current time in timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")

        return current_time
    except:
        return "Sorry, I couldn't find the timezone for that location"
    
# function 2 get stock market data

def get_stock_market_data(index):

    available_indices = ["S&P 500", "NASDAQ Composite", "Dow Jones Industrial Average", "Financial Times Stock Exchange 100 Index"]

    if index not in available_indices:
        return "Invalid index. Please choose from 'S&P 500', 'NASDAQ Composite', 'Dow Jones Industrial Average', 'Financial Times Stock Exchange 100 Index'."

    # Read the CSV file
    data = pd.read_csv('stock_data.csv')

    # Filter data for the given index
    data_filtered = data[data['Index'] == index]

    # Remove 'Index' column
    data_filtered = data_filtered.drop(columns=['Index'])

    # Convert the DataFrame into a dictionary
    hist_dict = data_filtered.to_dict()

    for key, value_dict in hist_dict.items():
        hist_dict[key] = {k: v for k, v in value_dict.items()}

    return json.dumps(hist_dict)

# function 3 calculator

import math

def calculator(num1, num2, operator):
    if operator == '+':
        return str(num1 + num2)
    elif operator == '-':
        return str(num1 - num2)
    elif operator == '*':
        return str(num1 * num2)
    elif operator == '/':
        return str(num1 / num2)
    elif operator == '**':
        return str(num1 ** num2)
    elif operator == 'sqrt':
        return str(math.sqrt(num1))
    else:
        return "Invalid operator"
    
# 1. Call the model with the user query and a set of functions defined in the functions parameter
# 2. The model can choose to call a function; if so, the content will be a stringfied JSON object adhering to your custom schema 
# 3. Parse the string into JSON in your code, and call your function with the provided arguments if they exist
# 4. Call the model again by appending the function response as a new message, and let the model summarize the results back to the user

functions = [
        {
            "name": "get_current_time",
            "description": "Get the current time in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location name. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London",
                    }
                },
                "required": ["location"],
            },
        },
        {
            "name": "get_stock_market_data",
            "description": "Get the stock market data for a given index",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "string",
                        "enum": ["S&P 500", "NASDAQ Composite", "Dow Jones Industrial Average", "Financial Times Stock Exchange 100 Index"]},
                },
                "required": ["index"],
            },    
        },
             {
            "name": "calculator",
            "description": "A simple calculator used to perform basic arithmetic operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {"type": "number"},
                    "num2": {"type": "number"},
                    "operator": {"type": "string", "enum": ["+", "-", "*", "/", "**", "sqrt"]},
                },
                "required": ["num1", "num2", "operator"],
            },
        }
    ]

available_functions = {
            "get_current_time": get_current_time,
            "get_stock_market_data": get_stock_market_data,
            "calculator": calculator,
        } 

# define a helper function to validate the function call

import inspect

# helper method used to check if the correct arguments are provided to a function
def check_args(function, args):
    sig = inspect.signature(function)
    params = sig.parameters

    # Check if there are extra arguments
    for name in args:
        if name not in params:
            return False
    # Check if the required arguments are provided 
    for name, param in params.items():
        if param.default is param.empty and name not in args:
            return False

    return True

def run_conversation(messages, functions, available_functions, deployment_id):

    response = openai.ChatCompletion.create(
        deployment_id=deployment_id,
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        print("Recommended Function call:")
        print(response_message.get("function_call"))
        print()

        function_name = response_message["function_call"]["name"]

        # verify function exists
        if function_name not in available_functions:
            return "Function" + function_name + "does not exist"
        function_to_call = available_functions[function_name]

        # verify function has correct number of arguments
        function_args = json.loads(response_message["function_call"]["arguments"])
        if check_args(function_to_call, function_args) is False:
            return "Invalid number of arguments for function: " + function_name
        function_response = function_to_call(**function_args)
        
        print("Output of function call:")
        print(function_response)
        print()

        # send info on function call and function response to GPT
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
        )  # extend conversation with function response

        print("Messages in second request:")
        for message in messages:
            print(message)
        print()

        second_response = openai.ChatCompletion.create(
            messages=messages,
            deployment_id=deployment_id
        )  # get a new response from GPT where it can see the function response

        return second_response
    
# messages = [{"role": "user", "content": "What time is it in New York?"}]
# assistant_response = run_conversation(messages, functions, available_functions, deployment_id)
# print(assistant_response['choices'][0]['message'])

def run_multiturn_conversation(messages, functions, available_functions, deployment_name):
    # Step 1: send the conversation and available functions to GPT

    response = openai.ChatCompletion.create(
        deployment_id=deployment_name,
        messages=messages,
        functions=functions,
        function_call="auto", 
        temperature=0
    )

    # Step 2: check if GPT wanted to call a function
    while response["choices"][0]["finish_reason"] == 'function_call':
        response_message = response["choices"][0]["message"]
        print("Recommended Function call:")
        print(response_message.get("function_call"))
        print()
        
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        
        function_name = response_message["function_call"]["name"]
        
        # verify function exists
        if function_name not in available_functions:
            return "Function " + function_name + " does not exist"
        function_to_call = available_functions[function_name]  
        
        # verify function has correct number of arguments
        function_args = json.loads(response_message["function_call"]["arguments"])
        if check_args(function_to_call, function_args) is False:
            return "Invalid number of arguments for function: " + function_name
        function_response = function_to_call(**function_args)
        
        print("Output of function call:")
        print(function_response)
        print()
        
        # Step 4: send the info on the function call and function response to GPT
        
        # adding assistant response to messages
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
        )  # extend conversation with function response

        print("Messages in next request:")
        for message in messages:
            print(message)
        print()

        response = openai.ChatCompletion.create(
            messages=messages,
            deployment_id=deployment_name,
            function_call="auto",
            functions=functions,
            temperature=0
        )  # get a new response from GPT where it can see the function response

    return response

# Can add system prompting to guide the model to call functions and perform in specific ways
next_messages = [{"role": "system", "content": "Assistant is a helpful assistant that helps users get answers to questions. Assistant has access to several tools and sometimes you may need to call multiple tools in sequence to get answers for your users."}]
next_messages.append({"role": "user", "content": "How much did S&P 500 change between July 12 and July 13? Use the calculator."})

assistant_response = run_multiturn_conversation(next_messages, functions, available_functions, deployment_id)
print("Final Response:")
print(assistant_response["choices"][0]["message"])
print("Conversation complete!") 