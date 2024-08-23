"""
This script is designed to extract and process ontology
terms and their relationships from text. It uses OpenAI's 
GPT-3.5 Turbo to generate a network graph of key terms and 
their relations, based on a predefined prompt. 
The script includes functionality for retrieving and processing 
term pairs and their relationships from given input text.
"""

import sys
sys.path.append("..")
import json
from openai import OpenAI

SYS_PROMPT = (
    "You are a network graph maker who extracts terms and their relations from a given context. "
    "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
    "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
    "# Instructions\n"
    "Thought 1: While traversing through text, Think about the key terms mentioned in it.\n"
        "\tTerms may include object, entity, location, organization, person, \n"
        "\tcondition, acronym, documents, service, concept, etc.\n"
        "\tTerms should be as atomistic as possible.\n\n"
    "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
        "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
        "\tTerms can be related to many other terms\n\n"
    "Thought 3: Find out the relation between each such related pair of terms. \n\n"
    "Format your output as a list of json. Each element of the list contains a pair of terms"
    "and the relation between them, like the following: \n"
    "[\n"
    "   {\n"
    '       "node_1": "A concept from extracted ontology of maximum four words composed by ONLY Entities or Nouns",\n'
    '       "node_2": "A related concept from extracted ontology of maximum four words composed by ONLY Entities or Nouns",\n'
    '       "edge": "A relationship between node_1 and node_2 composed by one or two verbs"\n'
    "   }, {...}\n"
    "]\n\n"
)

    
def process_target_node(node2:str=None, edge:str=None):
    """Process the target node to find common text in node2 and edge.

    Args:
        node2 (str): The target node.
        edge (str): The edge string.

    Returns:
        str: The processed node2 with common text removed.
    """    
    text, common = "", ""
    for part in node2.split(" "):
        if len(text) == 0:
            text = part
        else:
            text = text + " " + part
            
        if text in edge:
            common = text

    return node2.replace(common,"")

def retrieve_triplets(input: str=None, process:bool=True):
    """Retrieve triplets from a given input.

    Args:
        input (str): The input string.
        process (bool): Whether to process the retrieved triplets.

    Returns:
        list: A list of retrieved triplets.
    """ 
    response = None
    
    try:
        messages = [
            { "role": "system", "content": SYS_PROMPT },
            { "role": "user", "content": f"# Text: ```{input}``` \n\n Reply:" },
        ]

        client = OpenAI()
        response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=messages,
                                max_tokens=4000,
                                temperature=0,
                            )

        # Extract content from the response
        response = response.choices[0].message.content
        result = json.loads(response)
    except Exception as e:
        print("ERROR ### Here is the buggy response: ", response)
        print(e)
        result = None
        
    if process and result is not None:
        result = [{'node_1':item['node_1'], 'node_2':process_target_node(item['node_2'], item['edge']), 'edge':item['edge']} for item in result]
        
    return result