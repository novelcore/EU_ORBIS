import re
import json
from tqdm import tqdm
from .wiki_data import *
from .kg_prompts import retrieve_triplets
from .neo4j_connection import Neo4jConnection

def text_preprocess(text:str=None)->str:
    '''
        Text preprocessing

        Parameters
        ----------
        text: input text
        stripping_accents_and_lowercasing: True/False if the user requires stripping accents & lowercasing
        Returns
        -------
        preprocessed text
    '''
    # Remove "\n"
    text = text.replace('.\n', '. ')
    text = text.replace("-\n", '')
    text = text.replace('\n', ' ')
    # Remove quotes
    text = text.replace('"','').replace("'",'')
    # Replace tabs
    text = text.replace('\t', ' ')
    # Remove multiple spaces using a regular expression
    text = re.sub(' +', ' ', text) 
    text = re.sub(r'\.+', ".", text)
    # Remove errors
    text = re.sub('\xa0', ' ', text)
    text = text.replace("\x07","")
        
    return text


def text2KG(data: Dict[str, Any]=None, neo4j_settings: Dict[str, str]=None, clean_knowledge_graph: bool=True):
    """
    Convert text to KG and perform Semantical Enrichement with Wiki data in NER Entities.

    Args:
        data (Dict[str, Any]): The JSON object containing bcause data.
        neo4j_settings (Dict[str, str]): Dictionary containing Neo4j connection settings.
        clean_knowledge_graph (bool): Flag indicating whether to clean the knowledge graph in Neo4j.
        model_name (str): The name of the model to use. Default is 'kunoichi'.
        refined_model: The pre-initialized Refined model.

    Returns:
        None
    """
    # Set default Neo4j settings if not provided
    neo4j_settings = neo4j_settings or {
                                        "connection_url": "neo4j+s://c1f16148.databases.neo4j.io",
                                        "username": "neo4j",
                                        "password": "9G8El-Vuef7aSWjLGPrm4-LdYPIQHCyVomA5Mp3bNR8",
                                    }
    
    # Create Graph-Database in Neo4j
    graph = Neo4jConnection(
        uri=neo4j_settings["connection_url"],
        user=neo4j_settings["username"],
        pwd=neo4j_settings["password"],
    )

    # Optionally clean Neo4j KG if specified
    if clean_knowledge_graph:
        graph.clean_base()

    # Create Suject node
    query = f"""MERGE (s:SUBJECT {{id:"{data['id']}", author_id:"{data['author_id']}", title:"{data['title']}", tagline:"{data['tagline']}"}})"""
    graph.query(query)

    # Create Positions Posts
    for item in data["posts"]:

        if item['discussion_type'] == "position":
            text = text_preprocess(item["text"].replace('"','\''))
            author_id = item["author_id"]
            id = item["id"]
            
            # Sanity check: Skip processing if text is too short (less than 5 characters)
            # Reason: It is impractical to generate meaningful triplets from very short text
            # Triplet generation requires a minimum amount of text to identify entities and their relationships
            if len(text) < 5: continue
            
            query = f"""MATCH (s:SUBJECT {{id:"{data['id']}"}})
            MERGE (p:POSITION {{id:"{id}", author_id:"{author_id}", text:"{text}"}})
            MERGE (s)-[:HAS_POSITION]-(p)       
            """
            graph.query(query)

    # Create IN FAVOR / AGAINST nodes and add Wiki nodes if any entities are identified in previous nodes.
    for item in tqdm(data["posts"]):
        text = text_preprocess(item["text"])
        author_id = item["author_id"]
        id = item["id"]

        # Sanity checks
        if len(text) < 5: continue
                
        # INFAVOR / AGAINST Nodes
        if item['discussion_type'] not in ["argument_against", "argument_for"]: continue
                                        
        if item['discussion_type'] == "argument_against":   
            node_type = "AGAINST"
        else:
            node_type = "INFAVOR"
            
        query = f"""MATCH (p:POSITION {{id:"{item['linked_parent_item']}"}})
        MERGE (a:{node_type} {{id:"{id}", author_id:"{author_id}", text:"{text}"}})
        MERGE (p)-[:{"HAS_" + node_type}]-(a)   
        """

        graph.query(query)
        
        # CLUSTER Nodes
        # Do not create Cluster nodes in case they are outlier
        if not item['cluster_id'].endswith("-1"):
            query = f"""MATCH (a:{node_type} {{id:"{id}"}})
            MERGE (c:CLUSTER {{id:"{item['cluster_id']}", title:"{item['cluster_title']}", summary:"{item['summary'].replace('"','')}"}})
            MERGE (c)-[:HAS_POST]->(a)
            """

            # Execute the second query
            graph.query(query)
            
            # KEYPHRASE Nodes
            for keyphrase in item['keyphrase']:
                query = f"""MATCH (c:CLUSTER {{id:"{item['cluster_id']}"}})
                MERGE (k:KEYPHRASE {{keyphrase:"{keyphrase}"}})
                MERGE (c)-[:HAS_KEYPHRASE]->(k)
                """
                
                # Execute the third query
                graph.query(query)
        
            # Get wiki info
            wiki_inf = wiki_info(text)
            # print(wiki_inf)/
            
            # Get triples
            triplets = retrieve_triplets(
                                            input=text, 
                                        )

            if triplets is None:
                continue

            # Import triples in KG
            for triplet in triplets:

                if(triplet['node_1'] == '' or triplet['node_2'] == '' or triplet['edge'] == ''):
                    continue
                
                node_1_wiki = []
                node_2_wiki = []

                relation = triplet['edge'].replace(" ", "_").replace(";", "").replace("-","_").replace(",","").replace("'","").upper()
                triplet['node_1'] = triplet['node_1'][1:] if triplet['node_1'][0] == " " else triplet['node_1']
                triplet['node_2'] = triplet['node_2'][1:] if triplet['node_2'][0] == " " else triplet['node_2']

                # Iterate all Entities that was found
                for entity in wiki_inf[text][0]['Entities']:

                    if (entity['Text'] is None):
                        continue
                    
                    # Check if there in any entity in the 1st node
                    if (entity["Original_Text"].lower() in triplet["node_1"].lower()):

                        if(entity["Original_Text"] != entity["Text"]):
                            triplet["node_1"] = triplet["node_1"].replace(entity["Original_Text"], entity["Text"])
                            
                        # Address entity disambiguation by creating distinct nodes for the same text appearing in different contexts.
                        if(entity["Original_Text"] != entity["Text"]):
                            if(entity["Label"]):
                                triplet["node_1"] = triplet["node_1"].replace(entity["Original_Text"], f'{entity["Original_Text"]} ({entity["Label"]})')
                            else:
                                triplet["node_1"] = triplet["node_1"].replace(entity["Original_Text"], entity["Text"])

                        node_1_info = {
                            "Original_Text": entity['Original_Text'],
                            "Text": entity['Text'],
                            "Label": entity['Label'],
                            "WIKI_ID": entity['WIKI_ID'],
                            "WIKI_URL": entity['WIKI_URL'],
                            "Description": entity['Description']
                        }

                        node_1_wiki.append(node_1_info)

                    # Check if there in any entity in the 2nd node
                    if (entity["Original_Text"].lower() in triplet["node_2"].lower()):

                        # Address entity disambiguation by creating distinct nodes for the same text appearing in different contexts.
                        if(entity["Original_Text"] != entity["Text"]):
                            if(entity["Label"]):
                                triplet["node_2"] = triplet["node_2"].replace(entity["Original_Text"], f'{entity["Original_Text"]} ({entity["Label"]})')
                            else:
                                triplet["node_2"] = triplet["node_2"].replace(entity["Original_Text"], entity["Text"])

                        node_2_info = {
                            "Original_Text": entity['Original_Text'],
                            "Text": entity['Text'],
                            "Label": entity['Label'],
                            "WIKI_ID": entity['WIKI_ID'],
                            "WIKI_URL": entity['WIKI_URL'],
                            "Description": entity['Description']
                        }

                        node_2_wiki.append(node_2_info)

                # Text2KG
                query = f"""MATCH (a:{node_type} {{id:"{item["id"]}"}})
                    MERGE (e1:ENTITY {{value: "{triplet['node_1']}"}})
                    MERGE (e2:ENTITY {{value: "{triplet['node_2']}"}})
                    MERGE (a)-[:MENTION]-(e1)
                    MERGE (a)-[:MENTION]-(e2)
                    MERGE (e1)-[:{relation}]-(e2)
                """
                
                graph.query(query)
                
                # Add WIKI Nodes
                for element in node_1_wiki:
                    query = f"""
                    MATCH (e1:ENTITY {{value: "{triplet['node_1']}"}})
                    MERGE (w:WIKI {{label: "{element['Text']}", tag: "{element['Label']}", wiki_id: "{element['WIKI_ID']}", wiki_url: "{element['WIKI_URL']}", description: "{element['Description']}"}})
                    MERGE (e1)-[:HAS_WIKI_DATA]->(w)
                    """
                    graph.query(query)

                for element in node_2_wiki:
                    query = f"""
                    MATCH (e2:ENTITY {{value: "{triplet['node_2']}"}})
                    MERGE (w:WIKI {{label: "{element['Text']}", tag: "{element['Label']}", wiki_id: "{element['WIKI_ID']}", wiki_url: "{element['WIKI_URL']}", description: "{element['Description']}"}})
                    MERGE (e2)-[:HAS_WIKI_DATA]->(w)
                    """
                    graph.query(query)


# def text2KG(data: Dict[str, Any]=None, neo4j_settings: Dict[str, str]=None, clean_knowledge_graph: bool=True):
#     """
#     Convert text to KG and perform Semantical Enrichement with Wiki data in NER Entities.

#     Args:
#         data (Dict[str, Any]): The JSON object containing bcause data.
#         neo4j_settings (Dict[str, str]): Dictionary containing Neo4j connection settings.
#         clean_knowledge_graph (bool): Flag indicating whether to clean the knowledge graph in Neo4j.
#         model_name (str): The name of the model to use. Default is 'kunoichi'.
#         refined_model: The pre-initialized Refined model.

#     Returns:
#         None
#     """
#     # Set default Neo4j settings if not provided
#     neo4j_settings = neo4j_settings or {
#                                         "connection_url": "neo4j+s://c1f16148.databases.neo4j.io",
#                                         "username": "neo4j",
#                                         "password": "9G8El-Vuef7aSWjLGPrm4-LdYPIQHCyVomA5Mp3bNR8",
#                                     }
    
#     # Create Graph-Database in Neo4j
#     graph = Neo4jConnection(
#         uri=neo4j_settings["connection_url"],
#         user=neo4j_settings["username"],
#         pwd=neo4j_settings["password"],
#     )

#     # Optionally clean Neo4j KG if specified
#     if clean_knowledge_graph:
#         graph.clean_base()

#     # Create Suject node
#     query = f"""MERGE (s:SUBJECT {{id:"{data['id']}", author_id:"{data['author_id']}", title:"{data['title']}", tagline:"{data['tagline']}"}})"""
#     graph.query(query)

#     # Create Positions Posts
#     for item in data["posts"]:

#         if item['discussion_type'] == "position":
#             text = text_preprocess(item["text"])
#             author_id = item["author_id"]
#             id = item["id"]
            
#             # Sanity check: Skip processing if text is too short (less than 5 characters)
#             # Reason: It is impractical to generate meaningful triplets from very short text
#             # Triplet generation requires a minimum amount of text to identify entities and their relationships
#             if len(text) < 5: continue
            
#             query = f"""MATCH (s:SUBJECT {{id:"{data['id']}"}})
#             MERGE (p:POSITION {{id:"{id}", author_id:"{author_id}", text:"{text}"}})
#             MERGE (s)-[:HAS_POSITION]-(p)       
#             """
#             graph.query(query)

#     # Create IN FAVOR / AGAINST nodes and add Wiki nodes if any entities are identified in previous nodes.
#     for item in tqdm(data["posts"]):
#         text = text_preprocess(item["text"])
#         author_id = item["author_id"]
#         id = item["id"]

#         # Sanity checks
#         if len(text) < 5: continue
                
#         # INFAVOR / AGAINST Nodes
#         if item['discussion_type'] not in ["argument_against", "argument_for"]: continue
                                        
#         if item['discussion_type'] == "argument_against":   
#             node_type = "AGAINST"
#         else:
#             node_type = "INFAVOR"
            
#         query = f"""MATCH (p:POSITION {{id:"{item['linked_parent_item']}"}})
#         MERGE (a:{node_type} {{id:"{id}", author_id:"{author_id}", text:"{text}"}})
#         MERGE (p)-[:{"HAS_" + node_type}]-(a)   
#         """

#         graph.query(query)
        
#         # CLUSTER Nodes
#         query = f"""MATCH (a:{node_type} {{id:"{id}"}})
#         MERGE (c:CLUSTER {{id:"{item['cluster_id']}", title:"{item['cluster_title']}", summary:"{item['summary'].replace('"','')}"}})
#         MERGE (c)-[:HAS_POST]->(a)
#         """
        

#         # Execute the second query
#         graph.query(query)
        
#         # KEYPHRASE Nodes
#         for keyphrase in item['keyphrase']:
#             query = f"""MATCH (c:CLUSTER {{id:"{item['cluster_id']}"}})
#             MERGE (k:KEYPHRASE {{keyphrase:"{keyphrase}"}})
#             MERGE (c)-[:HAS_KEYPHRASE]->(k)
#             """
            
#             # Execute the third query
#             graph.query(query)
        
#         # Get triples
#         triplets = retrieve_triplets(input=text)

#         if triplets is None:
#             continue

#         # Import triples in KG
#         for triplet in triplets:

#             if(triplet['node_1'] == '' or triplet['node_2'] == '' or triplet['edge'] == ''):
#                 continue
            
#             relation = triplet['edge'].replace(" ", "_").replace(";", "").replace("-","_").replace(",","").replace("'","").upper()
#             triplet['node_1'] = triplet['node_1'][1:] if triplet['node_1'][0] == " " else triplet['node_1']
#             triplet['node_2'] = triplet['node_2'][1:] if triplet['node_2'][0] == " " else triplet['node_2']

#             # Text2KG
#             query = f"""MATCH (a:{node_type} {{id:"{item["id"]}"}})
#                 MERGE (e1:ENTITY {{value: "{triplet['node_1']}"}})
#                 MERGE (e2:ENTITY {{value: "{triplet['node_2']}"}})
#                 MERGE (a)-[:MENTION]-(e1)
#                 MERGE (a)-[:MENTION]-(e2)
#                 MERGE (e1)-[:{relation}]-(e2)
#             """
            
#             graph.query(query)
            
#             # Enrich ENTITY nodes directly using Wikidata
#             node_1_wiki = wiki_info(triplet['node_1'])
#             node_2_wiki = wiki_info(triplet['node_2'])
            
#             # Add WIKI Nodes
#             for key, value in node_1_wiki.items():  # Iterate over the dictionary key-value pairs
#                 node_1_wiki = value  # Directly assign the list to node_1_wiki
#                 if node_1_wiki:
#                     for element in node_1_wiki[0].get('Entities', []):

#                         if (element['Text'] is None):
#                             continue
            
#                         query = f"""
#                         MATCH (e1:ENTITY {{value: "{triplet['node_1']}"}})
#                         MERGE (w:WIKI {{label: "{element['Text']}", tag: "{element['Label']}", wiki_id: "{element['WIKI_ID']}", wiki_url: "{element['WIKI_URL']}", description: "{element['Description']}"}})
#                         MERGE (e1)-[:HAS_WIKI_DATA]->(w)
#                         """
#                         graph.query(query)


#             for key, value in node_2_wiki.items():  # Iterate over the dictionary key-value pairs
#                 node_2_wiki = value  # Directly assign the list to node_2_wiki
#                 if node_2_wiki:
#                     for element in node_2_wiki[0].get('Entities', []):

#                         if (element['Text'] is None):
#                             continue
            
#                         query = f"""
#                         MATCH (e2:ENTITY {{value: "{triplet['node_2']}"}})
#                         MERGE (w:WIKI {{label: "{element['Text']}", tag: "{element['Label']}", wiki_id: "{element['WIKI_ID']}", wiki_url: "{element['WIKI_URL']}", description: "{element['Description']}"}})
#                         MERGE (e2)-[:HAS_WIKI_DATA]->(w)
#                         """
#                         graph.query(query)