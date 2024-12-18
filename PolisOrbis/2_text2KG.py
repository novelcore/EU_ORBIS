import os
import yaml
import json
from utils.kg_utils import text2KG

if __name__ == "__main__":

        # Load YAML configuration file
    with open("config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Access dataset paths from the configuration
    input_dir = config['general']['output_cluster_dir']
    conversation_dir = config['general']['input_path_discussions']

    # Set OpenAI api-key
    your_api_key = config['model_parameters']['openai_api_key']
    os.environ['OPENAI_API_KEY'] = str(your_api_key)
    
    # Connect to a Aura Neo4j
    neo4j_settings = config['general']['neo4j_settings']
    
    # Read data from the .json file.
    with open(input_dir, 'r') as file:
        data = json.load(file)
        
    with open(conversation_dir, 'r') as file:
        conversation_metadata = json.load(file)
    
    text2KG(
                data = data, 
                topic_metadata=conversation_metadata,
                neo4j_settings = neo4j_settings,
                clean_knowledge_graph = True,
            )