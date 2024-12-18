import yaml
import json
import warnings
warnings.filterwarnings("ignore")
from utils.download_dataset import fetch_polis_comments, fetch_polis_conversation


if __name__ == "__main__":

    # Load YAML configuration file
    with open("config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Access dataset paths from the configuration
    input_dir_discussions = config['general']['input_path_discussions']
    input_dir_discussion = config['general']['input_path_discussion']
    
    # Initialize all urls and the authorization token.
    auth_polis_settings = config['general']['auth_polis_settings']
    
    # Initialize Polis Conversation id
    conversation_id = config['general']['conversation_id']
        
    # Get the conversation
    conversation = fetch_polis_conversation(
        conversation_id=conversation_id, 
        settings=auth_polis_settings
    )
    
    comments = fetch_polis_comments(
        conversation_id=conversation_id,
        settings=auth_polis_settings
    )
    
    # Save all conversation data
    with open(input_dir_discussions, 'w', encoding = 'utf-8') as f:
        json.dump(conversation, f, ensure_ascii = False, indent = 4, separators = (',', ':'))
        
    with open(input_dir_discussion, 'w', encoding = 'utf-8') as f:
        json.dump(comments, f, ensure_ascii = False, indent = 4, separators = (',', ':'))