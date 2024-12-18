import yaml
import json
import warnings
warnings.filterwarnings("ignore")
from utils.download_dataset import get_discussions, get_discussion_details


if __name__ == "__main__":

    # Load YAML configuration file
    with open("config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Access dataset paths from the configuration
    input_dir_discussions = config['general']['input_path_discussions']
    input_dir_discussion = config['general']['input_path_discussion']
    
    # Initialize all urls and the authorization token.
    auth_bcause_settings = config['general']['auth_bcause_settings']

    
    # Get the list of discussions
    discussions = get_discussions(
                                    settings=auth_bcause_settings
                                )
    
    # Save all discussion data in a single .json file.
    with open(input_dir_discussions, 'w', encoding = 'utf-8') as f:
        json.dump(discussions, f, ensure_ascii = False, indent = 4, separators = (',', ':'))
    
    # Load a specific discussion. For now, set index to 0 to retrieve a random discussion.
    if discussions:
        
        index = 0
        discussion_id = discussions[index]['id']  
        discussion_details = get_discussion_details(
                                                        discussion_id=discussion_id, 
                                                        settings=auth_bcause_settings
                                                    )
        
        discussion_details["author_id"] = discussions[index]["author_id"]
        discussion_details["creation_timestamp"] = discussions[index]["creation_timestamp"]
        discussion_details["last_update_timestamp"] = discussions[index]["last_update_timestamp"]
        discussion_details["title"] = discussions[index]["title"]
        discussion_details["tagline"] = discussions[index]["tagline"]
        discussion_details["image"] = discussions[index]["image"]
       
        # Save a specific discussion data in a single .json file.
        with open(input_dir_discussion, 'w', encoding = 'utf-8') as f:
            json.dump(discussion_details, f, ensure_ascii = False, indent = 4, separators = (',', ':'))