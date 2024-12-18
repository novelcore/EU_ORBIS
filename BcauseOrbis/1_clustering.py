import os
import yaml
import json
import warnings
warnings.filterwarnings("ignore")
from utils.clustering import clustering_and_preprocess

if __name__ == "__main__":

    # Load YAML configuration file
    with open("config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Access dataset paths from the configuration
    input_dir = config['general']['input_path_discussion']
    output_dir = config['general']['output_cluster_dir']
    embs_model = config['general']['embeddings_model']

    # Set OpenAI api-key
    your_api_key = config['model_parameters']['openai_api_key']
    os.environ['OPENAI_API_KEY'] = str(your_api_key)
    
    # Read data from the .json file.
    with open(input_dir, encoding = 'utf-8', errors = 'ignore') as f:
        bcause_data = json.load(f)
        
    
    bcause_data_with_clusters = clustering_and_preprocess(
                                                            bcause_data=bcause_data, 
                                                            embs_model = embs_model, 
                                                            cluster_threshold = 6, 
                                                            save_model = False
                                                        )
    
    # Save all results in a JSON file
    with open(output_dir, 'w') as json_file:
        json.dump(bcause_data_with_clusters, json_file, ensure_ascii = False, indent = 4, separators = (',', ':'))