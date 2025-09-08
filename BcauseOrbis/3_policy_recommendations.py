import os
import yaml
import json
import warnings
warnings.filterwarnings("ignore")
from utils.policy_recommendations import process_bcause_data

if __name__ == "__main__":

    # Load YAML configuration file
    with open("config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Access dataset paths from the configuration
    input_dir = config['general']['output_cluster_dir']
    output_dir = config['general']['output_policy_dir']

    # Set OpenAI api-key
    your_api_key = config['model_parameters']['openai_api_key']
    os.environ['OPENAI_API_KEY'] = str(your_api_key)
    
    print("\033[92m[INFO]\033[0m Starting policy recommendation generation...")
    
    # Read clustered data from the .json file
    with open(input_dir, 'r', encoding='utf-8') as file:
        clustered_data = json.load(file)
    
    print(f"\033[92m[INFO]\033[0m Loaded clustered data from: {input_dir}")
    
    # Generate policy recommendations
    policy_recommendations = process_bcause_data(clustered_data)
    
    print(f"\033[92m[INFO]\033[0m Generated recommendations for {len(policy_recommendations)} clusters")
    
    # Save policy recommendations to JSON file
    with open(output_dir, 'w', encoding='utf-8') as json_file:
        json.dump(policy_recommendations, json_file, ensure_ascii=False, indent=4, separators=(',', ':'))
    
    print(f"\033[92m[INFO]\033[0m Policy recommendations saved to: {output_dir}")
    print("\033[92m[SUCCESS]\033[0m Policy recommendation generation completed!")
