import os
import ast
import torch
import pickle
import numpy as np
from utils.utils import *
import umap.umap_ as umap
from openai import OpenAI
from typing import Any, Dict
from utils.summarization import *
from sklearn.cluster import KMeans
from .title_prompts import gpt_prompt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

def find_optimal_clusters(data: np.ndarray, min_clusters: int = 2, max_clusters: int = 10) -> int:
    """
    Finds the optimal number of clusters for KMeans clustering using the silhouette score.

    Parameters:
    data (np.ndarray): The input data for clustering. Should be a 2D array where each row is a sample and each column is a feature.
    min_clusters (int): The minimum number of clusters to consider. Default is 2.
    max_clusters (int): The maximum number of clusters to consider. Default is 10.

    Returns:
    int: The optimal number of clusters based on the silhouette score.
    """
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    
    if min_clusters < 2:
        raise ValueError("Minimum number of clusters must be at least 2.")
    
    if max_clusters <= min_clusters:
        raise ValueError("Maximum number of clusters must be greater than the minimum number of clusters.")
    
    optimal_clusters = min_clusters
    highest_silhouette_score = -1
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        
        if silhouette_avg > highest_silhouette_score:
            highest_silhouette_score = silhouette_avg
            optimal_clusters = n_clusters
    
    return optimal_clusters

def clustering(embeddings: np.ndarray=None, print_stat:bool=True, tuning: bool=True) -> Tuple[np.ndarray, int]:
    """
        Perform clustering using either HDBSCAN or SoftDBSCAN algorithm.

        Parameters:
            embeddings (np.ndarray): The data embeddings for clustering.
            print_stat (bool): Whether to print clustering statistics. Default is True.
            tuning (bool): Whether to perform hyperparameter tuning. Default is True.

        Returns:
            Tuple[np.ndarray, int]: Tuple containing cluster labels and total number of clusters found.
    """
    
    if(tuning):
        # Find the optimal number of clusters
        num_clusters = find_optimal_clusters(embeddings, 2, 15)
    else:
        num_clusters = 3

    # Clustering
    clustering_model = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    clustering_model.fit(embeddings)
    labels = clustering_model.labels_

    if(print_stat):

        silhouette_avg = silhouette_score(embeddings, labels)

        print("\033[92m[INFO]\033[0m" + f" Silhouette Score: {silhouette_avg:.3f}")
        print("\033[92m[INFO]\033[0m" + f" Total Clusters found: {num_clusters}")

    return labels, num_clusters, clustering_model


def summarize_posts(cluster_dict: Dict[int, Any], device: str='cpu', models: Tuple[Any, Any, Any, Any]=(None, None, None, None)) -> Dict[int, str]:
    """
    Summarize the posts of each cluster.

    Parameters:
        cluster_dict (Dict[int, Any]): A dictionary containing clusters where keys are cluster labels and values are lists of posts.
        device (str): The device for model inference. Default is 'cpu'.
        models (Tuple[Any, Any, Any, Any]): Tuple containing loaded models.

    Returns:
        Dict[int, str]: A dictionary containing summaries of each cluster.
    """

    # Load required models
    nlp, tokenizer, language_model, _ = models

    # Summarize the posts of each cluster
    posts_summaries = {
        int(label): hybrid_summarization(posts, nlp, tokenizer, language_model, device=device) 
        for label, posts in cluster_dict.items()
    }

    return posts_summaries

def generate_keyphrases(cluster_dict: Dict[int, Any], models: Tuple[Any, Any, Any, Any]) -> Dict[int, str]:
    """
    Extract the top-5 keyphrases from each cluster.

    Parameters:
        cluster_dict (Dict[int, Any]): A dictionary containing clusters where keys are cluster labels and values are lists of posts.
        models (Tuple[Any, Any, Any, Any]): Tuple containing loaded models.

    Returns:
        Dict[int, str]: A dictionary containing keyphrases of each cluster.
    """

    # Load required models
    _, _, _, key_ext_model = models

    # Extract the top-10 keyphrases from each cluster
    posts_keyphrases = {
        int(label): extract_keyphrases(posts, key_ext_model, top_n=5, deduplicate=True)
        for label, posts in cluster_dict.items()
    }

    return posts_keyphrases

def generate_titles(summaries_text: str=None, num_clusters: int=0, subject: str=None) -> Dict[int, str]:
    """
    Generate titles for each cluster using specified models.

    Parameters:
        num_clusters (int): The number of clusters.
        subject (str): The subject of the clusters. Default is None.

    Returns:
        Dict[int, str]: A dictionary containing titles for each cluster.
    """

    # Generate prompt for GPT-3.5 Turbo
    prompt = gpt_prompt.format(
                                    num_topics=num_clusters, 
                                    subject=subject, 
                                    summaries_text=summaries_text
                                )

    # Prepare messages for ChatGPT-based models
    messages = [
            { "role": "system", "content": "You are an AI expert in creating titles for specific topics based on a list of summaries." },
            { "role": "user", "content": prompt },
        ]

    client = OpenAI()
    response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            max_tokens=200,
                            temperature=0,
                        )

    # Extract content from the response
    generated_titles = response.choices[0].message.content

    # Check if the titles are in a valid format
    try:
        generated_titles = ast.literal_eval(generated_titles)
        if not isinstance(generated_titles, dict):
            print("\033[91mError:\033[0m The generated titles are not valid dictionaries.")
    except (SyntaxError, ValueError):
        print("\033[91mError:\033[0m Unable to evaluate the generated titles.")

    return generated_titles

def clustering_details(cluster_dict: Dict[int, Any], subject: str=None, num_clusters: int=0, device: str='cpu') -> Dict[str, Any]:
    """
    Generate cluster summaries, keyphrases, and titles using specified models.

    Parameters:
        cluster_dict (Dict[int, Any]): A dictionary containing clusters where keys are cluster labels and values are lists of posts.
        subject (str): The subject of the clusters. Default is None.
        num_clusters (int): The number of clusters. Default is 0.
        device (str): The device for model inference. Default is 'cpu'.

    Returns:
        Dict[str, Any]: A dictionary containing cluster details including titles, keyphrases, and summaries.
    """

    # Load required models
    models = load_models(device=device)

    # Generate summaries and keyphrases
    posts_summaries = summarize_posts(cluster_dict, device, models)
    posts_keyphrases = generate_keyphrases(cluster_dict, models)

    # Collect and prepare the summaries and the keyphrases in a paragraph
    keyphrases_text = ""
    for topic, keyphrases in list(posts_keyphrases.items()):
        if(topic != -1):
            keyphrases_text += f"- Topic {topic} keyphrases are: \"{keyphrases}\"\n"

    summaries_text = ""
    for topic, summary in list(posts_summaries.items()):
        if(topic != -1):
            summaries_text += f"- Topic {topic} summary is: \"{summary}\"\n"

    # Generate titles
    cluster_titles = generate_titles(
                                        summaries_text=summaries_text,
                                        num_clusters=num_clusters, 
                                        subject=subject, 
                                    )

    # Collect all results together
    results = {
                'clusters': cluster_dict,
                'cluster_titles': cluster_titles,
                'keyphrases': posts_keyphrases,
                'summaries': posts_summaries 
            }

    return results


def clustering_and_preprocess(bcause_data: Dict[str, Any]=None, embs_model: str = None, cluster_threshold: int=6, save_model: bool=False) -> Dict[str, Any]:
    """
    Perform clustering and preprocessing on the input DataFrame of Positions and Arguments.

    Args:
        bcause_data (Dict[str, Any]): The JSON object containing bcause data.
        model_name (str): The name of the SentenceTransformer embeddings model to load.
        cluster_threshold (int): The threshold for the number of arguments required to perform clustering.
        save_model (bool): Whether to save the model. Default is False.

    Returns:
        Dict[str, Any]: Processed JSON object containing clustering information.
    """

    # Store each Position with its Arguments against and in favor in a Dataframe
    df = load_discussion(bcause_data)

    # Clean Data (Positions)
    target_column = "Position_Text"
    df[target_column] = df[target_column].apply(get_clean, special_characters=True, lowercase=True)
    
    # Merge lists for Arguments_In_Favor and Arguments_Against 
    # columns for duplicates and keep only unique values
    df = df.groupby('Position_Text').agg({
        'Arguments_In_Favor': lambda x: list(set(sum(x, []))),
        'Arguments_Against': lambda x: list(set(sum(x, []))),
        'Position_ID': 'first'  # Keep the first Position_ID
    }).reset_index()

    # Clean Data (In favor / Against)
    df["Arguments_Against"] = df["Arguments_Against"].apply(lambda lst: [get_clean(string, special_characters=True, lowercase=True) for string in lst])
    df["Arguments_In_Favor"] = df["Arguments_In_Favor"].apply(lambda lst: [get_clean(string, special_characters=True, lowercase=True) for string in lst])

    # Set device.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    target_cols = ['Arguments_In_Favor', 'Arguments_Against']
    position_col = "Position_Text"
    id_col = "Position_ID"

    all_results = []
    sent_model = SentenceTransformer(embs_model)

    for i in tqdm(range(df.shape[0])):
        for target_col in target_cols:
            query = df[position_col].iloc[i]
            data = df[target_col].iloc[i]
            pos_id = df[id_col].iloc[i]

            results = {}
            
            if(len(data) != 0):

                # Here we assume that we perform clustering only for more than 5 feedbacks
                if(len(data) >= cluster_threshold):

                    # Create embeddings
                    embeddings = sent_model.encode(data, show_progress_bar=False, device = device)
                    
                    #Scaling
                    scaler = StandardScaler()
                    scaled_embeddings = scaler.fit_transform(embeddings)
                    
                    # Dimensionality Reduction
                    umap_model = umap.UMAP( n_neighbors=30,
                                            n_components=10 if len(data) >= 15 else 5 if len(data) > 6 else 3,
                                            metric='cosine',
                                            random_state=42,
                                            verbose=False
                                        )

                    umap_embeddings = umap_model.fit_transform(scaled_embeddings)

                    
                    # HDBScan
                    soft_clusters, total_clusters, model = clustering(
                                                                embeddings=umap_embeddings,
                                                                print_stat=False,
                                                                tuning=False
                                                            )

                    if(save_model):
                        model_path = "Data/models/" + str(pos_id) + "_" + str(target_col) + ".pkl"
                        
                        with open(model_path, "wb") as f:
                            pickle.dump(model, f)
                    
                    # Save the results in a dictionary
                    cluster_dict = {c: [] for c in range(total_clusters)}

                    for cluster_label, data_text in zip(soft_clusters, data):
                        cluster_dict[cluster_label].append(data_text)

                    # Cluster Details
                    results = clustering_details(
                                                    cluster_dict=cluster_dict,
                                                    subject=query,
                                                    num_clusters=total_clusters,
                                                    device=device,
                                                )

                else:
                    
                    # Here we assume that we have only one cluster
                    cluster_dict = {0: []}
                    total_clusters = 1

                    for data_text in data:
                        cluster_dict[0].append(data_text)

                    # Cluster Details
                    results = clustering_details(
                                                    cluster_dict=cluster_dict,
                                                    subject=query,
                                                    num_clusters=total_clusters,
                                                    device=device
                                                )

            # Save the results in a json format
            if(results != {}):
                # Add additional information to results
                results['Position_Text'] = query
                results['Position_ID'] =  pos_id
                results['Arguments_Type'] = 'In_Favor' if target_col == 'Arguments_In_Favor' else 'Against'
                all_results.append(results)
    
    # Iterate through each post in the JSON data
    for post in bcause_data['posts']:
        if post['discussion_type'] in ['argument_against', 'argument_for']:
            # Find the corresponding Position_ID in the results
            position_id = post['linked_parent_item']
            argument_text = post['text']
            for result in all_results:
                if result.get('Position_ID') == position_id:

                    # Access the 'clusters' key and its values
                    clusters = result['clusters']
                    
                    # Iterate over the clusters
                    for cluster_id, cluster_values in clusters.items():
                        for argument in cluster_values:
                            if (get_clean(argument_text, special_characters=True, lowercase=True) == argument):
                                new_cluster_id = str(position_id) + "_" + str(post['discussion_type']) + "_" + str(cluster_id)
                                post['cluster_id'] = new_cluster_id
                                post['cluster_title'] = result.get('cluster_titles')[cluster_id]
                                post['keyphrase'] = result.get('keyphrases')[cluster_id]
                                post['summary'] = result.get('summaries')[cluster_id]

    # Set -1 cluster to the dublicated feedbacks
    for post in bcause_data['posts']:
        if post['discussion_type'] in ['argument_against', 'argument_for']:
            if 'cluster_id' not in post:
                position_id = post['linked_parent_item']
                post['cluster_id'] = str(position_id) + "_-1"
                
    return bcause_data