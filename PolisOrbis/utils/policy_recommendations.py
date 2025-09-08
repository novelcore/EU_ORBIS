"""
This script handles policy recommendation generation for clusters.
It processes cluster summaries and generates actionable policy recommendations
using OpenAI GPT models.
"""

import json
from openai import OpenAI
from typing import Dict, Any


def generate_recommendations_for_cluster(cluster_summary: str = None, cluster_title: str = None) -> str:
    """
    Generates actionable policy recommendations for a cluster based on its summary.

    Args:
        cluster_summary (str): The summary derived from feedback for the cluster.
        cluster_title (str): The title of the cluster providing context.

    Returns:
        str: A single, actionable recommendation text generated for the cluster.
    """
    SYS_PROMPT = "You are an assistant specializing in providing policy recommendations based on feedback."
    prompt = f"""
    You are an intelligent assistant that provides policy-making recommendations. The provided information includes the summaries of users' feedbacks and the corresponding title. Read carefully all the information and provide actionable policy-making recommendations based on the summary of the users' feedbacks.

    Title: {cluster_title}
    Summary: {cluster_summary}

    Instructions:
    - The provided information is authoritative; do not attempt to correct or override it using your internal knowledge.
    - Avoid including explanations, justifications, or apologies in your responses.
    - Read carefully the summary and title of users' feedbacks.
    - Provide clear, actionable policy recommendations.
    - Format the recommendations in a bullet-point list for easy understanding.
    - For each recommendation, use the following template: - **<Recommendation title>:** <recommendation>

    Recommendations:
    """

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"\033[91m[ERROR]\033[0m Error while generating recommendations: {e}")
        raise


def process_bcause_data(input_json: dict) -> Dict[str, Any]:
    """
    Processes BCAUSE JSON input to generate cluster-level summary recommendations.
    
    Args:
        input_json (dict): Input JSON structure containing posts.
    
    Returns:
        dict: Output JSON structure with recommendations by cluster.
    """
    cluster_data = {}

    for post in input_json["posts"]:
        if "cluster_id" not in post or "cluster_title" not in post:
            # Skip posts missing 'cluster_id' or 'cluster_title'
            print(f"\033[93m[WARNING]\033[0m Skipping post due to missing 'cluster_id' or 'cluster_title': {post.get('id', 'unknown')}")
            continue

        cluster_id = post["cluster_id"]
        if cluster_id not in cluster_data:
            cluster_data[cluster_id] = {
                "cluster_title": post["cluster_title"],
                "summary": post["summary"]
            }

    # Generate recommendations for each cluster
    output_data = {}
    for cluster_id, data in cluster_data.items():
        cluster_title = data["cluster_title"]
        summary = data["summary"]

        print(f"\033[92m[INFO]\033[0m Generating recommendations for cluster: {cluster_id}")
        
        # Generate recommendations for the cluster summary
        recommendations = generate_recommendations_for_cluster(
            cluster_summary=summary,
            cluster_title=cluster_title,
        )

        # Store the recommendations in the output structure
        output_data[cluster_id] = {
            "cluster_title": cluster_title,
            "recommendations": recommendations
        }

    return output_data


def process_polis_data(input_json: list) -> Dict[str, Any]:
    """
    Processes Polis JSON input to generate cluster-level summary recommendations.
    
    Args:
        input_json (list): Input JSON structure containing posts.
    
    Returns:
        dict: Output JSON structure with recommendations by cluster.
    """
    cluster_data = {}

    for post in input_json:
        if "cluster_id" not in post or "cluster_title" not in post:
            # Skip posts missing 'cluster_id' or 'cluster_title'
            print(f"\033[93m[WARNING]\033[0m Skipping post due to missing 'cluster_id' or 'cluster_title': {post.get('tid', 'unknown')}")
            continue

        cluster_id = post["cluster_id"]
        if cluster_id not in cluster_data:
            cluster_data[cluster_id] = {
                "cluster_title": post["cluster_title"],
                "summary": post["summary"]
            }

    # Generate recommendations for each cluster
    output_data = {}
    for cluster_id, data in cluster_data.items():
        cluster_title = data["cluster_title"]
        summary = data["summary"]

        print(f"\033[92m[INFO]\033[0m Generating recommendations for cluster: {cluster_id}")
        
        # Generate recommendations for the cluster summary
        recommendations = generate_recommendations_for_cluster(
            cluster_summary=summary,
            cluster_title=cluster_title,
        )

        # Store the recommendations in the output structure
        output_data[cluster_id] = {
            "cluster_title": cluster_title,
            "recommendations": recommendations
        }

    return output_data
