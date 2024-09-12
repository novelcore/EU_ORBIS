"""
This script provides functions to interact with the BCAUSE API. 
The functions handle authentication, URL configuration, and API 
requests, with default settings if not provided.
"""

import time
import requests
from tqdm import tqdm
from typing import Dict, Any, List


def get_json_request(url: str=None, auth_token: str=None, params: Dict = {}):
    """
    Function which performs a GET request and returns a json-like object.

    Parameters
    -----------
    url: http(s) url to connect (str).
    auth_token: authorization token (str).
    params: dict of key parameters and their values (Dict).
   
    Returns
    --------
    A json-like object.
    """

    # Connect to the http url and send the GET request.
    requests_obj = requests.get(
        url, 
        headers = {'Authorization': f'{auth_token}'}, 
        params = params
    )

    # Check if the connection and authorization was succesful.
    # If not raise an exception.
    if requests_obj.status_code != 200:
        raise Exception('BCAUSE: {url} endpoint not working!')
    
    return requests_obj.json()


def get_discussions(settings: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Function to extract all discussions from the BCAUSE API.

    Parameters
    -----------
    settings: Dict[str, str], optional
        Dictionary containing 'auth_token' and 'discussion_url'.
        If None, default values will be used.

    Returns
    --------
    List[Dict[str, Any]]: A list of discussions.
    """
    # Initialize the URL and the authorization token.
    settings = settings or {
        "auth_token": '...',
        "discussion_url": 'https://europe-west1-bcause-alpha01.cloudfunctions.net/getDiscussionsV2/',
    }

    # Get all discussions.
    discussions = get_json_request(settings['discussion_url'], settings['auth_token'])

    return discussions

def get_discussion_details(discussion_id: str, settings: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Function to extract discussion details, including posts and participants, from the BCAUSE API
    based on a specific discussion ID.

    Parameters
    -----------
    discussion_id: str
        The ID of the discussion to retrieve details for.
    settings: Dict[str, str], optional
        Dictionary containing 'auth_token', 'discussion_posts_url', and 'discussion_participants_url'.
        If None, default values will be used.

    Returns
    --------
    Dict[str, Any]: A dictionary containing the discussion details, posts, and participants.
    """
    # Initialize URLs and the authorization token.
    settings = settings or {
        "auth_token": '...',
        "discussion_posts_url": 'https://europe-west1-bcause-alpha01.cloudfunctions.net/getContributionsV2',
        "discussion_participants_url": 'https://europe-west1-bcause-alpha01.cloudfunctions.net/getParticipantsV2',
    }

    # Get all discussion posts based on the discussion ID.
    posts = get_json_request(
        settings['discussion_posts_url'], 
        settings['auth_token'], 
        params={'debateId': discussion_id}
    )

    # Get all discussion participants based on the discussion ID.
    participants = get_json_request(
        settings['discussion_participants_url'], 
        settings['auth_token'], 
        params={'debateId': discussion_id}
    )

    return {
        "id": discussion_id,
        "posts": posts,
        "participants": participants
    }