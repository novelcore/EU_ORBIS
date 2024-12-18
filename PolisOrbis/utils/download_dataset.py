"""
This script provides functions to interact with the Polis API. 
The functions handle authentication, URL configuration, and API 
requests, with default settings if not provided.
"""

import json
import httpx
from typing import Dict, Optional


def fetch_polis_comments(conversation_id: str=None, settings: Optional[Dict[str, str]] = None)->  Dict:
    """
    Fetches POLIS comments for a specific conversation ID.

    Parameters
    ----------
    conversation_id : str
        The unique ID of the conversation for which comments are to be fetched. This is required to query the POLIS API.
    settings : Optional[Dict[str, str]]
        Optional settings dictionary with server domain.

    Returns
    -------
    Dict:
        A JSON of the fetched comments, or None if an error occurs.
    """
    try:

        # Initialize settings with default values
        settings = settings or {
            "server_domain": "https://polisorbis.copernicani.it",  # Replace with the correct domain
        }

        # Construct the API URL
        api_url = (
            f"{settings['server_domain']}/api/v3/comments?"
            f"moderation=true&mod_gt=-1&include_voting_patterns=true&conversation_id={conversation_id}"
        )
        
        # Fetch data from the Comments API
        with httpx.AsyncClient() as client:
            response = client.get(api_url)
            response.raise_for_status()  # Raise HTTP errors if any
            comments = response.json()  # Parse JSON response
            
        # Log the response
        print(f"Retrieved comments for conversation ID {conversation_id}: {comments}")
                
        json_data = json.dumps(comments, indent=4, separators=(",", ":"))
        return json_data
    
    except httpx.HTTPError as e:
        print(f"HTTP error while fetching comments: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def fetch_polis_conversation(conversation_id: str=None, settings: Optional[Dict[str, str]] = None)->Dict:
    """
    Fetches details of a POLIS conversation for a specific conversation ID.

    Parameters
    ----------
    conversation_id : str
        The unique ID of the POLIS conversation to fetch. This is required to query the POLIS API.
    settings : Optional[Dict[str, str]]
        Optional settings dictionary with server domain.
    -------
    Returns
    -------
    Dict:
        A JSON of the fetched conversation details, or None if an error occurs.
    """
    try:
        # Initialize the URL and the authorization token.
        settings = settings or {
            "server_domain": "https://polisorbis.copernicani.it",  # Replace with the correct domain
        }
        
        # Construct the API URL
        api_url = f"{settings['server_domain']}/api/v3/conversations?conversation_id={conversation_id}"
        
        # Fetch data from the Conversations API
        with httpx.AsyncClient() as client:
            response = client.get(api_url)
            response.raise_for_status()  # Raise HTTP errors if any
            conversation_details = response.json()  # Parse JSON response
            
        # Log the response
        print(f"Retrieved conversation details for ID {conversation_id}: {conversation_details}")

        # Save conversation
        json_data = json.dumps(conversation_details, indent=4, separators=(",", ":"))
        return json_data
    
    except httpx.HTTPError as e:
        print(f"HTTP error while fetching conversation: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None