"""
This script interacts with Wikidata to fetch descriptions and 
uses the Refined model for entity extraction from text. It retrieves
descriptions of entities from Wikidata and processes text to identify and 
detail entities, including their Wikipedia titles, labels, Wikidata IDs, 
and descriptions, while providing links to their Wikidata pages. 
The script supports GPU acceleration for enhanced performance.
"""

import torch
from typing import Dict, Any  
from wikidata.client import Client
from utils.ReFinED.src.refined.inference.processor import Refined

client = Client()

def get_entity_description(wikidata_entity_id: str=None) -> str:
    """
    Get the description of a Wikidata entity.

    Args:
        wikidata_entity_id (str): The ID of the Wikidata entity to retrieve the description for.

    Returns:
        str: The description of the Wikidata entity. If the description is not available, returns "Description not available".
    """
    entity = client.get(wikidata_entity_id, load=True)

    try:
        description = entity.description
        return description
    except KeyError:
        return "Description not available"

def wiki_info(text: str) -> Dict[str, Any]:
    """Process a single text using Refined and extract entity information from Wikipedia.

    Args:
        text (str): The input text to process.

    Returns:
        Dict[str, Any]: A dictionary containing the results.
    """
    refined = Refined.from_pretrained(  model_name='wikipedia_model_with_numbers',
                                        entity_set="wikipedia",
                                        use_precomputed_descriptions=True,
                                        device="cuda:0" if torch.cuda.is_available() else "cpu"
                                      )
    results = {}
    item_results = []

    for item in refined.process_text_batch(texts=[text], return_special_spans=False, max_batch_size=8):
        entity_info = {}
        entity_info["Entities"] = []

        for x in item.spans:
            # print(x.end)
            entity = {}
            entity["Original_Text"] = x.text
            entity["Text"] = x.predicted_entity.wikipedia_entity_title
            entity["Label"] = x.coarse_mention_type
            entity["WIKI_ID"] = x.predicted_entity.wikidata_entity_id

            if x.predicted_entity.wikidata_entity_id is not None:
                entity["WIKI_URL"] = f"https://www.wikidata.org/wiki/{x.predicted_entity.wikidata_entity_id}"
                entity["Description"] = get_entity_description(x.predicted_entity.wikidata_entity_id)
            else:
                entity["WIKI_URL"] = None
                entity["Description"] = None

            entity_info["Entities"].append(entity)

        item_results.append(entity_info)

    results[text] = item_results

    return results