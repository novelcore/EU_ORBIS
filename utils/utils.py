"""
This script is designed for text preprocessing, model loading, and data extraction. 
It includes functions for cleaning and normalizing text, removing stopwords, and handling special characters. 
It also supports loading and managing various NLP models, including spaCy and Hugging Face models. 
Additionally, the script extracts and processes discussion data from a specific format,
organizing positions and associated arguments into a structured DataFrame. 
The script facilitates the setup and utilization of NLP models for various tasks, 
including text preprocessing and discussion analysis.
"""

import re
import nltk
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from LMRank.model import LMRank
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple, Set, TypeVar

# Generic type classes.
Model = TypeVar('Model')

nltk.download('punkt', quiet=True)

def text_preprocess(text:str=None)->str:
    '''
        Text preprocessing

        Parameters
        ----------
        text: input text
        stripping_accents_and_lowercasing: True/False if the user requires stripping accents & lowercasing
        Returns
        -------
        preprocessed text
    '''
    # Remove "\n"
    text = text.replace('.\n', '. ')
    text = text.replace("-\n", '')
    text = text.replace('\n', ' ')
    # Replace tabs
    text = text.replace('\t', ' ')
    # Remove multiple spaces using a regular expression
    text = re.sub(' +', ' ', text) 
    text = re.sub(r'\.+', ".", text)
    # Remove errors
    text = re.sub('\xa0', ' ', text)
    text = text.replace("\x07","")
        
    return text

def remove_stopwords(x: str, stopwords: set) -> str:
    """
    Removes stopwords from the input string.

    Parameters:
    x (str): The input string from which stopwords will be removed.
    stopwords (set): A set of stopwords to be removed from the input string.

    Returns:
    str: A string with the stopwords removed.
    """
    return ' '.join([t for t in x.split() if t not in stopwords])


def get_clean(x: str = None, special_characters: bool = False,
                             stopwords: bool = False, 
                             lowercase: bool = False) -> str:
    '''
    Clean and preprocess text data for NER task.

    Parameters
    ----------
    x: str
        The input text to be cleaned.
    lemma: bool, optional
        Perform lemmatization if True.
    special_characters: bool, optional
        Remove special characters if True.
    stopwords: bool, optional
        Remove stopwords if True.
    lowercase: bool, optional
        Convert text to lowercase if True.

    Returns
    -------
    str
        The cleaned and preprocessed text.
    '''

    # Convert text to lowercase if specified
    if lowercase:
        x = str(x).lower()

    # Remove URLs
    x = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)

    # Remove emails
    x = re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)

    # Remove HTML tags
    x = BeautifulSoup(x, 'lxml').get_text().strip()

    # Remove special characters if specified
    if special_characters:

        x = re.sub(r'’', "'", x)
        x = re.sub(r'‘', "'", x)
        x = re.sub(r'´', "'", x)
        x = re.sub(r'–', '-', x)
        x = re.sub(r':', '', x)
        x = re.sub(r'…', "...", x)
        x = re.sub(r'"', r"'", x)
        x = re.sub(r'“', r"'", x)
        x = re.sub(r'”', r"'", x)
        x = re.sub(r'\[\.\]', '', x)
        x = re.sub(r'\u200E', '', x)
        x = re.sub(r'.\n', '. ', x)
        x = re.sub(r"-\n", '', x)
        x = re.sub(r'\n', ' ', x)
        x = re.sub(r'\t', ' ', x)

        # Convert looove -> love
        x = re.sub("(.)\\1{2,}", "\\1", x)

        # Remove extra spaces
        x = re.sub(r'\s+', ' ', x)

        # Remove square brackets []
        x = re.sub(r'[\[\]]', '', x)

        # Replace multiple punctuations into one
        x = re.sub(r'([!@#$%^&*()_+\-=\[\]{};:\'",.<>?\\/|`~])\1+', r'\1', x)

    # Remove extra spaces
    x = re.sub(r'\s+', ' ', x)

    # Remove stopwords if specified
    if stopwords:
        x = remove_stopwords(x, stopwords=stopwords)

    # Remove empty parenthesis
    x = re.sub(r'\(\s*\)|\[\s*\]|\{\s*\}', '', x)

    # Replace multiple punctuations into one
    x = re.sub(r'([!@#$%^&*()_+\-=\[\]{};:\'",.<>?\\/|`~])\1+', r'\1', x)

    return x


def retrieve_spacy_model(model_name: str) -> Model:
    """
    Function which loads or downloads, 
    the required nlp model, while disabling
    a list of unnecessary components.
    
    Parameters
    ----------
    model_name: path to spaCy model (str).

    Returns
    -------
    nlp: the spacy nlp object (Model)
    """
    disable_list = [
        'ner',
        'entity_linker',
        'entity_ruler',
        'textcat',
        'textcat_multilabel',
        'transformer'
    ]
    try:
        nlp = spacy.load(model_name, disable = disable_list)
    except OSError:
        message = f'First time setup: Downloading the {model_name} NLP model....'
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp

def load_models(
        spacy_model: str = 'en_core_web_sm', 
        language_model: str = 'facebook/bart-large-cnn',
        device: str = 'cuda:0'
    ) -> Tuple[Model, Model, Model, Model]:
    """
    Utility function which loads the required models.
    
    Parameters
    ------------
    
    spacy_model: path to spacy model (str).
    language_model: path to short language model (str).
    long_language_model: path to long language model (str).
    device: device to load and run model ['cuda', 'cuda:0', 'cpu'] (str).

    Returns
    --------
    <object>: All model objects (Tuple[Model, Model, Model, Model, Model]).
    """
    # Load the spacy english NLP model.
    nlp = retrieve_spacy_model(spacy_model)

    # Load the tokenizers and pre-trained language models automatically.
    tokenizer = AutoTokenizer.from_pretrained(language_model, model_max_length = 1024, truncation = True, padding = 'max_length')
    language_model = AutoModelForSeq2SeqLM.from_pretrained(language_model)

    # Send the language models to the pre-specified device (cpu / gpu).
    language_model = language_model.to(device)

    key_ext_model = LMRank(language_setting = 'english')

    return (nlp, tokenizer, language_model, key_ext_model)


def load_discussion(bcause_data: List[dict]) -> Tuple[Set[str], Set[str]]:
    """
    Load a specific discussion from bcause_data and extract positions, arguments for, and arguments against.

    Args:
        bcause_data (list): A list of dictionaries representing bcause data.

    Returns:
        tuple: A tuple containing sets of positions, arguments for, and arguments against.
    """
    # Load posts from the specified discussion
    positions = []
    arguments_in_favor_dict = {}
    arguments_against_dict = {}

     # Iterate through posts to extract positions and arguments
    for post in bcause_data['posts']:
        # Extracting discussion type, id, and text
        discussion_type = post.get('discussion_type')
        position_id = post.get('id')
        position_text = post.get('text')

        # Only process if the post is a position
        if discussion_type == 'position':
            positions.append({'Position_ID': position_id, 'Position_Text': position_text})

            arguments_in_favor = []
            arguments_against = []

            # Iterate through linked children items
            for child_id in post.get('linked_children_items', []):
                # Search for the child item in the posts
                child_item = next((child for child in bcause_data['posts'] if child.get('id') == child_id), None)
                if child_item:
                    child_text = child_item.get('text')
                    # Categorize child item based on its discussion type
                    if child_item['discussion_type'] == 'argument_for':
                        arguments_in_favor.append(child_text)
                    elif child_item['discussion_type'] == 'argument_against':
                        arguments_against.append(child_text)

            # Store arguments for and against for each position
            arguments_in_favor_dict[position_id] = arguments_in_favor
            arguments_against_dict[position_id] = arguments_against

    # Create DataFrame for positions
    df = pd.DataFrame(positions)

    # Add columns for arguments in favor and against
    df['Arguments_In_Favor'] = df['Position_ID'].map(arguments_in_favor_dict)
    df['Arguments_Against'] = df['Position_ID'].map(arguments_against_dict)
    
    return df