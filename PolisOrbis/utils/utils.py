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
from typing import List, Tuple, Set, TypeVar
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from opinion_classification_prompts import get_label_from_document

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


def process_feedback_result(label: str = None, valid_labels: list = None) -> str:
    """
    Process the feedback label obtained from the model.

    Args:
    - label (str): The label obtained from the model.
    - valid_labels (list): A list of valid labels to be accepted.

    Returns:
    - str: Processed label. 
    """
    lowercase_label = label.lower().replace('\n', '')

    # Regular expression pattern to match any of the valid labels
    pattern = r'\b(?:' + '|'.join(valid_labels) + r')\b'

    # Search for valid labels in the lowercase label
    matches = re.findall(pattern, lowercase_label)

    # Check if more than one valid label is present
    if len(matches) > 1:
        return "invalid"
    elif matches:
        return matches[0]  # Return the first matched label
    else:
        return "invalid"

def opinion_classification(data: List[str]=[], query: str=None, method: str="gpt-35-turbo") -> Tuple[List[str], List[str]]:
    """
    Classifies user feedback data as 'in favor' or 'against' based on the specified method.

    Parameters:
    data (List[str]): A list of user feedback strings to be classified.
    query (str, optional): The query or context for classification. Default is None.
    method (str, optional): The method to use for classification. Options are "gpt-35-turbo".

    Returns:
    Tuple[List[str], List[str]]: Two lists of classified data - the first list contains feedback classified as 'against',
                                 and the second list contains feedback classified as 'in favor'.
    """
    
    if method.startswith("gpt"):
        labels = []
        processed_labels = []

        for user_feedback in data:
            
            # Perform Classification and parse the output of LLM
            label = get_label_from_document(method, user_feedback, query)

            valid_labels = ["in favor", "against"]
            processed_label = process_feedback_result(label, valid_labels)
            
            # Append the results to the lists
            labels.append(label)
            processed_labels.append(processed_label)
        
        # Create new_data based on processed_labels
        against_data = [data[i] for i, label in enumerate(processed_labels) if label == "against"]
        in_favor_data = [data[i] for i, label in enumerate(processed_labels) if label == "in favor"]

        return against_data, in_favor_data
    
    else:
        raise ValueError(f"Invalid method name: {method}. Expected GPT based.")