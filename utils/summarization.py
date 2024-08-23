"""
This script facilitates text summarization and keyphrase extraction. 
It supports combining multiple text fragments into a single string, 
performing both extractive and abstractive summarization, and extracting keyphrases. 
The summarization processes can be fine-tuned with parameters for algorithm choice, 
sentence limits, and token limits, including hybrid methods that integrate both 
extractive and abstractive techniques. Keyphrase extraction is also available,
focusing on extracting and deduplicating the most relevant phrases from the text.
"""

import math
from typing import List, Union, TypeVar

# Generic type class for spaCy / hugginface model objects.
Model = TypeVar('Model')  

def aggregate_text(arguments: Union[List[str], str]) -> str:
    """
    Utility function, which aggregates the text
    from arguments, if necessary.
    
    Parameters
    ------------
    arguments: text of arguments (List [str] / str).

    Exceptions:
    ------------
    ValueError: for unsupported data types.
    
    Returns
    --------
    aggregated_text: the aggregated text of arguments (str).
    """
    if type(arguments) == list:
        aggregated_text = ' '.join(argument for argument in arguments)
    elif type(arguments) == str:
        aggregated_text = arguments
    else: 
        raise ValueError('Summarize() supports either str or List[str]!')
    return aggregated_text


def extractive_summarization(arguments: Union[List[str], str], nlp: Model, 
                             algorithm: str = 'textrank', top_n: int = 20, 
                             top_sent: int = 5, percentage_sentences = 0.0) -> str:
    """
    Function that extracts the top_sent most significant
    sentences from the aggregated text of arguments,
    using a graph-based algorithm. The algorithm 
    is implemented as a spaCy pipeline component.

    Parameters
    -----------
    arguments: text of arguments (List [str] / str)/
    nlp: spaCy nlp model (Model).
    algorithm: algorithm name (str). 
    top_n: number of considered phrases (int).
    top_sent: number of extracted sentences (int).
    percentage_sentences: dynamically extracts a percentages of sentences (float).

    Returns
    --------
    summary: extractive summary of arguments (str)
    """
    
    # Aggregate the arguments.
    text = aggregate_text(arguments)

    # Temporarily add the algorithm to the spaCy pipeline.
    nlp.add_pipe(algorithm, last = True)

    # Create the document using the pipeline.
    doc = nlp(text)

    # Remove the algorithm from the pipeline.
    nlp.remove_pipe(algorithm)

    # If the percentage_sentences is set, set the number of sentences dynamically.
    if percentage_sentences:
        top_sent = math.ceil(percentage_sentences * len(list(doc.sents)))

    # Access the baseline textrank component to perform summarization.
    summary = ' '.join(
        sent.text
        for sent in doc._.textrank.summary(
            limit_phrases = top_n, 
            limit_sentences = top_sent
        )
    )

    return summary


def abstractive_summarization(arguments: Union[List[str], str], tokenizer: Model,
                              language_model: Model, device: str = 'cpu') -> str:
    """
    Function that performs abstractive summarization, 
    given a pre-trained tokenizer and language model as input.
    The parameter max_new_tokens sets the amount of generated tokens.
    Parameters
    -----------
    arguments: text of arguments (List [str] / str).
    tokenizer: huggingface tokenizer model (Model).
    language model: huggingface language model (Model).
    device: device to load and run model ['cuda', 'cuda:0', 'cpu'] (str).

    Returns
    --------
    summary: abstractive summary of arguments (str)
    """

    # Aggregate the arguments.
    text = aggregate_text(arguments)
    
    # Tokenize the text to the max length, encode it and get the input ids.
    input_ids = tokenizer(
        text,
        return_tensors = 'pt',
        padding = 'longest',       
        truncation = True
    )['input_ids'].to(device)

    # Use the pre-trained language model to generate the output ids.
    output_ids = language_model.generate(
        input_ids = input_ids
    )[0].to(device)

    # Decode the output ids into a textual sequence.
    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens = True,
        clean_up_tokenization_spaces = True
    )
    return summary


def hybrid_summarization(arguments: Union[List[str], str], nlp: Model,
                         tokenizer: Model, language_model: Model, 
                         device: str = 'cpu') -> str:
    """
    Function that summarizes a list of arguments or an aggregated text from 
    a list of arguments, using the hybrid (reduce then summarize) strategy 
    by combining extractive and abstractive summarization. This function 
    takes as input the nlp, tokenizer and pre-trained language models.
    
    Parameters
    -----------
    arguments: text of arguments (List [str] / str).
    nlp: spaCy nlp model (Model).
    tokenizer: huggingface tokenizer model (Model).
    language model: huggingface language model (Model).
    device: device to load and run model ['cuda', 'cuda:0', 'cpu'] (str).

    Returns
    --------
    summary: abstractive summary of arguments (str)
    """
    
    # Aggregate the arguments.
    text = aggregate_text(arguments)
    
    # if the aggregated text is empty, return early.
    if text == '':
        return ''

    # Extract the max supported length from the model config.
    max_length = language_model.config.max_position_embeddings

    # Tokenize the text to find the number of sentences and tokens.
    number_of_sentences = len(list(nlp(text).sents))
    number_of_tokens = len(tokenizer.tokenize(text))
    avg_tokens_per_sentence = number_of_tokens // number_of_sentences

    # Check to see if the text surpasses the max length of the model.
    if number_of_tokens < max_length:
        summary = abstractive_summarization(text, tokenizer, language_model, device)
    else: # If it does, reduce the text by selecting its top-n sentences.
        # The constant factor takes into account the mismatch between 
        # the number of subword tokens and word-level tokens.
        top_n_sentences = int(max_length / (avg_tokens_per_sentence * 1.15))
        reduced_text = extractive_summarization(text, nlp, top_sent = top_n_sentences)
        summary = abstractive_summarization(reduced_text, tokenizer, language_model, device) 

    return summary


def hybrid_summarization_percentage(arguments: Union[List[str], str], nlp: Model,
                         tokenizer: Model, language_model: Model, 
                         device: str = 'cuda:0', percentage: float = 0.25) -> str:
    """
    Function that summarizes a list of arguments or an aggregated text from 
    a list of arguments, using the hybrid (reduce then summarize) strategy 
    by combining extractive and abstractive summarization. This function 
    takes as input the nlp, tokenizer and pre-trained language models.
    
    Parameters
    -----------
    arguments: text of arguments (List [str] / str).
    nlp: spaCy nlp model (Model).
    tokenizer: huggingface tokenizer model (Model).
    language model: huggingface language model (Model).
    device: device to load and run model ['cuda', 'cuda:0', 'cpu'] (str).
    percentage: the percentage of extracted sentences as a fraction of the original text (float).

    Returns
    --------
    summary: abstractive summary of arguments (str)
    """
    
    # Aggregate the arguments.
    text = aggregate_text(arguments)
    
    # if the aggregated text is empty, return early.
    if text == '':
        return ''

    # Extract the max supported length from the model config.
    max_length = language_model.config.max_position_embeddings

    # Tokenize the text to find the number of tokens.
    number_of_tokens = len(tokenizer.tokenize(text))

    # Check to see if the text surpasses the max length of the model.
    if number_of_tokens < max_length:
        summary = abstractive_summarization(text, tokenizer, language_model, device)
    else: # If it does, reduce the text by selecting its top-n sentences.
        # The constant factor takes into account the mismatch between 
        # the number of subword tokens and word-level tokens.
        reduced_text = extractive_summarization(text, nlp, percentage_sentences = percentage)
        summary = abstractive_summarization(reduced_text, tokenizer, language_model, device) 

    return summary


def extract_keyphrases(texts, model, top_n = 10, deduplicate = True):
    
    # Aggregate the texts.
    text = aggregate_text(texts)

    # Extract the top_n keyphrases from the model.
    keyphrases = [
        keyphrase
        for [keyphrase, _]
        in model.extract_keyphrases(text, language_code = 'en', top_n = top_n, deduplicate = deduplicate)
    ]

    return keyphrases