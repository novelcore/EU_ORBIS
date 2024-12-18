from openai import OpenAI

gpt_prompt_class = '''You are an AI expert in text classification.
There is a subject in which citizens share their opinion and provide feedback.

Subject:
"{subject}"

Feedback:
"{data_sample}"

Classify the feedback into 2 labels:
- "In favor"
- "Against"

Answer:'''

def generate_prompt_class(model_name: str, data_sample: str, subject: str) -> str:
    """
    Generates a prompt based on the specified model name and data sample.

    Parameters:
    - model_name (str): The name of the model to generate the prompt for.
    - data_sample (str): The sample text or document for the prompt.
    - subject (str): The main topic addressed by the data sample.

    Returns:
    str: The generated prompt.

    Raises:
    ValueError: If an invalid model name is provided.
    """

    if model_name.startswith('gpt'):
        return gpt_prompt_class.format(data_sample=data_sample, subject=subject)
    else:
        raise ValueError("Invalid model name. Supported models are 'gpt'.")
    
def get_label_from_document(model_name: str = 'gpt-4o-mini', data_sample: str = None, subject: str = None, temperature: float = 0) -> str:
    """
    Classify the data sample using LLM.

    Parameters:
    - model_name (str): The LLM model name used for the NER task.
    - data_sample (str): The text document to analyze.
    - subject (str): The summary of the subject.
    - temperature (float): A parameter controlling the randomness of entity extraction.

    Returns:
    str: Extracted label from the document.
    """

    # Check if the model is not ChatGPT-based
    if model_name.startswith("gpt"):
        
        prompt = generate_prompt_class(
            model_name=model_name,
            data_sample=data_sample,
            subject=subject
        )
        
        # Prepare messages for ChatGPT-based models
        messages = [
                { "role": "user", "content": prompt },
            ]

        client = OpenAI()
        response = client.chat.completions.create(
                                model=model_name,
                                messages=messages,
                                temperature=temperature,
                            )

        # Extract content from the response
        response = response.choices[0].message.content

    return response