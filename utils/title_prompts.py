gpt_prompt = '''We have a subject in which the citizens are provide their feedback.
The feedbacks are categorized in {num_topics} topics based on specific keyphrases.
Your goal is to find representative and generic titles that effectively combines the subject and each topic with its summary and keyphrases.

Subject:
{subject}

Keyphrases per Topic:
{keyphrases_text}

Summaries per Topic:
{summaries_text}

Desired Answer Format:
A valid dictionary with each number of cluster and its title.

Instructions:
- Generate distinct titles between topics.

Answer:'''