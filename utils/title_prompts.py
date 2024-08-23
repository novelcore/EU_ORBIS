gpt_prompt = '''We have a subject in which the citizens are provide their feedback.
The feedbacks are categorized in {num_topics} topics based on specific summaries.
Your goal is to find representative and generic titles that effectively combines the subject and each topic with its summary..

Subject:
{subject}

Summaries per Topic:
{summaries_text}

Desired Answer Format:
A valid dictionary with each number of cluster and its title.

Instructions:
- Generate distinct titles between topics.

Answer:'''