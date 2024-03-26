import openai
import os




def gpt_generate(prompt: str,
                 client: openai.OpenAI,):
    message = [
        {'role': 'user', 'content': prompt},
    ]

    response = client.chat.completions.create(
        messages = message,
        model = "gpt-3.5-turbo"
    )

    answer = response.choices[0].message.content.strip()
    answer = answer.split("\n")
    processed_answer = []
    return processed_answer