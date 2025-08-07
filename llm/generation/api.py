"""
Code partially derived from https://github.com/ZubinGou/multi-view-prompting.git
"""
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def llm_chat(prompt, model, stop=None):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=300,
        stop="\n\n"
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content
    return result

def llm_chat_finetuned(text, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You extract aspect term, aspect category and sentiment polarity."},
            {"role": "user", "content": f"Text: {text}"}
        ],
        temperature=0,
        max_tokens=300,
        stop=["\n\n"]
    )

    return response.choices[0].message.content.strip()