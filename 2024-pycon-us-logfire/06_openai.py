import webbrowser
import openai
import logfire

logfire.configure()
client = openai.Client()
logfire.instrument_openai(client)

with logfire.span('Picture of a cat in the style of a famous painter'):
    response = client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role': 'system', 'content': 'Response entirely in plain text, with just a name'},
            {'role': 'user', 'content': 'Who was the influential painter in the 20th century?'},
        ],
    )
    chat_response = response.choices[0].message.content
    print(chat_response)

    response = client.images.generate(
        prompt=f'Create an image of a cat in the style of {chat_response}',
        model='dall-e-3',
    )
    url = response.data[0].url
    print(url)
webbrowser.open(url)
