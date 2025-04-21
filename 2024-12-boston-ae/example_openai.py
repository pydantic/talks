from datetime import date
from pydantic import BaseModel
from openai import OpenAI


class User(BaseModel):
    """Definition of a user"""
    id: int
    name: str
    dob: date


response = OpenAI().chat.completions.create(
    model='gpt-4o',
    messages=[
        {'role': 'system', 'content': 'Extract information about the user'},
        {'role': 'user', 'content': 'The user with ID 123 is called Samuel, born on Jan 28th 87'}
    ],
    tools=[
        {
            'function': {
                'name': User.__name__,
                'description': User.__doc__,
                'parameters': User.model_json_schema(),
            },
            'type': 'function'
        }
    ]
)
user = User.model_validate_json(response.choices[0].message.tool_calls[0].function.arguments)
print(repr(user))
