from datetime import date
from pydantic import BaseModel

from devtools import debug


class User(BaseModel):
    id: int
    name: str
    dob: date


user = User(id='123', name='Samuel Colvin', dob='198-1-28')
debug(user)

# debug(User.model_json_schema())
