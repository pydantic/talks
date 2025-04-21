from datetime import datetime
from pydantic import BaseModel
import logfire

logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))


class Delivery(BaseModel):
    timestamp: datetime
    dims: tuple[int, int]


input_json = [
    '{"timestamp": "2020-01-02T03:04:05Z", "dims": ["10", "20"]}',
    '{"timestamp": "2020-01-02T04:04:05Z", "dims": ["15", "25"]}',
    '{"timestamp": "2020-01-02T05:04:05Z", "dims": ["20", "30"]}',
]
deliveries = [Delivery.model_validate_json(json) for json in input_json]
Delivery.model_validate_json('{"timestamp": "2020-01-02T03:04:05Z", "dims": ["10"]}')
