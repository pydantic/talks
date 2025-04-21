from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Mapping, Any, Literal

from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route
from starlette.requests import Request
from email import message_from_bytes
from email.utils import parsedate_to_datetime
from httpx import AsyncClient

from .send_reply import send_reply
from .analyse import analyse_email

import logfire

logfire.configure(scrubbing=False)
logfire.info('running server')


@dataclass
class EmailInfo:
    from_: str
    subject: str
    to: str
    message_id: str
    references: str | None
    timestamp: datetime
    text: str | None
    html: str | None


class AnalysisResponse(BaseModel):
    status: Literal['ok', 'reply', 'drop']


@asynccontextmanager
async def lifespan(_app: Starlette) -> AsyncIterator[Mapping[str, Any]]:
    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_all=True)
        yield {'httpx_client': client}


async def analyze_email(request: Request):
    body = await request.body()
    msg = message_from_bytes(body)
    logfire.info('{body=}', body=body.decode(errors='ignore'), smtp_headers=dict(msg.items()))
    text_body = None
    html_body = None
    for part in msg.walk():
        content_type = part.get_content_type()

        charset = part.get_content_charset() or 'utf-8'
        if content_type == 'text/plain':
            text_body = part.get_payload(decode=True).decode(charset)
        elif content_type == 'text/html':
            html_body = part.get_payload(decode=True).decode(charset)

    date = msg['Date']
    if date:
        timestamp = parsedate_to_datetime(str(date))
    else:
        timestamp = datetime.now()

    email = EmailInfo(
        from_=str(msg['from']),
        subject=str(msg['subject']),
        to=str(msg['to']),
        message_id=str(msg['Message-ID']),
        references=str(msg['References']) if 'References' in msg else None,
        timestamp=timestamp,
        text=text_body,
        html=html_body,
    )

    email_analysis = await analyse_email(email)
    logfire.info(f'{email_analysis=}')
    if email_analysis.status == 'reply':
        client: AsyncClient = request.state.httpx_client
        await send_reply(client, email, email_analysis)

    response = AnalysisResponse(status=email_analysis.status)
    return Response(response.model_dump_json(), headers={'content-type': 'application/json'})


app = Starlette(routes=[Route('/', analyze_email, methods=['post'])], lifespan=lifespan)
logfire.instrument_starlette(app, capture_headers=True)
