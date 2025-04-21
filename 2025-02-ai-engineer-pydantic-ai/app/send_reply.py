from __future__ import annotations as _annotations
import os
from html import escape
from textwrap import indent
from typing import TYPE_CHECKING

from markdown import markdown
from httpx import AsyncClient
from aioaws.ses import SesConfig, SesClient, SesRecipient
import logfire

if TYPE_CHECKING:
    from .server import EmailInfo
    from .analyse import EmailReply

ses_config = SesConfig(os.environ['AWS_ACCESS_KEY'], os.environ['AWS_SECRET_KEY'], 'us-east-1')


@logfire.instrument
async def send_reply(client: AsyncClient, email: EmailInfo, email_reply: EmailReply) -> None:
    plain_text = email_reply.text

    summary = f'On {email.timestamp:%d %b %Y at %H:%M}, {email.from_} wrote'
    if email.text:
        plain_text += f'\n\n{summary}:\n{indent(email.text, "> ")}'

    html = f'<div>{markdown(email_reply.text)}</div>'
    if email.html:
        quote_body = email.html
    elif email.text:
        quote_body = escape(email.text).replace('\n', '<br>')
    else:
        quote_body = None

    if quote_body:
        quote_styles = 'margin:0px 0px 0px 0.8ex;border-left:1px solid rgb(204,204,204);padding-left:1ex'
        html += (
            f'<div dir="ltr" class="gmail_attr">{escape(summary)}:</div>'
            f'<br>'
            f'<blockquote class="gmail_quote" style="{quote_styles}">{quote_body}</blockquote>'
        )

    ses_client = SesClient(client, ses_config)

    if '@pydantic.io' in email.from_:
        logfire.warning('Not sending email to pydantic.io')
        return

    if email.subject.lower().startswith('re:'):
        subject = email.subject
    else:
        subject = f'Re: {email.subject}'

    message_id = await ses_client.send_email(
        e_from=SesRecipient('spiced-ham@pydantic.io', 'Samuel', 'Colvin (Spiced Ham)'),
        subject=subject,
        to=[email.from_],
        text_body=plain_text,
        html_body=html,
        configuration_set='spiced-ham',
        smtp_headers={
            'In-Reply-To': email.message_id,
            'References': f'{email.references} {email.message_id}' if email.references else email.message_id,
        },
    )
    logfire.info(f'email sent: {message_id=}')
