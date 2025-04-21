from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
import logfire


if TYPE_CHECKING:
    from .server import EmailInfo


@dataclass
class EmailOk:
    reason: str
    status: Literal['ok'] = 'ok'


@dataclass
class EmailReply:
    text: str
    reason: str
    status: Literal['reply'] = 'reply'


@dataclass
class EmailDrop:
    reason: str
    status: Literal['drop'] = 'drop'


@logfire.instrument()
async def analyse_email(email: EmailInfo) -> EmailOk | EmailReply | EmailDrop:
    if email.references:
        return EmailOk(reason='This email is a reply, let all replies through.')

    if 'spam' in email.subject.lower():
        return EmailReply(
            reason='looks like your email is spam.',
            text='Please stop sending spam.'
        )
    else:
        return EmailOk(reason='This email is not spam.')
