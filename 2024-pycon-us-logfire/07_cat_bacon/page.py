from __future__ import annotations as _annotations

from fastui import AnyComponent
from fastui import components as c
from fastui.events import GoToEvent


def demo_page(*components: AnyComponent, title: str | None = None) -> list[AnyComponent]:
    return [
        c.PageTitle(text=f'Cat Bacon — {title}' if title else 'Cat Bacon'),
        c.Navbar(
            title='Cat Bacon',
            title_event=GoToEvent(url='/'),
            start_links=[
                c.Link(
                    components=[c.Text(text='Previous Images')],
                    on_click=GoToEvent(url='/images/'),
                    active='startswith:/images',
                ),
            ]
        ),
        c.Page(
            components=[
                *((c.Heading(text=title),) if title else ()),
                *components,
            ],
        ),
        c.Footer(
            links=[
                c.Link(components=[c.Text(text='Logfire Docs')], on_click=GoToEvent(url='https://docs.logfire.dev')),
            ],
        ),
    ]
