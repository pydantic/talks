from typing import Annotated

from fastapi import APIRouter, Request
from fastui import FastUI, AnyComponent, components as c, events
from fastui.components.display import DisplayMode, DisplayLookup
from fastui.events import GoToEvent
from fastui.forms import fastui_form
from pydantic import BaseModel

from .db import Database
from .page import demo_page
from . import images


router = APIRouter()


class AnimalModel(BaseModel):
    animal: str


@router.get('/', response_model=FastUI, response_model_exclude_none=True)
async def generate_image() -> list[AnyComponent]:
    return demo_page(
        c.Heading(text='Generate Image', level=2),
        c.Paragraph(text='Generate an image of an animal in the style of Francis Bacon.'),
        c.ModelForm(
            model=AnimalModel,
            display_mode='page',
            submit_url='/api/generate/',
            loading=[c.Spinner(text='Generating Image...')],
        ),
    )


@router.post('/generate/', response_model=FastUI, response_model_exclude_none=True)
async def login_form_post(form: Annotated[AnimalModel, fastui_form(AnimalModel)], db: Database, request: Request):
    async with db.acquire() as conn:
        image_id = await images.create_image(conn, request.app.state.openai, form.animal)
    return [c.FireEvent(event=GoToEvent(url=f'/images/{image_id}/'))]


@router.get('/images/', response_model=FastUI, response_model_exclude_none=True)
async def images_table_view(db: Database, page: int = 1) -> list[AnyComponent]:
    async with db.acquire() as conn:
        image_list, count = await images.list_images(conn, page)

    return demo_page(
        c.Heading(text='List of Images', level=2),
        c.Table(
            data=image_list,
            data_model=images.Image,
            columns=[
                DisplayLookup(field='prompt', on_click=GoToEvent(url='/images/{id}/')),
                DisplayLookup(field='ts', mode=DisplayMode.datetime),
            ],
        ),
        c.Pagination(page=page, page_size=images.PAGE_LIMIT, total=count)
    )


@router.get('/images/{image_id:int}/', response_model=FastUI, response_model_exclude_none=True)
async def image_view(db: Database, image_id: int) -> list[AnyComponent]:
    async with db.acquire() as conn:
        image = await images.get_image(conn, image_id)

    return demo_page(
        c.Link(components=[c.Text(text='Back')], on_click=events.BackEvent()),
        c.Details(
            data=image,
            fields=[
                DisplayLookup(field='id'),
                DisplayLookup(field='ts', mode=DisplayMode.datetime),
                DisplayLookup(field='prompt'),
            ]
        ),
        c.Image(src=image.url, alt=image.prompt, width=600, height=600),
        title=image.prompt,
    )


@router.get('/{path:path}', status_code=404)
async def api_404():
    # so we don't fall through to the index page
    return {'message': 'Not Found'}
