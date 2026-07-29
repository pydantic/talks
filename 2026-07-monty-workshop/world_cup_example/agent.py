from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode
from pydantic_monty import MountDir

from .charts import draw_chart
from .database import Database

# from .hack_websocket_code_mode import WebsocketCodeMode

db = Database()

THIS_DIR = Path(__file__).parent
INSTRUCTIONS = (THIS_DIR / 'instructions.md').read_text()

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    name='monty_world_cup_agent',
    instructions=INSTRUCTIONS,
    retries=3,
    tools=[
        db.query,
        db.list_tables,
        db.describe_table,
        db.table_count,
        draw_chart,
    ],
    model_settings={'max_tokens': 16384},
    capabilities=[
        CodeMode(
            mount=MountDir(
                host_path=THIS_DIR / '..' / 'agent_output',
                virtual_path='/output',
                mode='read-write',
            )
        )
        # WebsocketCodeMode(
        #     url='ws://localhost:8000',
        #     mount=MountDir(
        #         host_path=THIS_DIR / '..' / 'agent_output',
        #         virtual_path='/output',
        #         mode='read-write',
        #     ),
        # )
    ],
)
