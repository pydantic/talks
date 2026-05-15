from dataclasses import dataclass
from datetime import date, datetime, timedelta

import logfire
from devtools import debug
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

logfire.configure(service_name='calendar-agent')
logfire.instrument_pydantic_ai()

calendar_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-5',
    capabilities=[CodeMode()],
    instructions='Combine tool calls into a single code execution call.',
)


@calendar_agent.instructions
def get_current_date() -> str:
    """Get the current date."""
    return str(date.today())


@dataclass
class Appointment:
    id: str
    start: datetime
    duration: timedelta
    title: str
    location: str | None = None


next_apt_id = 0
appointments: dict[str, Appointment] = {}


@calendar_agent.tool_plain
def get_appointments(range_state: datetime, range_end: datetime) -> list[Appointment]:
    """Get appointments between `range_state` and `range_end`."""
    return [a for a in appointments.values() if a.start + a.duration > range_state and a.start < range_end]


@calendar_agent.tool_plain
def create_appointment(
    start: datetime,
    duration: timedelta,
    title: str,
    location: str | None = None,
) -> str:
    """Create a calendar appointment.

    Returns: The ID of the created appointment.
    """
    global next_apt_id
    next_apt_id += 1
    apt_id = f'appt-{next_apt_id}'
    appointments[apt_id] = Appointment(id=apt_id, start=start, duration=duration, title=title, location=location)
    return apt_id


@calendar_agent.tool_plain
def edit_appointment(
    id: str,
    start: datetime | None = None,
    duration: timedelta | None = None,
    title: str | None = None,
    location: str | None = None,
) -> str:
    """Edit an existing appointment. Only provided fields are updated.

    Returns: The ID of the edited appointment.
    """
    try:
        appt = appointments[id]
    except KeyError:
        return f'Appointment {id} not found'

    if start is not None:
        appt.start = start
    if duration is not None:
        appt.duration = duration
    if title is not None:
        appt.title = title
    if location is not None:
        appt.location = location
    return f'Appointment {id} updated'


if __name__ == '__main__':
    result = calendar_agent.run_sync('Create an appointment "brush my teeth" for 8am every week day next month.')
    print(result.output)
    debug(appointments)
else:
    app = calendar_agent.to_web()
