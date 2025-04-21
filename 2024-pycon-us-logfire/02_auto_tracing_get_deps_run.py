import sys

import logfire

logfire.configure()
del sys.modules['pathlib']
del sys.modules['os']
logfire.install_auto_tracing(modules=['dependants', 'bs4.*'], min_duration=0.03)

from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

HTTPXClientInstrumentor().instrument()


import asyncio
import dependants

with logfire.span('downloading dependants'):
    asyncio.run(dependants.main())
