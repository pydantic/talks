// Browser tracing and session replay for the demo page, sent to Logfire.
// Built into ../static/vendor/logfire/ by `make browser-telemetry`; the page loads the built copy.
import * as logfire from '@pydantic/logfire-browser'
import { sessionReplayIntegration } from '@pydantic/logfire-session-replay/integration'

const { logfire_frontend: frontend } = await fetch('/api/config').then(r => r.json())

// Off unless the server has a frontend application token (LOGFIRE_FRONTEND_TOKEN).
if (frontend) {
  logfire.configureFrontend({
    baseUrl: frontend.base_url,
    token: frontend.token,
    sessionReplay: sessionReplayIntegration({
      // Record every session: this is a single-user demo.
      sessionSampleRate: 1,
      onErrorSampleRate: 1,
      captureConsole: true,
      // Replay masks text, inputs and images by default. Everything on this page is invented or
      // public data, and a masked replay shows nothing worth watching.
      maskAllText: false,
      maskAllInputs: false,
      maskAllImages: false,
    }),
  })
}
