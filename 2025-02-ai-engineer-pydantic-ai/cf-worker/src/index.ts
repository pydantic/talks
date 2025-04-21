interface Env {}

interface AnalysisResult {
  status: 'ok' | 'reply' | 'drop'
}

export default {
  async email(message, env, ctx) {
    const r = await fetch('https://samuelcolvin.eu.ngrok.io', {
      method: 'POST',
      body: message.raw,
    })
    const { status }: AnalysisResult = await r.json()
    if (status == 'ok') {
      console.log('forwarding email')
      await message.forward('samuel@pydantic.dev')
    }
  },
} satisfies ExportedHandler<Env>
