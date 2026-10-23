import type { ChatCompletionRequestStreaming, CompletionUsage } from '@mlc-ai/web-llm'

export async function vision() {
  const [{ CreateWebWorkerMLCEngine }, { createModels }, { createWebLLMProvider }, { Agent }] =
    await Promise.all([
      import('@mlc-ai/web-llm'),
      import('@earendil-works/pi-ai/models'),
      import('pi-ai-web-llm'),
      import('@earendil-works/pi-agent-core'),
    ])
  const modelId = 'Phi-3.5-vision-instruct-q4f16_1-MLC'
  const prompt = 'What color is this image? Answer with the color.'
  const request = { max_tokens: 4096, temperature: 0, seed: 7 }
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 224
  const drawing = canvas.getContext('2d')!
  drawing.fillStyle = 'red'
  drawing.fillRect(0, 0, 224, 224)
  const imageUrl = canvas.toDataURL('image/png')
  const image = { type: 'image' as const, mimeType: 'image/png', data: imageUrl.split(',')[1]! }
  const context = (data: string) => ({
    messages: [
      {
        role: 'user' as const,
        timestamp: 0,
        content: [
          { type: 'text' as const, text: prompt },
          { ...image, data },
        ],
      },
    ],
  })
  const worker = new Worker(new URL('../src/worker.ts', import.meta.url), { type: 'module' })
  try {
    const engine = await CreateWebWorkerMLCEngine(worker, modelId, undefined, {
      context_window_size: 8192,
    })
    try {
      const create = engine.chat.completions.create.bind(engine.chat.completions)
      let latest: ChatCompletionRequestStreaming | undefined
      engine.chat.completions.create = new Proxy(engine.chat.completions.create, {
        apply(target, self, args) {
          latest = structuredClone(args[0])
          return Reflect.apply(target, self, args)
        },
      })
      async function native(input: ChatCompletionRequestStreaming) {
        await engine.resetChat()
        let text = ''
        let usage: CompletionUsage | undefined
        for await (const chunk of await create(input)) {
          text += chunk.choices[0]?.delta.content ?? ''
          usage = chunk.usage ?? usage
        }
        return { text, usage }
      }
      const baseline = await native({
        ...request,
        model: modelId,
        n: 1,
        stream: true,
        stream_options: { include_usage: true },
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: imageUrl } },
            ],
          },
        ],
      })
      const models = createModels()
      let raw = ''
      const provider = createWebLLMProvider({
        engine,
        modelId,
        contextWindow: 8192,
        imageInputs: true,
        request,
        onChunk: (chunk) => {
          const choice = chunk.choices[0]
          if (choice && 'delta' in choice) raw += choice.delta.content ?? ''
        },
      })
      models.setProvider(provider)
      const model = provider.getModels()[0]!
      await engine.resetChat()
      const failed = await models.complete(model, context('AA=='))
      raw = ''
      const response = await models.complete(model, context(image.data))
      const responseRaw = raw
      await engine.resetChat()
      raw = ''
      const agent = new Agent({
        initialState: { model },
        streamFn: models.streamSimple.bind(models),
      })
      await agent.prompt(prompt, [image])
      const answer = agent.state.messages.filter((message) => message.role === 'assistant').at(-1)
      if (!answer || !latest) throw new Error('Pi did not return an assistant response.')
      const agentRequest = latest
      const agentRaw = raw
      const agentNative = await native(agentRequest)
      return {
        imageUrl,
        baseline,
        failed,
        response,
        responseRaw,
        answer,
        agentRaw,
        agentNative,
        agentRequest,
      }
    } finally {
      await engine.unload()
    }
  } finally {
    worker.terminate()
  }
}
