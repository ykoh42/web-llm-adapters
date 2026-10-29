import type { Agent, InvokeArgs } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'

export interface ExampleInput {
  engine: MLCEngineInterface
  modelId: string
  prompt: string
  file: File | undefined
  signal: AbortSignal
  write: (text: string) => void
  info: (value: unknown) => void
}

export async function runAgent(agent: Agent, prompt: InvokeArgs, input: ExampleInput) {
  let text = ''
  for await (const update of agent.stream(prompt, {
    cancelSignal: input.signal,
    limits: { turns: 3 },
  })) {
    if (update.type === 'toolResultEvent') input.info(update)
    if (update.type === 'agentResultEvent') reportReasoning(agent, input)
    // Hooks can end a turn with a final message that has no streamed deltas.
    if (update.type === 'agentResultEvent' && !text) {
      text = update.result.lastMessage.content
        .filter((block) => block.type === 'textBlock')
        .map((block) => block.text)
        .join('\n')
      input.write(text)
    }
    if (update.type !== 'modelStreamUpdateEvent') continue
    const event = update.event
    if (event.type === 'modelMetadataEvent') input.info(event)
    if (event.type !== 'modelContentBlockDeltaEvent' || event.delta.type !== 'textDelta') continue
    text += event.delta.text
    input.write(event.delta.text)
  }
  return text
}

function reportReasoning(agent: Agent, input: ExampleInput): void {
  const reasoning = agent.messages
    .flatMap((message) => message.content)
    .filter((block) => block.type === 'reasoningBlock')
    .map((block) => block.text ?? '')
  if (reasoning.length) input.info({ reasoning })
}
