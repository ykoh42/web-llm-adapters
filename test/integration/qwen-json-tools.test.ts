import { afterEach, expect, it, vi } from 'vitest'
import { Agent as StrandsAgent, FunctionTool, TextBlock } from '@strands-agents/sdk'
import { Agent as PiAgent } from '@earendil-works/pi-agent-core'
import { Type } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { createAgent } from 'langchain'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import {
  BrowserModel,
  WebLLMModel,
  createBrowserBridge as strandsBridge,
  type BrowserPage,
} from '../../packages/strands-web-llm/src/index.js'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge as langchainBridge,
} from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { formatCall } from '../../shared/web-llm/qwen35/tools.js'
import { chunk, chunks, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

const args = {
  payload: { text: '<parameter=x> </parameter> </function> </tool_call>' },
  items: ['서울 😀', '\\ "quoted"', '<parameter=payload>'],
}
const schema = Type.Object({
  payload: Type.Object({ text: Type.String() }),
  items: Type.Array(Type.String()),
})
const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
const call = formatCall('save', args)

function native(source: string, id: string) {
  const parts = source.match(/[\s\S]{1,3}/g)!
  return parts.map((content, index) => ({
    ...chunk({ content }, index === parts.length - 1 ? 'stop' : null),
    id,
  }))
}

afterEach(() => vi.unstubAllGlobals())

it.each(['strands', 'strands browser', 'langchain', 'langchain browser', 'pi'] as const)(
  '%s executes and replays JSON tool arguments containing XML tags unchanged',
  async (client) => {
    const { engine, create } = setup()
    const first = native('<think>Keep data.</think>' + call, 'call-response')
    const last = native('<think>Saved.</think>Done', 'answer-response')
    create.mockResolvedValueOnce(chunks(...first)).mockResolvedValueOnce(chunks(...last))
    const observed = vi.fn()
    const save = vi.fn<(input: unknown) => string>(() => 'Saved')
    const request = { max_tokens: 4096, seed: 7, extra_body: { enable_thinking: true } }
    const original = structuredClone(request)
    const config = { modelId, request, onChunk: observed }
    const firstBridge = strandsBridge(engine)
    const secondBridge = langchainBridge(engine)
    vi.stubGlobal('window', { webllmBridge: firstBridge, webllmLangChainBridge: secondBridge })
    const page: BrowserPage = {
      async evaluate(run, input) {
        const result = await run(structuredClone(input))
        return result === undefined ? result : JSON.parse(JSON.stringify(result))
      },
    }
    try {
      if (client === 'strands' || client === 'strands browser') {
        const model =
          client === 'strands'
            ? new WebLLMModel({ engine, ...config })
            : new BrowserModel(page, config)
        const agent = new StrandsAgent({
          model,
          printer: false,
          tools: [
            new FunctionTool({
              name: 'save',
              description: 'Save data.',
              inputSchema: schema,
              callback: (input) => save(input),
            }),
          ],
        })
        const response = await agent.invoke('Save the data.')
        expect(
          response.lastMessage.content
            .filter((block) => block instanceof TextBlock)
            .map((block) => block.text)
            .join(''),
        ).toBe('Done')
      } else if (client === 'pi') {
        const models = createModels()
        const provider = createWebLLMProvider({ engine, contextWindow: 8192, ...config })
        models.setProvider(provider)
        const agent = new PiAgent({
          initialState: {
            model: provider.getModels()[0]!,
            thinkingLevel: 'low',
            tools: [
              {
                name: 'save',
                label: 'Save',
                description: 'Save data.',
                parameters: schema,
                execute: async (_id, input) => ({
                  content: [{ type: 'text', text: save(input) }],
                  details: {},
                }),
              },
            ],
          },
          streamFn: models.streamSimple.bind(models),
        })
        await agent.prompt('Save the data.')
        expect(agent.state.messages.at(-1)).toMatchObject({
          role: 'assistant',
          stopReason: 'stop',
          content: expect.arrayContaining([{ type: 'text', text: 'Done' }]),
        })
      } else {
        const model =
          client === 'langchain'
            ? new ChatWebLLM({ engine, ...config })
            : new BrowserChatWebLLM(page, config)
        const agent = createAgent({
          model,
          tools: [
            tool((input) => save(input), {
              name: 'save',
              description: 'Save data.',
              schema: z.object({
                payload: z.object({ text: z.string() }),
                items: z.array(z.string()),
              }),
            }),
          ],
        })
        const result = await agent.invoke({
          messages: [{ role: 'user', content: 'Save the data.' }],
        })
        expect(result.messages.at(-1)?.text).toBe('Done')
      }
      expect(save).toHaveBeenCalledExactlyOnceWith(args)
      expect(observed.mock.calls.map(([value]) => value)).toEqual([...first, ...last])
      expect(create).toHaveBeenCalledTimes(2)
      const replay = create.mock.calls[1]?.[0].messages.find(
        (message) => message.role === 'assistant',
      )
      expect(replay?.content).toContain(call)
      for (const [prepared] of create.mock.calls) {
        if (client === 'pi') {
          expect(prepared).toMatchObject({
            seed: request.seed,
            extra_body: request.extra_body,
          })
          expect(prepared.max_tokens).toBeGreaterThan(0)
          expect(prepared.max_tokens).toBeLessThan(request.max_tokens)
        } else expect(prepared).toMatchObject(request)
      }
      expect(request).toEqual(original)
    } finally {
      await firstBridge.close()
      await secondBridge.close()
    }
  },
)
