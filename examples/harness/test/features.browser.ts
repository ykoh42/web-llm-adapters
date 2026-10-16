import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { createServer } from 'node:http'
import { join } from 'node:path'
import { FunctionTool } from '@strands-agents/sdk'
import { test, expect } from '../../../test/browser.js'
import { openHarness, useTool, finalText } from './setup.js'

test('Gemma activates a Harness skill and updates the default todo plugin', async ({ page }) => {
  const harness = await openHarness(page)
  try {
    const dir = join(harness.dir, 'skills', 'label')
    const code = 'skill-' + crypto.randomUUID().slice(0, 8)
    await mkdir(dir, { recursive: true })
    await writeFile(
      join(dir, 'SKILL.md'),
      `---\nname: label\ndescription: Look up the project label.\n---\nThe project label is ${code}.`,
    )
    const agent = await harness.create({ skills: join(harness.dir, 'skills') })
    useTool(agent, 'skills')
    const result = await agent.invoke(
      'Call skills with {"skill_name":"label"}. Tell me the project label in that skill.',
      { limits: { turns: 4 } },
    )
    expect(finalText(result)).toContain(code)
    useTool(agent, 'todo_write')
    await agent.invoke(
      'Call todo_write with {"todos":[{"content":"Read label","activeForm":"Reading label","status":"completed"}]}.',
      { limits: { turns: 4 } },
    )
    expect(agent.appState.get('todos')).toEqual([
      { content: 'Read label', activeForm: 'Reading label', status: 'completed' },
    ])
    const calls = agent.messages
      .flatMap((m) => m.content)
      .filter((b) => b.type === 'toolUseBlock')
      .map((b) => b.name)
    expect(calls).toContain('skills')
    expect(calls).toContain('todo_write')
    const prompts = await page.evaluate(() => window.gemmaPrompts)
    expect(prompts[0]).toContain('Working directory: ' + process.cwd())
    expect(prompts.at(-1)).toContain('Your current todo list:')
    expect(JSON.stringify(agent.messages)).not.toContain('<environment>')
  } finally {
    await harness.close()
  }
})

test('default file tools, shell and programmatic tool caller run in Node with Gemma', async ({
  page,
}) => {
  const harness = await openHarness(page)
  try {
    const agent = await harness.create()
    const path = join(harness.dir, 'note.txt')
    const code = 'file-' + Math.floor(Math.random() * 1_000_000)
    useTool(agent, 'write')
    await agent.invoke(`Use write to create ${path} with exactly "old" as its content.`, {
      limits: { turns: 3 },
    })
    expect(await readFile(path, 'utf8')).toBe('old')
    useTool(agent, 'edit')
    await agent.invoke(`Use edit on ${path}, old_str "old", new_str "${code}".`, {
      limits: { turns: 3 },
    })
    expect(await readFile(path, 'utf8')).toBe(code)
    const shellCode = 'shell-' + Math.floor(Math.random() * 1_000_000)
    await writeFile(path, shellCode)
    useTool(agent, 'shell')
    const shell = await agent.invoke(`Use shell to run cat '${path}' and tell me its output.`, {
      limits: { turns: 3 },
    })
    expect(finalText(shell)).toContain(shellCode)
    const pythonCode = 'python-' + Math.floor(Math.random() * 1_000_000)
    await writeFile(path, pythonCode)
    useTool(agent, 'programmatic_tool_caller')
    const result = await agent.invoke(
      `The file has changed. Use programmatic_tool_caller to run exactly this Python code: print(await read(path='${path}'))\nTell me the file's content from the new result.`,
      { limits: { turns: 3 } },
    )
    const results = agent.messages
      .flatMap((message) => message.content)
      .filter((block) => block.type === 'toolResultBlock')
    expect(JSON.stringify(results)).toContain(pythonCode)
    expect((await page.evaluate(() => window.gemmaPrompts)).at(-1)).toContain(pythonCode)
    expect(finalText(result)).toContain(pythonCode)
    const calls = agent.messages
      .flatMap((m) => m.content)
      .filter((b) => b.type === 'toolUseBlock')
      .map((b) => b.name)
    expect(calls).toEqual(
      expect.arrayContaining(['write', 'edit', 'shell', 'programmatic_tool_caller']),
    )
  } finally {
    await harness.close()
  }
})

test('web_fetch uses the same Gemma weights to answer over a real HTTP response', async ({
  page,
}) => {
  const harness = await openHarness(page)
  const code = 'page-' + crypto.randomUUID().slice(0, 8)
  const server = createServer((_, response) => response.end(`<p>The page code is ${code}.</p>`))
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  try {
    const address = server.address()
    if (!address || typeof address === 'string') throw new Error('Missing test server port')
    const agent = await harness.create()
    useTool(agent, 'web_fetch')
    const result = await agent.invoke(
      `Call web_fetch with url http://127.0.0.1:${address.port} and prompt "What is the page code?". Tell me that code.`,
      { limits: { turns: 4 } },
    )
    expect(finalText(result)).toContain(code)
    const blocks = agent.messages.flatMap((message) => message.content)
    const call = blocks.find((block) => block.type === 'toolUseBlock' && block.name === 'web_fetch')
    if (!call || call.type !== 'toolUseBlock') throw new Error('Gemma did not call web_fetch')
    const fetched = blocks.find(
      (block) => block.type === 'toolResultBlock' && block.toolUseId === call.toolUseId,
    )
    expect(fetched).toMatchObject({ type: 'toolResultBlock', status: 'success' })
    expect(JSON.stringify(fetched)).toContain(code)
  } finally {
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    )
    await harness.close()
  }
})

test('a default background subagent shares the Gemma engine and delivers its final answer', async ({
  page,
}) => {
  const harness = await openHarness(page)
  try {
    const childAnswers: string[] = []
    const stream = harness.model.stream.bind(harness.model)
    // The parent has a session; the default child does not. Observe actual child text.
    harness.model.stream = async function* (messages, options = {}) {
      let text = ''
      for await (const event of stream(messages, options)) {
        if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
          text += event.delta.text
        yield event
      }
      if (!options.agentMetadata) childAnswers.push(text)
    }
    const agent = await harness.create()
    useTool(agent, 'subagent')
    const result = await agent.invoke(
      'Ask subagent to name one fruit in one word. Keep your replies short. Repeat its fruit when the result arrives.',
      { limits: { turns: 8 } },
    )
    expect(result.stopReason).toBe('endTurn')
    expect(childAnswers).toHaveLength(1)
    const fruit = childAnswers[0]!
      .toLowerCase()
      .match(
        /\b(apple|banana|orange|pear|mango|grape|peach|lemon|cherry|kiwi|melon|berry|pineapple)\b/,
      )?.[0]
    expect(fruit).toBeDefined()
    expect(finalText(result).toLowerCase()).toContain(fruit!)
    const history = JSON.stringify(agent.messages)
    expect(history).toContain('Background task dispatched.')
    expect(history).toContain('strands_manage_background_task')
    const results = agent.messages
      .flatMap((m) => m.content)
      .filter((b) => b.type === 'toolResultBlock')
    expect(JSON.stringify(results).toLowerCase()).toContain(fruit!)
  } finally {
    await harness.close()
  }
})

test('a tool runs in the default background policy and returns data unknown to Gemma', async ({
  page,
}) => {
  const harness = await openHarness(page)
  try {
    const code = 'task-' + Math.floor(Math.random() * 1_000_000)
    let calls = 0
    const label = new FunctionTool({
      name: 'get_label',
      description: 'Get the project label.',
      inputSchema: { type: 'object', properties: {} },
      callback: async () => {
        calls++
        return code
      },
    })
    const agent = await harness.create({ tools: [label] })
    useTool(agent, 'get_label')
    const result = await agent.invoke(
      'Call get_label with _background_execution true. After it completes, return the actual task- code from the text field of the background result, including the full random suffix.',
      { limits: { turns: 6 } },
    )
    expect(calls).toBe(1)
    expect(JSON.stringify(agent.messages)).toContain(code)
    expect((await page.evaluate(() => window.gemmaPrompts)).at(-1)).toContain(code)
    expect(finalText(result)).toContain(code)
    expect(JSON.stringify(agent.messages)).toContain('Background task dispatched.')
    expect(JSON.stringify(agent.messages)).toContain('strands_manage_background_task')
  } finally {
    await harness.close()
  }
})
