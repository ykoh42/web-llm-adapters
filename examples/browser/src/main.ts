import { CreateMLCEngine, type MLCEngineInterface } from '@mlc-ai/web-llm'
import { Agent } from '@strands-agents/sdk'
import { WebLLMModel } from 'strands-webllm'

const modelSelect = document.querySelector<HTMLSelectElement>('#model')!
const loadButton = document.querySelector<HTMLButtonElement>('#load')!
const unloadButton = document.querySelector<HTMLButtonElement>('#unload')!
const runButton = document.querySelector<HTMLButtonElement>('#run')!
const stopButton = document.querySelector<HTMLButtonElement>('#stop')!
const prompt = document.querySelector<HTMLTextAreaElement>('#prompt')!
const status = document.querySelector<HTMLParagraphElement>('#status')!
const progress = document.querySelector<HTMLProgressElement>('#progress')!
const output = document.querySelector<HTMLPreElement>('#output')!
let engine: MLCEngineInterface | undefined
let model: WebLLMModel | undefined
let agent: Agent | undefined

loadButton.onclick = async () => {
  loadButton.disabled = true
  modelSelect.disabled = true
  status.textContent = 'Loading model…'
  try {
    engine = await CreateMLCEngine(
      modelSelect.value,
      {
        initProgressCallback: (report) => {
          progress.value = report.progress
          status.textContent = report.text
        },
      },
      { context_window_size: 2048 },
    )
    model = new WebLLMModel({
      engine,
      modelId: modelSelect.value,
      contextWindowLimit: 2048,
      maxTokens: 256,
      temperature: 0,
    })
    runButton.disabled = false
    unloadButton.disabled = false
    status.textContent = 'Model loaded'
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error)
    loadButton.disabled = false
    modelSelect.disabled = false
  }
}

runButton.onclick = async () => {
  if (!model || !prompt.value.trim()) return
  agent = new Agent({ model })
  runButton.disabled = true
  unloadButton.disabled = true
  stopButton.disabled = false
  output.textContent = ''
  status.textContent = 'Running…'
  try {
    for await (const event of agent.stream(prompt.value)) {
      if (event.type !== 'modelStreamUpdateEvent') continue
      const part = event.event
      if (part.type === 'modelContentBlockDeltaEvent' && part.delta.type === 'textDelta') {
        output.textContent += part.delta.text
      }
    }
    status.textContent = agent.cancelSignal.aborted ? 'Stopped' : 'Done'
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error)
  } finally {
    runButton.disabled = false
    unloadButton.disabled = false
    stopButton.disabled = true
    agent = undefined
  }
}

stopButton.onclick = () => agent?.cancel()

unloadButton.onclick = async () => {
  if (!engine) return
  unloadButton.disabled = true
  runButton.disabled = true
  try {
    await engine.unload()
    engine = undefined
    model = undefined
    loadButton.disabled = false
    modelSelect.disabled = false
    progress.value = 0
    status.textContent = 'Model unloaded'
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error)
    unloadButton.disabled = false
  }
}
