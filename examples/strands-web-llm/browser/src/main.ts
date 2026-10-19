import { examples } from './examples/catalog.js'

const exampleSelect = document.querySelector<HTMLSelectElement>('#example')!
const engineSelect = document.querySelector<HTMLSelectElement>('#engine')!
const modelSelect = document.querySelector<HTMLSelectElement>('#model')!
const cacheSelect = document.querySelector<HTMLSelectElement>('#cache')!
const loadButton = document.querySelector<HTMLButtonElement>('#load')!
const unloadButton = document.querySelector<HTMLButtonElement>('#unload')!
const runButton = document.querySelector<HTMLButtonElement>('#run')!
const stopButton = document.querySelector<HTMLButtonElement>('#stop')!
const cacheButton = document.querySelector<HTMLButtonElement>('#check-cache')!
const deleteButton = document.querySelector<HTMLButtonElement>('#delete-cache')!
const prompt = document.querySelector<HTMLTextAreaElement>('#prompt')!
const file = document.querySelector<HTMLInputElement>('#file')!
const status = document.querySelector<HTMLParagraphElement>('#status')!
const progress = document.querySelector<HTMLProgressElement>('#progress')!
const output = document.querySelector<HTMLPreElement>('#output')!
const details = document.querySelector<HTMLPreElement>('#details')!
const reports: unknown[] = []
let loaded: Awaited<ReturnType<typeof import('./engine/load.js').openEngine>> | undefined
let controller: AbortController | undefined
let busy = false

if (globalThis.chrome?.runtime?.id)
  engineSelect.add(new Option('Chrome extension worker', 'extension'))
cacheSelect.add(new Option('Cross-origin storage (Main thread only)', 'cross-origin'))

for (const example of examples) exampleSelect.add(new Option(example.name, example.id))
const modelResponse = await fetch('/models.json')
if (!modelResponse.ok) throw new Error('Could not load WebLLM models.')
const modelIds = (await modelResponse.json()) as string[]
for (const modelId of modelIds) modelSelect.add(new Option(modelId, modelId))

function selectedExample() {
  return examples.find((example) => example.id === exampleSelect.value)!
}

function updateButtons() {
  for (const select of [exampleSelect, engineSelect, modelSelect, cacheSelect]) {
    select.disabled = busy || Boolean(loaded)
  }
  loadButton.disabled = busy || Boolean(loaded)
  unloadButton.disabled = busy || !loaded
  runButton.disabled = busy || !loaded
  stopButton.disabled = !controller
  cacheButton.disabled = busy
  deleteButton.disabled = busy || Boolean(loaded) || cacheSelect.value === 'cross-origin'
  file.disabled = busy
  prompt.disabled = busy
}

function showError(error: unknown) {
  status.textContent = error instanceof Error ? error.message : String(error)
}

function info(value: unknown) {
  reports.push(value)
  details.textContent = JSON.stringify(reports, null, 2)
}

exampleSelect.onchange = () => {
  const example = selectedExample()
  modelSelect.value = example.model
  prompt.value = example.prompt
  file.value = ''
  file.accept = example.accept ?? ''
  file.hidden = !example.accept
  document.querySelector<HTMLLabelElement>('#file-label')!.hidden = !example.accept
}
function updateEngineOptions() {
  const mainOnly = cacheSelect.value === 'cross-origin'
  for (const option of engineSelect.options) option.disabled = mainOnly && option.value !== 'main'
  if (mainOnly) engineSelect.value = 'main'
}

cacheSelect.onchange = () => {
  updateEngineOptions()
  updateButtons()
}
exampleSelect.dispatchEvent(new Event('change'))

loadButton.onclick = async () => {
  busy = true
  updateButtons()
  status.textContent = 'Loading model…'
  details.textContent = ''
  reports.length = 0
  try {
    const { openEngine, cacheConfig } = await import('./engine/load.js')
    const example = selectedExample()
    const appConfig = await cacheConfig(cacheSelect.value)
    loaded = await openEngine(
      engineSelect.value,
      modelSelect.value,
      example.contextWindow ?? 2048,
      appConfig,
      (report) => {
        progress.value = report.progress
        status.textContent = report.text
      },
    )
    info({
      gpu: await loaded.engine.getGPUVendor(),
      maxBufferBytes: await loaded.engine.getMaxStorageBufferBindingSize(),
    })
    status.textContent = 'Model loaded'
  } catch (error) {
    showError(error)
  } finally {
    busy = false
    updateButtons()
  }
}

runButton.onclick = async () => {
  if (!loaded || !prompt.value.trim()) return
  const { engine } = loaded
  const example = selectedExample()
  controller = new AbortController()
  const { signal } = controller
  const stop = () => engine.interruptGenerate()
  if (example.native) signal.addEventListener('abort', stop, { once: true })
  busy = true
  updateButtons()
  output.textContent = ''
  details.textContent = ''
  reports.length = 0
  status.textContent = 'Running…'
  try {
    await example.run({
      engine,
      modelId: modelSelect.value,
      prompt: prompt.value,
      file: file.files?.[0],
      signal,
      write: (text) => {
        output.textContent += text
      },
      info,
    })
    status.textContent = signal.aborted ? 'Stopped' : 'Done'
  } catch (error) {
    if (signal.aborted) status.textContent = 'Stopped'
    else showError(error)
  } finally {
    signal.removeEventListener('abort', stop)
    controller = undefined
    busy = false
    updateButtons()
  }
}

stopButton.onclick = () => controller?.abort()

unloadButton.onclick = async () => {
  if (!loaded) return
  busy = true
  updateButtons()
  try {
    await loaded.close()
    status.textContent = 'Model unloaded'
  } catch (error) {
    showError(error)
  } finally {
    loaded = undefined
    progress.value = 0
    busy = false
    updateButtons()
  }
}

cacheButton.onclick = async () => {
  try {
    const { cacheConfig } = await import('./engine/load.js')
    const { hasModelInCache } = await import('@mlc-ai/web-llm')
    const appConfig = await cacheConfig(cacheSelect.value)
    info({ cached: await hasModelInCache(modelSelect.value, appConfig) })
  } catch (error) {
    showError(error)
  }
}

deleteButton.onclick = async () => {
  busy = true
  updateButtons()
  try {
    const { cacheConfig } = await import('./engine/load.js')
    const { deleteModelAllInfoInCache } = await import('@mlc-ai/web-llm')
    const appConfig = await cacheConfig(cacheSelect.value)
    await deleteModelAllInfoInCache(modelSelect.value, appConfig)
    status.textContent = 'Cached model removed'
  } catch (error) {
    showError(error)
  } finally {
    busy = false
    updateButtons()
  }
}
