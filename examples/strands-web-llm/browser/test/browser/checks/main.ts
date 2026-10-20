const checks = {
  embeddings: async () => (await import('./native.js')).native('embeddings'),
  completions: async () => (await import('./native.js')).native('completions'),
  resume: async () => (await import('./native.js')).native('resume'),
  models: async () => (await import('./models.js')).models(),
  processor: async () => (await import('./processor.js')).processor(),
  state: async () => (await import('./state.js')).state(),
  integrity: async () => (await import('./integrity.js')).integrity(),
}
const select = document.querySelector<HTMLSelectElement>('#check')!
const run = document.querySelector<HTMLButtonElement>('#run')!
const statusElement = document.querySelector<HTMLElement>('#status')!
const output = document.querySelector<HTMLElement>('#output')!

for (const name of Object.keys(checks)) select.add(new Option(name, name))
run.onclick = async () => {
  run.disabled = select.disabled = true
  statusElement.textContent = 'Running…'
  output.textContent = ''
  try {
    const choice = Object.entries(checks).find(([name]) => name === select.value)
    if (!choice) throw new Error('Choose a compatibility check.')
    output.textContent = JSON.stringify(await choice[1](), null, 2)
    statusElement.textContent = 'Done'
  } catch (error) {
    statusElement.textContent = error instanceof Error ? error.message : String(error)
  } finally {
    run.disabled = select.disabled = false
  }
}
