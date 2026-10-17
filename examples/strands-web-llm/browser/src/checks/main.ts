import { models } from './models.js'
import { processor } from './processor.js'
import { state } from './state.js'
import { integrity } from './integrity.js'

const checks = { models, processor, state, integrity }
const select = document.querySelector<HTMLSelectElement>('#check')!
const run = document.querySelector<HTMLButtonElement>('#run')!
const status = document.querySelector<HTMLElement>('#status')!
const output = document.querySelector<HTMLElement>('#output')!

for (const name of Object.keys(checks)) select.add(new Option(name, name))
run.onclick = async () => {
  run.disabled = select.disabled = true
  status.textContent = 'Running…'
  output.textContent = ''
  try {
    const choice = Object.entries(checks).find(([name]) => name === select.value)
    if (!choice) throw new Error('Choose an engine example.')
    output.textContent = JSON.stringify(await choice[1](), null, 2)
    status.textContent = 'Done'
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error)
  } finally {
    run.disabled = select.disabled = false
  }
}
