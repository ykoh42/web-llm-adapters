import { load } from './engine.js'

const status = document.querySelector<HTMLParagraphElement>('#status')!
const answer = document.createTextNode('')
document.querySelector<HTMLPreElement>('#answer')!.replaceChildren(answer)
const details = document.querySelector<HTMLPreElement>('#details')!
const select = document.querySelector<HTMLSelectElement>('#model')!
const thinking = document.querySelector<HTMLInputElement>('#thinking')!
let cancel: (() => void) | undefined

async function run(prompt: string, enabled = thinking.checked) {
  const controller = new AbortController()
  cancel = () => controller.abort()
  try {
    const { createAgent } = await import('./agent.js')
    controller.signal.throwIfAborted()
    const agent = await createAgent(enabled)
    controller.signal.throwIfAborted()
    cancel = () => agent.abort()
    agent.subscribe((event) => {
      if (event.type === 'message_update' && event.assistantMessageEvent.type === 'text_delta')
        answer.appendData(event.assistantMessageEvent.delta)
    })
    await agent.prompt(prompt)
    return agent.state.messages
  } finally {
    cancel = undefined
  }
}

document.querySelector<HTMLButtonElement>('#load')!.onclick = () => {
  busy(true)
  void load(select.value, document.querySelector<HTMLInputElement>('#worker')!.checked, (text) => {
    status.textContent = text
  })
    .then(() => {
      thinking.disabled = select.value.startsWith('Qwen2.5-')
      thinking.checked = !thinking.disabled
      status.textContent = 'Ready'
    })
    .catch((error) => {
      status.textContent = String(error)
    })
    .finally(() => busy(false))
}
document.querySelector<HTMLButtonElement>('#cancel')!.onclick = () => cancel?.()
document.querySelector<HTMLFormElement>('#chat')!.onsubmit = (event) => {
  event.preventDefault()
  answer.data = ''
  details.textContent = ''
  status.textContent = 'Running'
  const running = run(document.querySelector<HTMLInputElement>('#message')!.value)
  busy(true)
  void running
    .then((messages) => {
      details.textContent = JSON.stringify(messages, null, 2)
      const final = messages.at(-1)
      status.textContent = final?.role === 'assistant' ? final.stopReason : 'Done'
    })
    .catch((error) => {
      status.textContent =
        error instanceof DOMException && error.name === 'AbortError' ? 'aborted' : String(error)
    })
    .finally(() => busy(false))
}

function busy(value: boolean) {
  for (const button of document.querySelectorAll<HTMLButtonElement>('button'))
    button.disabled = button.id === 'cancel' ? !value || !cancel : value
  select.disabled = value
}
