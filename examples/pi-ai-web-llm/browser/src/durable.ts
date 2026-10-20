import type { ConversationView, LiveState } from '@earendil-works/pi-durable'
import type { openDurable } from './durable-agent.js'

const status = document.querySelector<HTMLParagraphElement>('#status')!
const answer = document.querySelector<HTMLPreElement>('#answer')!
const details = document.querySelector<HTMLPreElement>('#details')!
const select = document.querySelector<HTMLSelectElement>('#model')!
let session: Awaited<ReturnType<typeof openDurable>> | undefined

document.querySelector<HTMLButtonElement>('#load')!.onclick = () => {
  busy(true)
  void open()
    .then(() => {
      status.textContent = 'Ready'
    })
    .catch(showError)
    .finally(() => busy(false))
}

async function open() {
  const previous = session
  session = undefined
  await previous?.close()
  const { openDurable } = await import('./durable-agent.js')
  session = await openDurable(select.value, render, (text) => {
    status.textContent = text
  })
}

document.querySelector<HTMLButtonElement>('#cancel')!.onclick = () => {
  void session?.cancel().catch(showError)
}
document.querySelector<HTMLFormElement>('#chat')!.onsubmit = (event) => {
  event.preventDefault()
  if (!session) return
  busy(true)
  status.textContent = 'Running'
  void session
    .run(document.querySelector<HTMLInputElement>('#message')!.value)
    .then((settled) => {
      if (settled.status === 'done') status.textContent = 'Done'
      else if (settled.reason === 'aborted') status.textContent = 'Stopped'
      else status.textContent = `Not answered: ${settled.reason}`
    })
    .catch(showError)
    .finally(() => busy(false))
}

function render(view: ConversationView) {
  const live = view.docs['pi.live'] as LiveState | undefined
  const latest = view.entries
    .flatMap((entry) => entry.model ?? [])
    .reverse()
    .find((message) => message.role === 'assistant')
  const message = live?.generation?.message ?? latest
  answer.textContent =
    message?.content
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join('') ?? ''
  details.textContent = JSON.stringify(view, null, 2)
}

function busy(value: boolean) {
  for (const button of document.querySelectorAll<HTMLButtonElement>('button'))
    button.disabled =
      button.id === 'cancel' ? !value || !session : value || (button.id === 'send' && !session)
  select.disabled = value
}

function showError(error: unknown) {
  status.textContent = String(error)
}
