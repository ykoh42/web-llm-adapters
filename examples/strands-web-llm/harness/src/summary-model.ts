import { Message, TextBlock, type StreamOptions } from '@strands-agents/sdk'
import { BrowserModel } from 'strands-web-llm'

/** Keep facts from long Harness context summaries by reading them in small parts. */
export class SummaryModel extends BrowserModel {
  override async *stream(messages: Message[], options: StreamOptions = {}) {
    const message = messages[0]
    if (
      messages.length !== 1 ||
      options.toolSpecs?.length ||
      message?.role !== 'user' ||
      !message.content.every((block) => block.type === 'textBlock') ||
      message.content[0]?.text !== '<content>' ||
      message.content.at(-1)?.text !== '</content>'
    ) {
      yield* super.stream(messages, options)
      return
    }
    const text = message.content
      .slice(1, -1)
      .map((block) => block.text)
      .join('\n')
    if (text.length <= 4000) {
      yield* super.stream(messages, options)
      return
    }
    const summaries: string[] = []
    for (const chunk of parts(text)) {
      let summary = ''
      for await (const event of super.stream([content(chunk)], options)) {
        if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
          summary += event.delta.text
        if (event.type === 'modelMessageStopEvent' && event.stopReason !== 'endTurn')
          throw new Error('Gemma did not finish a context summary.')
      }
      if (!summary.trim()) throw new Error('Gemma returned an empty context summary.')
      summaries.push(summary)
    }
    yield* super.stream([content(summaries.join('\n\n'))], options)
  }
}

function content(text: string) {
  return new Message({
    role: 'user',
    content: [new TextBlock('<content>'), new TextBlock(text), new TextBlock('</content>')],
  })
}

function* parts(text: string) {
  let part = ''
  // Keep words and identifiers whole, including Unicode and their trailing whitespace.
  for (const word of text.match(/\S+\s*|\s+/gu) ?? []) {
    if (part && part.length + word.length > 4000) {
      yield part
      part = ''
    }
    part += word
  }
  if (part) yield part
}
