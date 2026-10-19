import { mkdir, writeFile } from 'node:fs/promises'
import { prebuiltAppConfig } from '@mlc-ai/web-llm'

const modelIds = prebuiltAppConfig.model_list.map(({ model_id }) => model_id)

await mkdir('public', { recursive: true })
await writeFile('public/models.json', JSON.stringify(modelIds))
