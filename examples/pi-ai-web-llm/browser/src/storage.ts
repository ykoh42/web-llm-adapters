import { IndexedDbStorage } from '@earendil-works/pi-durable/storage/indexeddb'

/** Keep one owner for the official storage until its connection closes. */
export function openStorage(name: string): Promise<IndexedDbStorage> {
  return new Promise((resolve, reject) => {
    void navigator.locks
      .request(name, { ifAvailable: true }, async (lock) => {
        if (!lock) throw new Error('This conversation is already open in another tab.')
        const storage = await IndexedDbStorage.open(name)
        const close = storage.close.bind(storage)
        let release!: () => void
        const released = new Promise<void>((resolve) => {
          release = resolve
        })
        storage.close = async (context) => {
          try {
            await close(context)
          } finally {
            release()
          }
        }
        resolve(storage)
        await released
      })
      .catch(reject)
  })
}
