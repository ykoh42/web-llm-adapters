import { MemoryStorage, type Seq, type StorageWrite } from '@earendil-works/pi-durable'
import type { Context } from '@earendil-works/chord'

/** An append-only IndexedDB commit log, using Pi's reference storage for reads. */
export class BrowserStorage extends MemoryStorage {
  private release!: () => void
  private readonly released = new Promise<void>((resolve) => {
    this.release = resolve
  })

  private constructor(private readonly database: IDBDatabase) {
    super()
  }

  static open(name: string): Promise<BrowserStorage> {
    return new Promise((resolve, reject) => {
      // One Harness owns a database. A second tab must not replay and overwrite stale state.
      void navigator.locks
        .request(name, { ifAvailable: true }, async (lock) => {
          if (!lock) throw new Error('This conversation is already open in another tab.')
          const request = indexedDB.open(name, 1)
          request.onupgradeneeded = () =>
            request.result.createObjectStore('commits', { keyPath: 'seq' })
          const database = await result(request)
          try {
            const storage = new BrowserStorage(database)
            const transaction = database.transaction('commits', 'readonly')
            const commits = (await result(transaction.objectStore('commits').getAll())) as {
              seq: Seq
              writes: StorageWrite[]
            }[]
            for (const commit of commits) storage.prepareCommit(commit.writes, commit.seq).apply()
            resolve(storage)
            await storage.released
          } finally {
            database.close()
          }
        })
        .catch(reject)
    })
  }

  override async commit(writes: readonly StorageWrite[]): Promise<Seq> {
    const prepared = this.prepareCommit(writes)
    const transaction = this.database.transaction('commits', 'readwrite', { durability: 'strict' })
    const committed = new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => resolve()
      transaction.onabort = () =>
        reject(transaction.error ?? new Error('IndexedDB commit aborted.'))
    })
    transaction.objectStore('commits').add({ seq: prepared.seq, writes: prepared.writes })
    await committed
    // Nothing becomes visible to Pi or the page before the database has committed it.
    return prepared.apply()
  }

  override async close(context: Context) {
    try {
      await super.close(context)
    } finally {
      this.database.close()
      this.release()
    }
  }
}

function result<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}
