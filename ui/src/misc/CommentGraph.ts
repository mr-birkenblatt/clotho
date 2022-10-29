import { Set } from 'typescript';
import LRU from './LRU';
import { errHnd, json, toJson } from './util';

const URL_PREFIX = `${window.location.origin}/api`;
const BATCH_DELAY = 10;

type ApiRead = {
  messages: { [key: string]: string };
  skipped: MHash[];
};

export type LineIndex = number & { _lineIndex: void };
export type AdjustedLineIndex = number & { _adjustedLineIndex: void };
export type MHash = string & { _mHash: void };

type NotifyCB = (mhash: MHash, content: string) => void;

export class CommentPool {
  private readonly pool: LRU<MHash, string>;
  private readonly hashQueue: Set<MHash>;
  private readonly inFlight: Set<MHash>;
  private readonly listeners: Map<MHash, NotifyCB[]>;
  active: boolean;

  constructor(maxSize?: number) {
    this.pool = new LRU(maxSize || 10000);
    this.hashQueue = new Set<MHash>();
    this.inFlight = new Set<MHash>();
    this.listeners = new Map();
    this.active = false;
  }

  private fetchMessages(): void {
    if (this.active) {
      return;
    }
    this.active = true;
    setTimeout(() => {
      this.hashQueue.forEach(this.inFlight.add);
      this.hashQueue.clear();
      fetch(`${URL_PREFIX}/read`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: toJson({
          hashes: this.inFlight,
        }),
      })
        .then(json)
        .then((obj: ApiRead) => {
          const { messages, skipped } = obj;
          Object.entries(messages).forEach((cur) => {
            const [mhash, content] = cur as [MHash, string];
            this.pool.set(mhash, content);
            this.note(mhash);
            this.inFlight.delete(mhash);
          });
          skipped.forEach(this.inFlight.add);
          this.active = false;
          if (this.inFlight.size > 0) {
            this.fetchMessages();
          }
        })
        .catch((e) => {
          this.active = false;
          errHnd(e);
        });
    }, BATCH_DELAY);
  }

  private waitFor(mhash: MHash, notify: NotifyCB): void {
    let notes = this.listeners.get(mhash);
    if (notes === undefined) {
      notes = [];
      this.listeners.set(mhash, notes);
    }
    notes.push(notify);
    this.note(mhash);
  }

  private note(mhash: MHash): void {
    const content = this.pool.get(mhash);
    if (content !== undefined) {
      const notes = this.listeners.get(mhash);
      if (notes !== undefined) {
        this.listeners.delete(mhash);
        notes.forEach((cur) => cur(mhash, content));
      }
    }
  }

  retrieveMessage(mhash: MHash, notify: NotifyCB): void {
    if (!this.pool.has(mhash)) {
      this.hashQueue.add(mhash);
    }
    this.waitFor(mhash, notify);
  }

  getMessage(mhash: MHash, notify?: NotifyCB): string | undefined {
    const res = this.pool.get(mhash);
    if (res !== undefined) {
      return res;
    }
    this.hashQueue.add(mhash);
    if (notify !== undefined) {
      this.waitFor(mhash, notify);
    }
    this.fetchMessages();
    return undefined;
  }
} // CommentPool
