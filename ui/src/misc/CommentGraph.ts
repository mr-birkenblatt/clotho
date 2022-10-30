import { Set } from 'typescript';
import LRU from './LRU';
import { errHnd, json, range, toJson } from './util';

const URL_PREFIX = `${window.location.origin}/api`;
const BATCH_DELAY = 10;

type ApiTopic = {
  topics: { [key: string]: string };
};

type ApiRead = {
  messages: { [key: string]: string };
  skipped: MHash[];
};

type LinkResponse = {
  parent: MHash;
  child: MHash;
  user: string | undefined;
  first: number;
  votes: Votes;
};

type ApiLinkList = {
  links: LinkResponse[];
  next: number;
};

type LineBlock = number & { _lineBlock: void };
export type LineIndex = number & { _lineIndex: void };
export type AdjustedLineIndex = number & { _adjustedLineIndex: void };
export type MHash = string & { _mHash: void };
export type LinkKey = { mhash: MHash; isGetParent: boolean };
export type FullLinkKey = {
  mhash: MHash;
  isGetParent: boolean;
  index: AdjustedLineIndex;
};

export function asLinkKey(fullLinkKey: FullLinkKey): LinkKey {
  const { mhash, isGetParent } = fullLinkKey;
  return { mhash, isGetParent };
}

export function toFullLinkKey(
  linkKey: LinkKey,
  index: AdjustedLineIndex,
  ): FullLinkKey {
    const { mhash, isGetParent } = linkKey;
    return {
      mhash,
      isGetParent,
      index,
    };
  }

export type Votes = { [key: string]: number };

export type Link =
| {
  valid: true;
  parent: MHash;
  child: MHash;
  user: string;
  first: number;
  votes: Votes;
  }
| {
    valid?: false;
  };

export type NotifyContentCB = (mhash: MHash, content: string) => void;
export type NotifyLinkCB = (fullLinkKey: FullLinkKey, link: Link) => void;
export type TopicsCB = (topics: Readonly<Map<MHash, string>>) => void;

export class CommentPool {
  private readonly pool: LRU<MHash, string>;
  private readonly hashQueue: Set<MHash>;
  private readonly inFlight: Set<MHash>;
  private readonly listeners: Map<MHash, NotifyContentCB[]>;
  private topics: Map<MHash, string> | undefined;
  private active: boolean;

  constructor(maxSize?: number) {
    this.pool = new LRU(maxSize || 10000);
    this.hashQueue = new Set<MHash>();
    this.inFlight = new Set<MHash>();
    this.listeners = new Map();
    this.topics = undefined;
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

  private waitFor(mhash: MHash, notify: NotifyContentCB): void {
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

  retrieveMessage(mhash: MHash, notify: NotifyContentCB): void {
    if (!this.pool.has(mhash)) {
      this.hashQueue.add(mhash);
      this.fetchMessages();
    }
    this.waitFor(mhash, notify);
  }

  getMessage(mhash: MHash, notify?: NotifyContentCB): string | undefined {
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

  getTopics(notify: TopicsCB): void {
    if (this.topics) {
      notify(this.topics);
      return;
    }
    fetch(`${URL_PREFIX}/topic`)
      .then(json)
      .then((obj: ApiTopic) => {
        const { topics } = obj;
        const entries = Object.entries(topics) as [MHash, string][];
        this.topics = new Map(entries);
        notify(this.topics);
      })
      .catch(errHnd);
  }
} // CommentPool

class LinkLookup {
  private readonly blockSize: number;
  private readonly linkKey: LinkKey;
  private readonly line: LRU<AdjustedLineIndex, Link>;
  private readonly listeners: Map<AdjustedLineIndex, NotifyLinkCB[]>;
  private readonly activeBlocks: Set<LineBlock>;

  constructor(linkKey: LinkKey, maxLineSize: number, blockSize?: number) {
    this.blockSize = blockSize || 10;
    this.linkKey = linkKey;
    this.line = new LRU(maxLineSize);
    this.listeners = new Map();
    this.activeBlocks = new Set<LineBlock>();
  }

  getLinkKey(): Readonly<LinkKey> {
    return this.linkKey;
  }

  getFullLinkKey(index: AdjustedLineIndex): Readonly<FullLinkKey> {
    return toFullLinkKey(this.getLinkKey(), index);
  }

  private getBlock(index: AdjustedLineIndex): LineBlock {
    return Math.floor(index / this.blockSize) as LineBlock;
  }

  private toIndex(offset: number, block: LineBlock): AdjustedLineIndex {
    return (block * this.blockSize + offset) as AdjustedLineIndex;
  }

  private requestIndex(index: AdjustedLineIndex): void {
    this.fetchLinks(this.getBlock(index));
  }

  private fetchLinks(block: LineBlock): void {
    if (this.activeBlocks.has(block)) {
      return;
    }
    this.activeBlocks.add(block);
    const { mhash, isGetParent } = this.linkKey;
    const query = isGetParent ? { child: mhash } : { parent: mhash };
    const url = `${URL_PREFIX}/${isGetParent ? 'parents' : 'children'}`;

    const finish = () => {
      this.activeBlocks.delete(block);
    };

    const fetchRange = (blockOffset: number) => {
      const fromOffset = this.toIndex(blockOffset, block);
      const remainCount = this.blockSize - blockOffset;
      fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...query,
          offset: fromOffset,
          limit: remainCount,
          scorer: 'best',
        }),
      })
        .then(json)
        .then((obj: ApiLinkList) => {
          const { links, next } = obj;
          const curCount = next - fromOffset;
          const count = curCount > 0 ? curCount : remainCount;
          range(count).forEach((curOffset) => {
            const adjIndex = (fromOffset + curOffset) as AdjustedLineIndex;
            const curLink = links[curOffset];
            let res: Link;
            if (curLink !== undefined) {
              const { child, parent, first, user, votes } = curLink;
              res = {
                valid: true,
                child,
                parent,
                first,
                user: user || '[nouser]',
                votes,
              };
            } else {
              res = { valid: false };
            }
            this.line.set(adjIndex, res);
            this.note(adjIndex);
          });
          if (count < remainCount) {
            fetchRange(blockOffset + count);
          } else {
            finish();
          }
        })
        .catch((e) => {
          finish();
          errHnd(e);
        });
    };

    setTimeout(() => {
      fetchRange(0);
    }, BATCH_DELAY);
  }

  private waitFor(index: AdjustedLineIndex, notify: NotifyLinkCB): void {
    let notes = this.listeners.get(index);
    if (notes === undefined) {
      notes = [];
      this.listeners.set(index, notes);
    }
    notes.push(notify);
    this.note(index);
  }

  private note(index: AdjustedLineIndex): void {
    const link = this.line.get(index);
    if (link !== undefined) {
      const notes = this.listeners.get(index);
      if (notes !== undefined) {
        this.listeners.delete(index);
        notes.forEach((cur) => cur(this.getFullLinkKey(index), link));
      }
    }
  }

  retrieveLink(index: AdjustedLineIndex, notify: NotifyLinkCB): void {
    if (!this.line.has(index)) {
      this.requestIndex(index);
    }
    this.waitFor(index, notify);
  }

  getLink(index: AdjustedLineIndex, notify?: NotifyLinkCB): Link | undefined {
    const res = this.line.get(index);
    if (res !== undefined) {
      return res;
    }
    this.requestIndex(index);
    if (notify !== undefined) {
      this.waitFor(index, notify);
    }
    return undefined;
  }
} // LinkLookup

export class LinkPool {
  private readonly maxLineSize: number;
  private readonly pool: LRU<Readonly<LinkKey>, LinkLookup>;

  constructor(maxSize?: number, maxLineSize?: number) {
    this.maxLineSize = maxLineSize || 100;
    this.pool = new LRU(maxSize || 1000);
  }

  private getLine(linkKey: LinkKey): LinkLookup {
    let res = this.pool.get(linkKey);
    if (res === undefined) {
      res = new LinkLookup(linkKey, this.maxLineSize);
      this.pool.set(linkKey, res);
    }
    return res;
  }

  retrieveLink(fullLinkKey: FullLinkKey, notify: NotifyLinkCB): void {
    const line = this.getLine(asLinkKey(fullLinkKey));
    line.retrieveLink(fullLinkKey.index, notify);
  }

  getLink(fullLinkKey: FullLinkKey, notify?: NotifyLinkCB): Link | undefined {
    const line = this.getLine(asLinkKey(fullLinkKey));
    return line.getLink(fullLinkKey.index, notify);
  }
} // LinkPool
