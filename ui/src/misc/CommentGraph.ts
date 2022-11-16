import LRU from './LRU';
import {
  assertTrue,
  errHnd,
  json,
  LoggerCB,
  maybeLog,
  num,
  range,
  safeStringify,
  str,
  toJson,
} from './util';

const URL_PREFIX = `${window.location.origin}/api`;
const BATCH_DELAY = 10;

type ApiTopic = {
  topics: Readonly<{ [key: string]: string }>;
};

type ApiRead = {
  messages: Readonly<{ [key: string]: string }>;
  skipped: Readonly<MHash[]>;
};

type ApiLinkResponse = {
  parent: Readonly<MHash>;
  child: Readonly<MHash>;
  user: Readonly<string> | undefined;
  first: Readonly<number>;
  votes: Votes;
};

type ApiLinkList = {
  links: Readonly<ApiLinkResponse[]>;
  next: Readonly<number>;
};

export type ApiProvider = {
  topic: () => Promise<ApiTopic>;
  read: (hashes: Set<Readonly<MHash>>) => Promise<ApiRead>;
  link: (
    linkKey: Readonly<LinkKey>,
    offset: number,
    limit: number,
  ) => Promise<ApiLinkList>;
  singleLink: (
    parent: Readonly<MHash>,
    child: Readonly<MHash>,
  ) => Promise<ApiLinkResponse>;
};

/* istanbul ignore next */
export const DEFAULT_API: ApiProvider = {
  topic: async () => fetch(`${URL_PREFIX}/topic`).then(json),
  read: async (hashes) => {
    return fetch(`${URL_PREFIX}/read`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: toJson({ hashes }),
    }).then(json);
  },
  link: async (linkKey, offset, limit) => {
    const { mhash, isGetParent } = linkKey;
    const query = isGetParent ? { child: mhash } : { parent: mhash };
    const url = `${URL_PREFIX}/${isGetParent ? 'parents' : 'children'}`;
    return fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...query,
        offset,
        limit,
        scorer: 'best',
      }),
    }).then(json);
  },
  singleLink: async (parent, child) => {
    const url = `${URL_PREFIX}/link`;
    return fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        parent,
        child,
      }),
    }).then(json);
  },
};

type LineBlock = number & { _lineBlock: void };
export type AdjustedLineIndex = number & { _adjustedLineIndex: void };

export function adj(index: number): Readonly<AdjustedLineIndex> {
  return index as AdjustedLineIndex;
}

export function fromAdjustedIndex(index: Readonly<AdjustedLineIndex>): number {
  return num(index);
}

export type MHash = string & { _mHash: void };

export function fromMHash(mhash: Readonly<MHash>): Readonly<string> {
  return str(mhash);
}

interface LinkKey {
  invalid?: Readonly<false>;
  topic?: Readonly<false>;
  mhash: Readonly<MHash>;
  isGetParent: Readonly<boolean>;
}
interface TopicKey {
  invalid?: Readonly<false>;
  topic: Readonly<true>;
}
interface InvalidKey {
  invalid: Readonly<true>;
  topic?: Readonly<false>;
}
export type LineKey = LinkKey | TopicKey | InvalidKey;
export const INVALID_KEY: Readonly<InvalidKey> = { invalid: true };
export const TOPIC_KEY: Readonly<TopicKey> = { topic: true };

export function equalLineKey(
  keyA: Readonly<LineKey>,
  keyB: Readonly<LineKey>,
): boolean {
  if (keyA.invalid && keyB.invalid) {
    return true;
  }
  if (keyA.invalid || keyB.invalid) {
    return false;
  }
  if (keyA.topic && keyB.topic) {
    return true;
  }
  if (keyA.topic || keyB.topic) {
    return false;
  }
  if (keyA.mhash !== keyB.mhash) {
    return false;
  }
  return keyA.isGetParent === keyB.isGetParent;
}

export function equalLineKeys(keysA: LineKey[], keysB: LineKey[]): boolean {
  if (keysA.length !== keysB.length) {
    return false;
  }
  return keysA.reduce((prev, cur, ix) => {
    return prev && equalLineKey(cur, keysB[ix]);
  }, true);
}

export function toLineKey(
  hash: string,
  isGetParent: boolean,
): Readonly<LineKey> {
  return {
    mhash: hash as MHash,
    isGetParent,
  };
}

interface FullDirectKey {
  direct: Readonly<true>;
  invalid?: Readonly<false>;
  topic?: Readonly<false>;
  mhash: Readonly<MHash>;
  topLink?: Readonly<Link>;
}
interface FullLinkKey {
  direct?: Readonly<false>;
  invalid?: Readonly<false>;
  topic?: Readonly<false>;
  mhash: Readonly<MHash>;
  isGetParent: Readonly<boolean>;
  index: Readonly<AdjustedLineIndex>;
}
interface FullTopicKey {
  direct?: Readonly<false>;
  invalid?: Readonly<false>;
  topic: Readonly<true>;
  index: Readonly<AdjustedLineIndex>;
}
interface FullInvalidKey {
  direct?: Readonly<false>;
  invalid: Readonly<true>;
  topic?: Readonly<false>;
}
export type FullKey =
  | FullDirectKey
  | FullLinkKey
  | FullTopicKey
  | FullInvalidKey;
export type FullIndirectKey = FullLinkKey | FullTopicKey | FullInvalidKey;
export const INVALID_FULL_KEY: Readonly<FullInvalidKey> = { invalid: true };

export function asLineKey(
  fullKey: Readonly<FullIndirectKey>,
): Readonly<LineKey> {
  if (fullKey.invalid) {
    return INVALID_KEY;
  }
  if (fullKey.topic) {
    return TOPIC_KEY;
  }
  const { mhash, isGetParent } = fullKey;
  return { mhash, isGetParent };
}

export function toFullKey(
  lineKey: Readonly<LineKey>,
  index: Readonly<AdjustedLineIndex>,
): Readonly<FullIndirectKey> {
  if (lineKey.invalid) {
    return INVALID_FULL_KEY;
  }
  if (lineKey.topic) {
    return { topic: true, index };
  }
  const { mhash, isGetParent } = lineKey;
  return {
    mhash,
    isGetParent,
    index,
  };
}

export function equalFullKey(
  keyA: Readonly<FullKey>,
  keyB: Readonly<FullKey>,
  logger?: LoggerCB,
): boolean {
  const log = maybeLog(logger, 'equalFullKey:');
  if (keyA.invalid && keyB.invalid) {
    return true;
  }
  if (keyA.invalid || keyB.invalid) {
    log(
      `keyA.invalid:${safeStringify(keyA)}`,
      '!==',
      `keyB.invalid:${safeStringify(keyB)}`,
    );
    return false;
  }
  if (keyA.topic && keyB.topic) {
    if (keyA.index === keyB.index) {
      return true;
    }
    log(`topic: keyA.index:${keyA.index} !== keyB.index:${keyB.index}`);
    return false;
  }
  if (keyA.topic || keyB.topic) {
    log(`keyA.topic:${keyA.topic} !== keyB.topic:${keyB.topic}`);
    return false;
  }
  if (keyA.direct && keyB.direct) {
    // NOTE: topLink is a cache
    if (keyA.mhash === keyB.mhash) {
      return true;
    }
    log(`direct: keyA.mhash:${keyA.mhash} !== keyB.mhash:${keyB.mhash}`);
    return false;
  }
  if (keyA.direct || keyB.direct) {
    log(`keyA.direct:${keyA.direct} !== keyB.direct:${keyB.direct}`);
    return false;
  }
  if (keyA.index !== keyB.index) {
    log(`keyA.index:${keyA.index} !== keyB.index:${keyB.index}`);
    return false;
  }
  if (keyA.mhash !== keyB.mhash) {
    log(`keyA.mhash:${keyA.mhash} !== keyB.mhash:${keyB.mhash}`);
    return false;
  }
  if (keyA.isGetParent === keyB.isGetParent) {
    return true;
  }
  log(
    `keyA.isGetParent:${keyA.isGetParent}`,
    '!==',
    `keyB.isGetParent:${keyB.isGetParent}`,
  );
  return false;
}

export function asTopicKey(index: number): Readonly<FullIndirectKey> {
  return {
    topic: true,
    index: index as AdjustedLineIndex,
  };
}

export function asFullKey(
  hash: string,
  isGetParent: boolean,
  index: number,
): Readonly<FullIndirectKey> {
  return {
    mhash: hash as MHash,
    isGetParent,
    index: index as AdjustedLineIndex,
  };
}

export function asDirectKey(hash: string): Readonly<FullKey> {
  return { direct: true, mhash: hash as MHash, topLink: INVALID_LINK };
}

export type Votes = Readonly<{ [key: string]: Readonly<number> }>;

export type ValidLink = {
  invalid?: Readonly<false>;
  parent: Readonly<MHash>;
  child: Readonly<MHash>;
  user: Readonly<string>;
  first: Readonly<number>;
  votes: Votes;
};
export type InvalidLink = {
  invalid: Readonly<true>;
};
export type Link = ValidLink | InvalidLink;
export const INVALID_LINK: Readonly<InvalidLink> = { invalid: true };

export type ReadyCB = () => void;
export type NotifyContentCB = (
  mhash: Readonly<MHash> | undefined,
  content: Readonly<string>,
) => void;
export type NotifyHashCB = (mhash: Readonly<MHash> | undefined) => void;
export type NotifyLinkCB = (link: Readonly<Link>) => void;
type TopicsCB = (
  topics: Readonly<[Readonly<MHash>, Readonly<string>][]>,
) => void;
export type NextCB = (next: Readonly<LineKey>) => void;

class CommentPool {
  private readonly api: ApiProvider;
  private readonly pool: LRU<Readonly<MHash>, Readonly<string>>;
  private readonly hashQueue: Set<Readonly<MHash>>;
  private readonly inFlight: Set<Readonly<MHash>>;
  private readonly listeners: Map<Readonly<MHash>, NotifyContentCB[]>;
  private topics: [Readonly<MHash>, Readonly<string>][] | undefined;
  private active: boolean;

  constructor(api: ApiProvider, maxSize?: number) {
    this.api = api;
    this.pool = new LRU(maxSize ?? 10000);
    this.hashQueue = new Set<Readonly<MHash>>();
    this.inFlight = new Set<Readonly<MHash>>();
    this.listeners = new Map();
    this.topics = undefined;
    this.active = false;
  }

  private fetchMessages(): void {
    console.log('is active', this.active);
    if (this.active) {
      return;
    }
    this.active = true;
    console.log('fetch messages', this.hashQueue.size);
    setTimeout(() => {
      this.hashQueue.forEach((val) => {
        this.inFlight.add(val);
      });
      this.hashQueue.clear();
      console.log('fetch messages api', this.inFlight.size);
      this.api
        .read(this.inFlight)
        .then((obj: ApiRead) => {
          const { messages, skipped } = obj;
          // FIXME: how can we make sure it is not active when note is called
          this.active = false;
          Object.entries(messages).forEach((cur) => {
            const [mhash, content] = cur as [MHash, string];
            this.pool.set(mhash, content);
            console.log('got', mhash);
            this.note(mhash);
            this.inFlight.delete(mhash);
          });
          skipped.forEach((val) => {
            console.log('skipped', val);
            this.inFlight.add(val);
          });
          console.log('set active false');
          if (this.inFlight.size > 0) {
            this.fetchMessages();
          }
        })
        .catch(
          /* istanbul ignore next */
          (e) => {
            this.active = false;
            errHnd(e);
          },
        );
    }, BATCH_DELAY);
  }

  private waitFor(mhash: Readonly<MHash>, notify: NotifyContentCB): void {
    let notes = this.listeners.get(mhash);
    /* istanbul ignore else */
    if (notes === undefined) {
      notes = [];
      this.listeners.set(mhash, notes);
    }
    notes.push(notify);
    this.note(mhash);
  }

  private note(mhash: Readonly<MHash>): void {
    const content = this.pool.get(mhash);
    if (content !== undefined) {
      const notes = this.listeners.get(mhash);
      /* istanbul ignore else */
      if (notes !== undefined) {
        this.listeners.delete(mhash);
        notes.forEach((cur) => cur(mhash, content));
      }
    }
  }

  /* istanbul ignore next: not used anywhere */
  retrieveMessage(mhash: Readonly<MHash>, notify: NotifyContentCB): void {
    this.waitFor(mhash, notify);
    if (!this.pool.has(mhash)) {
      this.hashQueue.add(mhash);
      this.fetchMessages();
    }
  }

  getMessage(
    mhash: Readonly<MHash>,
    notify: NotifyContentCB,
  ): readonly [Readonly<MHash>, Readonly<string>] | undefined {
    const res = this.pool.get(mhash);
    console.log('mhash', mhash, res);
    if (res !== undefined) {
      return [mhash, res];
    }
    this.waitFor(mhash, notify);
    this.hashQueue.add(mhash);
    this.fetchMessages();
    return undefined;
  }

  getTopics(
    notify: TopicsCB,
  ): Readonly<[Readonly<MHash>, Readonly<string>][]> | undefined {
    if (this.topics !== undefined) {
      return this.topics;
    }
    this.api
      .topic()
      .then((obj: ApiTopic) => {
        const { topics } = obj;
        const entries = Object.entries(topics) as [MHash, string][];
        const topicMap = new Map(entries);
        const res: [MHash, string][] = Array.from(topicMap.keys())
          .sort()
          .map((mhash) => {
            const topic = topicMap.get(mhash);
            assertTrue(topic !== undefined);
            return [mhash, topic];
          });
        this.topics = res;
        notify(res);
      })
      .catch(
        /* istanbul ignore next */
        errHnd,
      );
    return undefined;
  }

  clearCache(): void {
    this.topics = undefined;
    this.pool.clear();
  }
} // CommentPool

class LinkLookup {
  private readonly api: Readonly<ApiProvider>;
  private readonly blockSize: Readonly<number>;
  private readonly linkKey: Readonly<LinkKey>;

  private readonly line: LRU<Readonly<AdjustedLineIndex>, Readonly<Link>>;
  private readonly listeners: Map<Readonly<AdjustedLineIndex>, NotifyLinkCB[]>;
  private readonly activeBlocks: Set<Readonly<LineBlock>>;

  constructor(
    api: Readonly<ApiProvider>,
    linkKey: Readonly<LinkKey>,
    maxLineSize: Readonly<number>,
    blockSize?: Readonly<number>,
  ) {
    this.api = api;
    this.blockSize = blockSize ?? 10;
    this.linkKey = linkKey;
    this.line = new LRU(maxLineSize);
    this.listeners = new Map();
    this.activeBlocks = new Set<LineBlock>();
  }

  /* istanbul ignore next: not used anywhere */
  getLinkKey(): Readonly<LinkKey> {
    return this.linkKey;
  }

  /* istanbul ignore next: not used anywhere */
  getFullLinkKey(index: Readonly<AdjustedLineIndex>): Readonly<FullLinkKey> {
    const { mhash, isGetParent } = this.getLinkKey();
    return { mhash, isGetParent, index };
  }

  private getBlock(index: Readonly<AdjustedLineIndex>): LineBlock {
    return Math.floor(num(index) / this.blockSize) as LineBlock;
  }

  private toIndex(
    offset: Readonly<number>,
    block: Readonly<LineBlock>,
  ): AdjustedLineIndex {
    return (num(block) * this.blockSize + offset) as AdjustedLineIndex;
  }

  private requestIndex(index: Readonly<AdjustedLineIndex>): void {
    this.fetchLinks(this.getBlock(index));
  }

  private fetchLinks(block: Readonly<LineBlock>): void {
    if (this.activeBlocks.has(block)) {
      return;
    }
    this.activeBlocks.add(block);

    const finish = () => {
      this.activeBlocks.delete(block);
    };

    const fetchRange = (blockOffset: Readonly<number>): void => {
      const fromOffset = this.toIndex(blockOffset, block);
      const remainCount = this.blockSize - blockOffset;
      this.api
        .link(this.linkKey, fromOffset, remainCount)
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
                child,
                parent,
                first,
                user: user ?? /* istanbul ignore next */ '[nouser]',
                votes,
              };
            } else {
              res = INVALID_LINK;
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
        .catch(
          /* istanbul ignore next */
          (e) => {
            finish();
            errHnd(e);
          },
        );
    };

    setTimeout(() => {
      fetchRange(0);
    }, BATCH_DELAY);
  }

  private waitFor(
    index: Readonly<AdjustedLineIndex>,
    notify: NotifyLinkCB,
  ): void {
    let notes = this.listeners.get(index);
    /* istanbul ignore else */
    if (notes === undefined) {
      notes = [];
      this.listeners.set(index, notes);
    }
    notes.push(notify);
    this.note(index);
  }

  private note(index: Readonly<AdjustedLineIndex>): void {
    const link = this.line.get(index);
    if (link !== undefined) {
      const notes = this.listeners.get(index);
      if (notes !== undefined) {
        this.listeners.delete(index);
        notes.forEach((cur) => cur(link));
      }
    }
  }

  retrieveLink(
    index: Readonly<AdjustedLineIndex>,
    notify: NotifyLinkCB,
  ): void {
    this.waitFor(index, notify);
    if (!this.line.has(index)) {
      this.requestIndex(index);
    }
  }

  getLink(
    index: Readonly<AdjustedLineIndex>,
    notify: NotifyLinkCB,
  ): Link | undefined {
    const res = this.line.get(index);
    if (res !== undefined) {
      return res;
    }
    this.waitFor(index, notify);
    this.requestIndex(index);
    return undefined;
  }
} // LinkLookup

class LinkPool {
  private readonly api: Readonly<ApiProvider>;
  private readonly maxLineSize: Readonly<number>;
  private readonly linkCache: LRU<
    Readonly<[Readonly<MHash>, Readonly<MHash>]>,
    Readonly<Link>
  >;

  private readonly pool: LRU<Readonly<LinkKey>, LinkLookup>;

  constructor(
    api: Readonly<ApiProvider>,
    maxSize?: Readonly<number>,
    maxLinkCache?: Readonly<number>,
    maxLineSize?: Readonly<number>,
  ) {
    this.api = api;
    this.maxLineSize = maxLineSize ?? 100;
    this.pool = new LRU(maxSize ?? 1000);
    this.linkCache = new LRU(maxLinkCache ?? 1000);
  }

  private getLine(linkKey: Readonly<LinkKey>): Readonly<LinkLookup> {
    let res = this.pool.get(linkKey);
    if (res === undefined) {
      res = new LinkLookup(this.api, linkKey, this.maxLineSize);
      this.pool.set(linkKey, res);
    }
    return res;
  }

  private constructNotify = (notify: NotifyLinkCB): NotifyLinkCB => {
    return (link) => {
      if (!link.invalid) {
        this.linkCache.set([link.parent, link.child], link);
      }
      notify(link);
    };
  };

  retrieveLink(
    fullLinkKey: Readonly<FullLinkKey>,
    notify: NotifyLinkCB,
  ): void {
    const { mhash, isGetParent } = fullLinkKey;
    const line = this.getLine({ mhash, isGetParent });
    line.retrieveLink(fullLinkKey.index, this.constructNotify(notify));
  }

  getLink(
    fullLinkKey: Readonly<FullLinkKey>,
    notify: NotifyLinkCB,
  ): Link | undefined {
    const { mhash, isGetParent } = fullLinkKey;
    const line = this.getLine({ mhash, isGetParent });
    return line.getLink(fullLinkKey.index, this.constructNotify(notify));
  }

  getSingleLink(
    parent: Readonly<MHash>,
    child: Readonly<MHash>,
    notify: NotifyLinkCB,
  ): void {
    const key: [Readonly<MHash>, Readonly<MHash>] = [parent, child];
    const res = this.linkCache.get(key);
    if (res !== undefined) {
      notify(res);
    } else {
      this.api
        .singleLink(parent, child)
        .then((linkRes) => {
          const { child, parent, first, user, votes } = linkRes;
          const link = {
            child,
            parent,
            first,
            user: user ?? /* istanbul ignore next */ '[nouser]',
            votes,
          };
          this.linkCache.set(key, link);
          notify(link);
        })
        .catch(
          /* istanbul ignore next */
          errHnd,
        );
    }
  }

  clearCache(): void {
    this.pool.clear();
    this.linkCache.clear();
  }
} // LinkPool

export default class CommentGraph {
  private readonly msgPool: Readonly<CommentPool>;
  private readonly linkPool: Readonly<LinkPool>;

  constructor(api?: Readonly<ApiProvider>) {
    const actualApi = api ?? /* istanbul ignore next */ DEFAULT_API;
    this.msgPool = new CommentPool(actualApi);
    this.linkPool = new LinkPool(actualApi);
  }

  private getMessageByHash(
    fullDirectKey: Readonly<FullDirectKey>,
    notify: NotifyContentCB,
  ): readonly [Readonly<MHash> | undefined, Readonly<string>] | undefined {
    const res = this.msgPool.getMessage(fullDirectKey.mhash, notify);
    if (res === undefined) {
      return undefined;
    }
    return res;
  }

  private getTopicMessage(
    fullTopicKey: Readonly<FullTopicKey>,
    notify: NotifyContentCB,
  ): readonly [Readonly<MHash> | undefined, Readonly<string>] | undefined {
    const { index } = fullTopicKey;

    const getTopicMessage = (
      topics: Readonly<[Readonly<MHash>, Readonly<string>][]>,
    ): Readonly<[Readonly<MHash> | undefined, Readonly<string>]> => {
      const ix = num(index);
      if (ix < 0 || ix >= topics.length) {
        return [undefined, '[unavailable]'];
      }
      return topics[ix];
    };

    const notifyTopics: TopicsCB = (topics) => {
      const [mhash, topic] = getTopicMessage(topics);
      notify(mhash, topic);
    };

    const order = this.msgPool.getTopics(notifyTopics);
    if (order === undefined) {
      return undefined;
    }
    return getTopicMessage(order);
  }

  private getFullLinkMessage(
    fullLinkKey: Readonly<FullLinkKey>,
    notify: NotifyContentCB,
  ): readonly [Readonly<MHash> | undefined, Readonly<string>] | undefined {
    const getMessage = (
      link: Readonly<Link>,
      notifyOnHit: boolean,
    ):
      | readonly [Readonly<MHash> | undefined, Readonly<string>]
      | undefined => {
      if (link.invalid) {
        const res: [Readonly<MHash> | undefined, Readonly<string>] = [
          undefined,
          '[deleted]',
        ];
        if (notifyOnHit) {
          notify(...res);
        }
        return res;
      }
      const mhash = fullLinkKey.isGetParent ? link.parent : link.child;
      const res = this.msgPool.getMessage(mhash, notify);
      if (notifyOnHit && res !== undefined) {
        notify(...res);
      }
      return res !== undefined ? res : undefined;
    };

    const notifyLink: NotifyLinkCB = (link) => {
      getMessage(link, true);
    };

    const link = this.linkPool.getLink(fullLinkKey, notifyLink);
    if (link === undefined) {
      return undefined;
    }
    return getMessage(link, false);
  }

  getMessage(
    fullKey: Readonly<FullKey>,
    notify: NotifyContentCB,
  ): readonly [Readonly<MHash> | undefined, Readonly<string>] | undefined {
    if (fullKey.invalid) {
      console.log('invalid');
      return [undefined, '[invalid]'];
    }
    if (fullKey.direct) {
      console.log('direct');
      return this.getMessageByHash(fullKey, notify);
    }
    if (!fullKey.topic) {
      console.log('full');
      return this.getFullLinkMessage(fullKey, notify);
    }
    console.log('topic');
    return this.getTopicMessage(fullKey, notify);
  }

  getHash(fullKey: Readonly<FullKey>, notify: NotifyHashCB): void {
    if (fullKey.invalid) {
      notify(undefined);
    } else if (fullKey.direct) {
      notify(fullKey.mhash);
    } else if (!fullKey.topic) {
      const notifyLink: NotifyLinkCB = (link) => {
        if (link.invalid) {
          notify(undefined);
        } else {
          notify(fullKey.isGetParent ? link.parent : link.child);
        }
      };
      const link = this.linkPool.getLink(fullKey, notifyLink);
      if (link !== undefined) {
        notifyLink(link);
      }
    } else {
      const notifyTopics: TopicsCB = (topics) => {
        const ix = fromAdjustedIndex(fullKey.index);
        const res = topics[ix];
        if (res === undefined) {
          notify(undefined);
        } else {
          notify(res[0]);
        }
      };
      const topics = this.msgPool.getTopics(notifyTopics);
      if (topics !== undefined) {
        notifyTopics(topics);
      }
    }
  }

  getSingleLink(
    parent: Readonly<FullKey>,
    child: Readonly<FullKey>,
    callback: NotifyLinkCB,
  ): void {
    this.getHash(parent, (parentHash) => {
      if (parentHash === undefined) {
        callback(INVALID_LINK);
        return;
      }
      const phash = parentHash;
      this.getHash(child, (childHash) => {
        if (childHash === undefined) {
          callback(INVALID_LINK);
          return;
        }
        this.linkPool.getSingleLink(phash, childHash, callback);
      });
    });
  }

  private getTopicNextLink(
    fullTopicKey: Readonly<FullTopicKey>,
    nextIndex: Readonly<AdjustedLineIndex>,
    isTop: boolean,
    notify: NotifyLinkCB,
  ): Link | undefined {
    const { index } = fullTopicKey;

    const getTopic = (
      topics: Readonly<[Readonly<MHash>, Readonly<string>][]>,
    ): Readonly<MHash> | undefined => {
      const ix = num(index);
      if (ix < 0 || ix >= topics.length) {
        return undefined;
      }
      return topics[ix][0];
    };

    const getTopicNextLink = (
      mhash: Readonly<MHash> | undefined,
      notifyOnHit: boolean,
    ): Link | undefined => {
      if (mhash === undefined) {
        const res = INVALID_LINK;
        if (notifyOnHit) {
          notify(res);
        }
        return res;
      }
      const res = this.linkPool.getLink(
        {
          mhash,
          isGetParent: isTop,
          index: nextIndex,
        },
        (link) => {
          notify(link);
        },
      );
      if (res !== undefined && notifyOnHit) {
        notify(res);
      }
      return res;
    };

    const notifyTopics: TopicsCB = (topics) => {
      const mhash = getTopic(topics);
      getTopicNextLink(mhash, true);
    };

    const order = this.msgPool.getTopics(notifyTopics);
    if (order === undefined) {
      return undefined;
    }
    return getTopicNextLink(getTopic(order), false);
  }

  private getFullNextLink(
    fullLinkKey: Readonly<FullLinkKey>,
    nextIndex: Readonly<AdjustedLineIndex>,
    isTop: boolean,
    notify: NotifyLinkCB,
  ): Link | undefined {
    const getLink = (
      link: Readonly<Link>,
      notifyOnHit: boolean,
    ): Link | undefined => {
      if (link.invalid) {
        if (notifyOnHit) {
          notify(link);
        }
        return link;
      }
      const topKey: Readonly<FullLinkKey> = {
        mhash: fullLinkKey.isGetParent ? link.parent : link.child,
        isGetParent: isTop,
        index: nextIndex,
      };
      const res = this.linkPool.getLink(topKey, (topLink) => {
        notify(topLink);
      });
      if (notifyOnHit && res !== undefined) {
        notify(res);
      }
      return res;
    };

    const notifyLink: NotifyLinkCB = (link) => {
      getLink(link, true);
    };

    const link = this.linkPool.getLink(fullLinkKey, notifyLink);
    if (link === undefined) {
      return undefined;
    }
    return getLink(link, false);
  }

  getTopLink(
    fullKey: Readonly<FullIndirectKey>,
    parentIndex: Readonly<AdjustedLineIndex>,
    notify: NotifyLinkCB,
  ): Readonly<Link> | undefined {
    if (fullKey.invalid) {
      return INVALID_LINK;
    }
    if (!fullKey.topic) {
      return this.getFullNextLink(fullKey, parentIndex, true, notify);
    }
    return this.getTopicNextLink(fullKey, parentIndex, true, notify);
  }

  getBottomLink(
    fullKey: Readonly<FullIndirectKey>,
    childIndex: Readonly<AdjustedLineIndex>,
    notify: NotifyLinkCB,
  ): Readonly<Link> | undefined {
    if (fullKey.invalid) {
      return INVALID_LINK;
    }
    if (!fullKey.topic) {
      return this.getFullNextLink(fullKey, childIndex, false, notify);
    }
    return this.getTopicNextLink(fullKey, childIndex, false, notify);
  }

  private getTopicNext(
    fullTopicKey: Readonly<FullTopicKey>,
    isGetParent: boolean,
    notify: NextCB,
  ): void {
    const { index } = fullTopicKey;

    const getTopic = (
      topics: Readonly<[Readonly<MHash>, Readonly<string>][]>,
    ): Readonly<MHash> | undefined => {
      const ix = fromAdjustedIndex(index);
      if (ix < 0 || ix >= topics.length) {
        return undefined;
      }
      return topics[ix][0];
    };

    const getTopicNext = (mhash: Readonly<MHash> | undefined): void => {
      if (mhash === undefined) {
        const res = INVALID_KEY;
        notify(res);
        return;
      }
      const res: LineKey = { mhash, isGetParent };
      notify(res);
    };

    const notifyTopics: TopicsCB = (topics) => {
      const mhash = getTopic(topics);
      getTopicNext(mhash);
    };

    const order = this.msgPool.getTopics(notifyTopics);
    if (order !== undefined) {
      getTopicNext(getTopic(order));
    }
  }

  private getFullNext(
    fullLinkKey: Readonly<FullLinkKey>,
    isGetParent: boolean,
    notify: NextCB,
  ): void {
    const notifyLink: NotifyLinkCB = (link) => {
      if (link.invalid) {
        const res = INVALID_KEY;
        notify(res);
        return;
      }
      const res: LineKey = {
        mhash: fullLinkKey.isGetParent ? link.parent : link.child,
        isGetParent,
      };
      notify(res);
    };

    this.linkPool.retrieveLink(fullLinkKey, notifyLink);
  }

  getParent(fullKey: Readonly<FullKey>, callback: NextCB): void {
    if (fullKey.invalid) {
      callback(asLineKey(fullKey));
    } else if (fullKey.direct) {
      callback({ mhash: fullKey.mhash, isGetParent: true });
    } else if (fullKey.topic) {
      this.getTopicNext(fullKey, true, callback);
    } else {
      this.getFullNext(fullKey, true, callback);
    }
  }

  getChild(fullKey: Readonly<FullKey>, callback: NextCB): void {
    if (fullKey.invalid) {
      callback(asLineKey(fullKey));
    } else if (fullKey.direct) {
      callback({ mhash: fullKey.mhash, isGetParent: false });
    } else if (fullKey.topic) {
      this.getTopicNext(fullKey, false, callback);
    } else {
      this.getFullNext(fullKey, false, callback);
    }
  }

  clearCache(): void {
    this.msgPool.clearCache();
    this.linkPool.clearCache();
  }
} // CommentGraph
