import LRU from './LRU';
import { assertTrue, errHnd, json, range, toJson } from './util';

const URL_PREFIX = `${window.location.origin}/api`;
const BATCH_DELAY = 10;

type ApiTopic = {
  topics: Readonly<{ [key: string]: string }>;
};

type ApiRead = {
  messages: Readonly<{ [key: string]: string }>;
  skipped: Readonly<MHash[]>;
};

type LinkResponse = {
  parent: Readonly<MHash>;
  child: Readonly<MHash>;
  user: Readonly<string> | undefined;
  first: Readonly<number>;
  votes: Votes;
};

type ApiLinkList = {
  links: Readonly<LinkResponse[]>;
  next: Readonly<number>;
};

export type ApiProvider = {
  topic: () => Promise<ApiTopic>;
  read: (hashes: Set<Readonly<MHash>>) => Promise<ApiRead>;
  link: (
    linkKey: LinkKey,
    offset: number,
    limit: number,
  ) => Promise<ApiLinkList>;
};

/* istanbul ignore next */
const DEFAULT_API: ApiProvider = {
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
};

type LineBlock = number & { _lineBlock: void };
export type AdjustedLineIndex = number & { _adjustedLineIndex: void };
export type MHash = string & { _mHash: void };

interface LinkKey {
  topic?: Readonly<false>;
  mhash: Readonly<MHash>;
  isGetParent: Readonly<boolean>;
}
interface TopicKey {
  topic: Readonly<true>;
}
export type LineKey = LinkKey | TopicKey;
export const TOPIC_KEY: Readonly<LineKey> = { topic: true };

interface FullLinkKey {
  topic?: Readonly<false>;
  mhash: Readonly<MHash>;
  isGetParent: Readonly<boolean>;
  index: AdjustedLineIndex;
}
interface FullTopicKey {
  topic: Readonly<true>;
  index: AdjustedLineIndex;
}
export type FullKey = FullLinkKey | FullTopicKey;

export function asLineKey(fullKey: Readonly<FullKey>): Readonly<LineKey> {
  if (fullKey.topic) {
    return { topic: true };
  }
  const { mhash, isGetParent } = fullKey;
  return { mhash, isGetParent };
}

export function toFullKey(
  lineKey: Readonly<LineKey>,
  index: AdjustedLineIndex,
): Readonly<FullKey> {
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

export type ReadyCB = () => void;
export type NotifyContentCB = (
  mhash: Readonly<MHash> | undefined,
  content: Readonly<string>,
) => void;
export type NotifyLinkCB = (link: Readonly<Link>) => void;
type TopicsCB = (
  topics: Readonly<[Readonly<MHash>, Readonly<string>][]>,
) => void;

class CommentPool {
  private readonly api: ApiProvider;
  private readonly pool: LRU<Readonly<MHash>, string>;
  private readonly hashQueue: Set<Readonly<MHash>>;
  private readonly inFlight: Set<Readonly<MHash>>;
  private readonly listeners: Map<Readonly<MHash>, NotifyContentCB[]>;
  private topics: [Readonly<MHash>, Readonly<string>][] | undefined;
  private active: boolean;

  constructor(api: ApiProvider, maxSize?: number) {
    this.api = api;
    this.pool = new LRU(maxSize || 10000);
    this.hashQueue = new Set<Readonly<MHash>>();
    this.inFlight = new Set<Readonly<MHash>>();
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
      this.hashQueue.forEach((val) => {
        this.inFlight.add(val);
      });
      this.hashQueue.clear();
      this.api
        .read(this.inFlight)
        .then((obj: ApiRead) => {
          const { messages, skipped } = obj;
          Object.entries(messages).forEach((cur) => {
            const [mhash, content] = cur as [MHash, string];
            this.pool.set(mhash, content);
            this.note(mhash);
            this.inFlight.delete(mhash);
          });
          skipped.forEach((val) => {
            this.inFlight.add(val);
          });
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

  private waitFor(mhash: Readonly<MHash>, notify: NotifyContentCB): void {
    let notes = this.listeners.get(mhash);
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
      if (notes !== undefined) {
        this.listeners.delete(mhash);
        notes.forEach((cur) => cur(mhash, content));
      }
    }
  }

  retrieveMessage(mhash: Readonly<MHash>, notify: NotifyContentCB): void {
    if (!this.pool.has(mhash)) {
      this.hashQueue.add(mhash);
      this.fetchMessages();
    }
    this.waitFor(mhash, notify);
  }

  getMessage(
    mhash: Readonly<MHash>,
    notify?: NotifyContentCB,
  ): string | undefined {
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
      .catch(errHnd);
    return undefined;
  }

  clearCache(): void {
    this.topics = undefined;
    this.pool.clear();
  }
} // CommentPool

class LinkLookup {
  private readonly api: ApiProvider;
  private readonly blockSize: number;
  private readonly linkKey: Readonly<LinkKey>;
  private readonly line: LRU<AdjustedLineIndex, Readonly<Link>>;
  private readonly listeners: Map<AdjustedLineIndex, NotifyLinkCB[]>;
  private readonly activeBlocks: Set<Readonly<LineBlock>>;

  constructor(
    api: ApiProvider,
    linkKey: Readonly<LinkKey>,
    maxLineSize: number,
    blockSize?: number,
  ) {
    this.api = api;
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
    const { mhash, isGetParent } = this.getLinkKey();
    return { mhash, isGetParent, index };
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

    const finish = () => {
      this.activeBlocks.delete(block);
    };

    const fetchRange = (blockOffset: number) => {
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
                user: user || '[nouser]',
                votes,
              };
            } else {
              res = { invalid: true };
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
        notes.forEach((cur) => cur(link));
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

class LinkPool {
  private readonly api: ApiProvider;
  private readonly maxLineSize: number;
  private readonly pool: LRU<Readonly<LinkKey>, LinkLookup>;

  constructor(api: ApiProvider, maxSize?: number, maxLineSize?: number) {
    this.api = api;
    this.maxLineSize = maxLineSize || 100;
    this.pool = new LRU(maxSize || 1000);
  }

  private getLine(linkKey: LinkKey): LinkLookup {
    let res = this.pool.get(linkKey);
    if (res === undefined) {
      res = new LinkLookup(this.api, linkKey, this.maxLineSize);
      this.pool.set(linkKey, res);
    }
    return res;
  }

  retrieveLink(fullLinkKey: FullLinkKey, notify: NotifyLinkCB): void {
    const { mhash, isGetParent } = fullLinkKey;
    const line = this.getLine({ mhash, isGetParent });
    line.retrieveLink(fullLinkKey.index, notify);
  }

  getLink(fullLinkKey: FullLinkKey, notify?: NotifyLinkCB): Link | undefined {
    const { mhash, isGetParent } = fullLinkKey;
    const line = this.getLine({ mhash, isGetParent });
    return line.getLink(fullLinkKey.index, notify);
  }

  clearCache(): void {
    this.pool.clear();
  }
} // LinkPool

export default class CommentGraph {
  private readonly msgPool: CommentPool;
  private readonly linkPool: LinkPool;

  constructor(api?: ApiProvider) {
    const actualApi = api || DEFAULT_API;
    this.msgPool = new CommentPool(actualApi);
    this.linkPool = new LinkPool(actualApi);
  }

  private getTopicMessage(
    fullTopicKey: FullTopicKey,
    notify: NotifyContentCB,
  ): string | undefined {
    const { index } = fullTopicKey;

    const getTopicMessage = (
      topics: Readonly<[Readonly<MHash>, Readonly<string>][]>,
    ): Readonly<[Readonly<MHash> | undefined, Readonly<string>]> => {
      if (index < 0 || index >= topics.length) {
        return [undefined, '[unavailable]'];
      }
      return topics[index];
    };

    const notifyTopics: TopicsCB = (topics) => {
      const [mhash, topic] = getTopicMessage(topics);
      notify(mhash, topic);
    };

    const order = this.msgPool.getTopics(notifyTopics);
    if (order === undefined) {
      return undefined;
    }
    return getTopicMessage(order)[1];
  }

  private getFullLinkMessage(
    fullLinkKey: Readonly<FullLinkKey>,
    notify: NotifyContentCB,
  ): string | undefined {
    const getMessage = (
      link: Readonly<Link>,
      notifyOnHit: boolean,
    ): string | undefined => {
      if (link.invalid) {
        const res = '[deleted]';
        if (notifyOnHit) {
          notify(undefined, res);
        }
        return res;
      }
      const mhash = fullLinkKey.isGetParent ? link.parent : link.child;
      const res = this.msgPool.getMessage(mhash, notify);
      if (notifyOnHit && res !== undefined) {
        notify(mhash, res);
      }
      return res;
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
  ): string | undefined {
    if (!fullKey.topic) {
      return this.getFullLinkMessage(fullKey, notify);
    }
    return this.getTopicMessage(fullKey, notify);
  }

  private getTopicNextLink(
    fullTopicKey: Readonly<FullTopicKey>,
    nextIndex: AdjustedLineIndex,
    isTop: boolean,
    notify: NotifyLinkCB,
  ): Link | undefined {
    const { index } = fullTopicKey;

    const getTopic = (
      topics: Readonly<[Readonly<MHash>, Readonly<string>][]>,
    ): Readonly<MHash> | undefined => {
      if (index < 0 || index >= topics.length) {
        return undefined;
      }
      return topics[index][0];
    };

    const getTopicTopLink = (
      mhash: Readonly<MHash> | undefined,
      notifyOnHit: boolean,
    ): Link | undefined => {
      if (mhash === undefined) {
        const res: Link = { invalid: true };
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
      getTopicTopLink(mhash, true);
    };

    const order = this.msgPool.getTopics(notifyTopics);
    if (order === undefined) {
      return undefined;
    }
    return getTopicTopLink(getTopic(order), false);
  }

  private getFullNextLink(
    fullLinkKey: Readonly<FullLinkKey>,
    nextIndex: AdjustedLineIndex,
    isTop: boolean,
    notify: NotifyLinkCB,
  ): Link | undefined {
    const getLink = (
      link: Readonly<Link>,
      notifyOnHit: boolean,
    ): Link | undefined => {
      if (fullLinkKey.isGetParent !== isTop) {
        if (notifyOnHit) {
          notify(link);
        }
        return link;
      }
      if (link.invalid) {
        if (notifyOnHit) {
          notify(link);
        }
        return link;
      }
      const topKey: Readonly<FullLinkKey> = {
        mhash: link.parent,
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
    fullKey: Readonly<FullKey>,
    parentIndex: AdjustedLineIndex,
    notify: NotifyLinkCB,
  ): Link | undefined {
    if (!fullKey.topic) {
      return this.getFullNextLink(fullKey, parentIndex, true, notify);
    }
    return this.getTopicNextLink(fullKey, parentIndex, true, notify);
  }

  getBottomLink(
    fullKey: Readonly<FullKey>,
    childIndex: AdjustedLineIndex,
    notify: NotifyLinkCB,
  ): Link | undefined {
    if (!fullKey.topic) {
      return this.getFullNextLink(fullKey, childIndex, false, notify);
    }
    return this.getTopicNextLink(fullKey, childIndex, false, notify);
  }

  getParent(
    fullKey: Readonly<FullKey>,
    parentIndex: AdjustedLineIndex,
    callback: (parent: Readonly<LineKey>) => void,
  ): void {
    const getParent = (link: Link) => {
      if (link.invalid) {
        return; // NOTE: we are not following broken links
      }
      callback({ mhash: link.parent, isGetParent: true });
    };

    const res = this.getTopLink(fullKey, parentIndex, getParent);
    if (res !== undefined) {
      getParent(res);
    }
  }

  getChild(
    fullKey: Readonly<FullKey>,
    childIndex: AdjustedLineIndex,
    callback: (child: Readonly<LineKey>) => void,
  ): void {
    const getChild = (link: Link) => {
      if (link.invalid) {
        return; // NOTE: we are not following broken links
      }
      callback({ mhash: link.child, isGetParent: false });
    };

    const res = this.getBottomLink(fullKey, childIndex, getChild);
    if (res !== undefined) {
      getChild(res);
    }
  }

  clearCache(): void {
    this.msgPool.clearCache();
    this.linkPool.clearCache();
  }
} // CommentGraph
