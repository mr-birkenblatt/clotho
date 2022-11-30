import { GraphApiProvider, DEFAULT_API } from '../api/graph';
import { toLink, toLinks, UserId } from '../api/types';
import {
  BATCH_DELAY,
  DEFAULT_BLOCK_SIZE,
  DEFAULT_COMMENT_POOL_SIZE,
  DEFAULT_LINE_SIZE,
  DEFAULT_LINK_CACHE_SIZE,
  DEFAULT_LINK_POOL_SIZE,
  DEFAULT_TOPIC_POOL_SIZE,
} from '../misc/constants';
import {
  adj,
  AdjustedLineIndex,
  asLineKey,
  FullDirectKey,
  FullIndirectKey,
  FullKey,
  FullKeyType,
  FullLinkKey,
  FullTopicKey,
  FullUserlikeKey,
  INVALID_KEY,
  INVALID_LINK,
  IsGet,
  KeyType,
  LineKey,
  Link,
  LinkKey,
  MHash,
  UserKey,
  userMHash,
} from './keys';
import LRU from '../misc/LRU';
import {
  assertTrue,
  BlockLoader,
  BlockResponse,
  errHnd,
  num,
  OnCacheMiss,
  reportCacheMiss,
} from '../misc/util';

type NotifyContentCB = (
  mhash: Readonly<MHash>,
  content: Readonly<string>,
) => void;

type CGSettings = {
  maxCommentPoolSize?: Readonly<number>;
  maxTopicSize?: Readonly<number>;
  maxLinkPoolSize?: Readonly<number>;
  maxLinkCache?: Readonly<number>;
  maxLineSize?: Readonly<number>;
  maxUserCache?: Readonly<number>;
  maxUserLineSize?: Readonly<number>;
  blockSize?: Readonly<number>;
};

type ContentValue = readonly [Readonly<MHash>, Readonly<string>];
export type ContentValueExt = readonly [
  Readonly<MHash> | undefined,
  Readonly<string>,
];
type LinkCacheKey = readonly [Readonly<MHash>, Readonly<MHash>];

class CommentPool {
  private readonly api: GraphApiProvider;
  private readonly pool: LRU<Readonly<MHash>, Readonly<string>>;
  private readonly hashQueue: Set<Readonly<MHash>>;
  private readonly inFlight: Set<Readonly<MHash>>;
  private readonly listeners: Map<Readonly<MHash>, NotifyContentCB[]>;
  private readonly topics: BlockLoader<AdjustedLineIndex, ContentValueExt>;

  private active: boolean;

  constructor(
    api: GraphApiProvider,
    maxSize: number,
    maxTopicSize: number,
    blockSize: number,
  ) {
    this.api = api;
    this.pool = new LRU(maxSize);
    this.hashQueue = new Set<Readonly<MHash>>();
    this.inFlight = new Set<Readonly<MHash>>();
    this.listeners = new Map();

    async function loading(
      offset: Readonly<AdjustedLineIndex>,
      limit: number,
    ): Promise<BlockResponse<AdjustedLineIndex, ContentValueExt>> {
      const { topics, next } = await api.topic(num(offset), limit);
      const entries = Object.entries(topics) as [MHash, string][];
      const topicMap = new Map(entries);
      const values: ContentValue[] = Array.from(topicMap.keys())
        .sort()
        .map((mhash) => {
          const topic = topicMap.get(mhash);
          assertTrue(topic !== undefined, `topic does not exist ${mhash}`);
          return [mhash, topic];
        });
      return { values, next: adj(next) };
    }

    this.topics = new BlockLoader(
      maxTopicSize,
      blockSize,
      [undefined, '[unavailable]'],
      loading,
    );
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
      this.api.read(this.inFlight).then(
        (obj) => {
          const { messages, skipped } = obj;
          // FIXME: how can we make sure it is not active when note is called
          this.active = false;
          Object.entries(messages).forEach((cur) => {
            const [mhash, content] = cur as [MHash, string];
            this.pool.set(mhash, content);
            this.note(mhash);
            this.inFlight.delete(mhash);
          });
          skipped.forEach((val) => {
            this.inFlight.add(val);
          });
          if (this.inFlight.size > 0) {
            this.fetchMessages();
          }
        },
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

  async getMessage(
    mhash: Readonly<MHash>,
    ocm: OnCacheMiss,
  ): Promise<ContentValue> {
    const res = this.pool.get(mhash);
    if (res !== undefined) {
      return [mhash, res];
    }
    reportCacheMiss(ocm);
    return new Promise((resolve) => {
      this.waitFor(mhash, (mhash, content) => {
        resolve([mhash, content]);
      });
      this.hashQueue.add(mhash);
      this.fetchMessages();
    });
  }

  async getTopic(
    index: Readonly<AdjustedLineIndex>,
    ocm: OnCacheMiss,
  ): Promise<ContentValueExt> {
    if (num(index) < 0) {
      return [undefined, '[unavailable]'];
    }
    return this.topics.get(index, ocm);
  }

  clearCache(): void {
    this.topics.clear();
    this.pool.clear();
  }
} // CommentPool

class LinkLookup {
  private readonly loader: Readonly<BlockLoader<AdjustedLineIndex, Link>>;

  constructor(
    api: Readonly<GraphApiProvider>,
    linkKey: Readonly<LinkKey>,
    maxLineSize: Readonly<number>,
    blockSize: Readonly<number>,
  ) {
    async function loading(
      offset: Readonly<AdjustedLineIndex>,
      limit: number,
    ): Promise<BlockResponse<AdjustedLineIndex, Link>> {
      const { links, next } = await api.link(linkKey, num(offset), limit);
      return {
        values: toLinks(links),
        next: adj(next),
      };
    }

    this.loader = new BlockLoader(
      maxLineSize,
      blockSize,
      INVALID_LINK,
      loading,
    );
  }

  async getLink(
    index: Readonly<AdjustedLineIndex>,
    ocm: OnCacheMiss,
  ): Promise<Link> {
    return this.loader.get(index, ocm);
  }
} // LinkLookup

class LinkPool {
  private readonly api: Readonly<GraphApiProvider>;
  private readonly maxLineSize: Readonly<number>;
  private readonly maxUserLineSize: Readonly<number>;
  private readonly blockSize: Readonly<number>;
  private readonly linkCache: LRU<LinkCacheKey, Readonly<Link>>;

  private readonly pool: LRU<Readonly<LinkKey>, LinkLookup>;
  private readonly userLinks: LRU<
    Readonly<UserId>,
    BlockLoader<AdjustedLineIndex, Link>
  >;

  constructor(
    api: Readonly<GraphApiProvider>,
    maxSize: Readonly<number>,
    maxLinkCache: Readonly<number>,
    maxLineSize: Readonly<number>,
    maxUserCache: Readonly<number>,
    maxUserLineSize: Readonly<number>,
    blockSize: Readonly<number>,
  ) {
    this.api = api;
    this.maxLineSize = maxLineSize;
    this.maxUserLineSize = maxUserLineSize;
    this.blockSize = blockSize;
    this.pool = new LRU(maxSize);
    this.linkCache = new LRU(maxLinkCache);
    this.userLinks = new LRU(maxUserCache);
  }

  private getLine(
    linkKey: Readonly<LinkKey>,
    ocm: OnCacheMiss,
  ): Readonly<LinkLookup> {
    let res = this.pool.get(linkKey);
    if (res === undefined) {
      reportCacheMiss(ocm);
      res = new LinkLookup(
        this.api,
        linkKey,
        this.maxLineSize,
        this.blockSize,
      );
      this.pool.set(linkKey, res);
    }
    return res;
  }

  private getUserLine(
    key: Readonly<UserKey>,
    ocm: OnCacheMiss,
  ): Readonly<BlockLoader<AdjustedLineIndex, Link>> {
    const { userId } = key;
    let res = this.userLinks.get(userId);
    const api = this.api;

    async function loading(
      offset: Readonly<AdjustedLineIndex>,
      limit: number,
    ): Promise<BlockResponse<AdjustedLineIndex, Link>> {
      const { links, next } = await api.userLink(key, num(offset), limit);
      return {
        values: toLinks(links),
        next: adj(next),
      };
    }

    if (res === undefined) {
      reportCacheMiss(ocm);
      res = new BlockLoader(
        this.maxUserLineSize,
        this.blockSize,
        INVALID_LINK,
        loading,
      );
      this.userLinks.set(userId, res);
    }
    return res;
  }

  async getLink(
    fullLinkKey: Readonly<FullLinkKey>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<Link>> {
    const { mhash, isGet } = fullLinkKey;
    const line = this.getLine(
      {
        keyType: KeyType.link,
        mhash,
        isGet,
      },
      ocm,
    );
    const link = await line.getLink(fullLinkKey.index, ocm);
    if (!link.invalid) {
      this.linkCache.set([link.parent, link.child], link);
    }
    return link;
  }

  async getUserLink(
    key: Readonly<UserKey>,
    index: Readonly<AdjustedLineIndex>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<Link>> {
    const userLine = this.getUserLine(key, ocm);
    return userLine.get(index, ocm);
  }

  async getSingleLink(
    parentHash: Readonly<MHash>,
    childHash: Readonly<MHash>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<Link>> {
    const key: LinkCacheKey = [parentHash, childHash];
    const res = this.linkCache.get(key);
    if (res !== undefined) {
      return res;
    }
    reportCacheMiss(ocm);
    const linkRes = await this.api.singleLink(parentHash, childHash);
    const link = toLink(linkRes);
    this.linkCache.set(key, link);
    return link;
  }

  clearCache(): void {
    this.pool.clear();
    this.linkCache.clear();
    this.userLinks.clear();
  }
} // LinkPool

export default class CommentGraph {
  private readonly msgPool: Readonly<CommentPool>;
  private readonly linkPool: Readonly<LinkPool>;

  constructor(
    api?: Readonly<GraphApiProvider>,
    settings?: Readonly<CGSettings>,
  ) {
    const actualApi = api ?? /* istanbul ignore next */ DEFAULT_API;
    const config = settings ?? /* istanbul ignore next */ {};
    this.msgPool = new CommentPool(
      actualApi,
      config.maxCommentPoolSize ??
        /* istanbul ignore next */ DEFAULT_COMMENT_POOL_SIZE,
      config.maxTopicSize ??
        /* istanbul ignore next */ DEFAULT_TOPIC_POOL_SIZE,
      config.blockSize ?? /* istanbul ignore next */ DEFAULT_BLOCK_SIZE,
    );
    this.linkPool = new LinkPool(
      actualApi,
      config.maxLinkPoolSize ??
        /* istanbul ignore next */ DEFAULT_LINK_POOL_SIZE,
      config.maxLinkCache ??
        /* istanbul ignore next */ DEFAULT_LINK_CACHE_SIZE,
      config.maxLineSize ?? /* istanbul ignore next */ DEFAULT_LINE_SIZE,
      config.maxUserCache ??
        /* istanbul ignore next */ DEFAULT_LINK_CACHE_SIZE,
      config.maxUserLineSize ?? /* istanbul ignore next */ DEFAULT_LINE_SIZE,
      config.blockSize ?? /* istanbul ignore next */ DEFAULT_BLOCK_SIZE,
    );
  }

  private async getMessageByHash(
    fullDirectKey: Readonly<FullDirectKey>,
    ocm: OnCacheMiss,
  ): Promise<ContentValue> {
    return this.msgPool.getMessage(fullDirectKey.mhash, ocm);
  }

  private async getTopicMessage(
    fullTopicKey: Readonly<FullTopicKey>,
    ocm: OnCacheMiss,
  ): Promise<ContentValueExt> {
    const { index } = fullTopicKey;
    return this.msgPool.getTopic(index, ocm);
  }

  private async getMessageFromLink(
    link: Readonly<Link>,
    isGet: Readonly<IsGet>,
    ocm: OnCacheMiss,
  ): Promise<ContentValueExt> {
    if (link.invalid) {
      return [undefined, '[deleted]'];
    }
    const mhash = isGet === IsGet.parent ? link.parent : link.child;
    return this.msgPool.getMessage(mhash, ocm);
  }

  private async getUserMessage(
    key: Readonly<FullUserlikeKey>,
    ocm: OnCacheMiss,
  ): Promise<ContentValueExt> {
    if (key.fullKeyType === FullKeyType.user) {
      return [userMHash(key), `${key.userId}`];
    }
    const { parentUser, index } = key;
    const res = await this.linkPool.getUserLink(
      { keyType: KeyType.user, userId: parentUser },
      index,
      ocm,
    );
    return this.getMessageFromLink(res, IsGet.child, ocm);
  }

  private async getFullLinkMessage(
    fullLinkKey: Readonly<FullLinkKey>,
    ocm: OnCacheMiss,
  ): Promise<ContentValueExt> {
    const link = await this.linkPool.getLink(fullLinkKey, ocm);
    return this.getMessageFromLink(link, fullLinkKey.isGet, ocm);
  }

  async getMessage(
    fullKey: Readonly<FullKey>,
    ocm: OnCacheMiss,
  ): Promise<ContentValueExt> {
    if (fullKey.fullKeyType === FullKeyType.invalid) {
      return [undefined, '[invalid]'];
    }
    if (
      fullKey.fullKeyType === FullKeyType.user ||
      fullKey.fullKeyType === FullKeyType.userchild
    ) {
      return this.getUserMessage(fullKey, ocm);
    }
    if (fullKey.fullKeyType === FullKeyType.direct) {
      return this.getMessageByHash(fullKey, ocm);
    }
    if (fullKey.fullKeyType === FullKeyType.topic) {
      return this.getTopicMessage(fullKey, ocm);
    }
    return this.getFullLinkMessage(fullKey, ocm);
  }

  async getHash(
    fullKey: Readonly<FullKey>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<MHash> | undefined> {
    if (fullKey.fullKeyType === FullKeyType.invalid) {
      return undefined;
    }
    if (fullKey.fullKeyType === FullKeyType.user) {
      return userMHash(fullKey);
    }
    if (fullKey.fullKeyType === FullKeyType.userchild) {
      const link = await this.linkPool.getUserLink(
        { keyType: KeyType.user, userId: fullKey.parentUser },
        fullKey.index,
        ocm,
      );
      if (link.invalid) {
        return undefined;
      }
      return link.child;
    }
    if (fullKey.fullKeyType === FullKeyType.direct) {
      return fullKey.mhash;
    }
    if (fullKey.fullKeyType === FullKeyType.topic) {
      const [topic, _] = await this.msgPool.getTopic(fullKey.index, ocm);
      return topic;
    }
    const link = await this.linkPool.getLink(fullKey, ocm);
    if (link.invalid) {
      return undefined;
    } else {
      return fullKey.isGet === IsGet.parent ? link.parent : link.child;
    }
  }

  async getSingleLink(
    parent: Readonly<FullKey>,
    child: Readonly<FullKey>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<Link>> {
    const parentHash = await this.getHash(parent, ocm);
    if (parentHash === undefined) {
      return INVALID_LINK;
    }
    const phash = parentHash;
    const childHash = await this.getHash(child, ocm);
    if (childHash === undefined) {
      return INVALID_LINK;
    }
    return this.linkPool.getSingleLink(phash, childHash, ocm);
  }

  private async getTopicNextLink(
    fullTopicKey: Readonly<FullTopicKey>,
    nextIndex: Readonly<AdjustedLineIndex>,
    isTop: boolean,
    ocm: OnCacheMiss,
  ): Promise<Readonly<Link>> {
    const { index } = fullTopicKey;
    const [mhash, _] = await this.msgPool.getTopic(index, ocm);
    if (mhash === undefined) {
      return INVALID_LINK;
    }
    return this.linkPool.getLink(
      {
        fullKeyType: FullKeyType.link,
        mhash,
        isGet: isTop ? IsGet.parent : IsGet.child,
        index: nextIndex,
      },
      ocm,
    );
  }

  private async getNextLinkFromLink(
    link: Readonly<Link>,
    isGet: Readonly<IsGet>,
    isTop: boolean,
    nextIndex: Readonly<AdjustedLineIndex>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<Link>> {
    if (link.invalid) {
      return link;
    }
    const topKey: Readonly<FullLinkKey> = {
      fullKeyType: FullKeyType.link,
      mhash: isGet === IsGet.parent ? link.parent : link.child,
      isGet: isTop ? IsGet.parent : IsGet.child,
      index: nextIndex,
    };
    return this.linkPool.getLink(topKey, ocm);
  }

  private async getFullNextLink(
    fullLinkKey: Readonly<FullLinkKey>,
    nextIndex: Readonly<AdjustedLineIndex>,
    isTop: boolean,
    ocm: OnCacheMiss,
  ): Promise<Readonly<Link>> {
    const isGet = fullLinkKey.isGet;
    const link = await this.linkPool.getLink(fullLinkKey, ocm);
    return this.getNextLinkFromLink(link, isGet, isTop, nextIndex, ocm);
  }

  private async getUserBottomLink(
    fullKey: Readonly<FullUserlikeKey>,
    childIndex: Readonly<AdjustedLineIndex>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<Link>> {
    if (fullKey.fullKeyType === FullKeyType.user) {
      const { userId } = fullKey;
      return this.linkPool.getUserLink(
        { keyType: KeyType.user, userId },
        childIndex,
        ocm,
      );
    }
    const res = await this.linkPool.getUserLink(
      { keyType: KeyType.user, userId: fullKey.parentUser },
      fullKey.index,
      ocm,
    );
    return this.getNextLinkFromLink(res, IsGet.child, false, childIndex, ocm);
  }

  async getTopLink(
    fullKey: Readonly<FullIndirectKey>,
    parentIndex: Readonly<AdjustedLineIndex>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<Link>> {
    if (fullKey.fullKeyType === FullKeyType.invalid) {
      return INVALID_LINK;
    }
    if (fullKey.fullKeyType === FullKeyType.user) {
      return INVALID_LINK;
    }
    if (fullKey.fullKeyType === FullKeyType.userchild) {
      if (num(parentIndex) !== 0) {
        return INVALID_LINK;
      }
      const { parentUser, index } = fullKey;
      return this.getUserBottomLink(
        { fullKeyType: FullKeyType.user, userId: parentUser },
        index,
        ocm,
      );
    }
    if (fullKey.fullKeyType === FullKeyType.topic) {
      return this.getTopicNextLink(fullKey, parentIndex, true, ocm);
    }
    return this.getFullNextLink(fullKey, parentIndex, true, ocm);
  }

  async getBottomLink(
    fullKey: Readonly<FullIndirectKey>,
    childIndex: Readonly<AdjustedLineIndex>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<Link>> {
    if (fullKey.fullKeyType === FullKeyType.invalid) {
      return INVALID_LINK;
    }
    if (
      fullKey.fullKeyType === FullKeyType.user ||
      fullKey.fullKeyType === FullKeyType.userchild
    ) {
      return this.getUserBottomLink(fullKey, childIndex, ocm);
    }
    if (fullKey.fullKeyType === FullKeyType.topic) {
      return this.getTopicNextLink(fullKey, childIndex, false, ocm);
    }
    return this.getFullNextLink(fullKey, childIndex, false, ocm);
  }

  private async getTopicNext(
    fullTopicKey: Readonly<FullTopicKey>,
    isGet: Readonly<IsGet>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<LineKey>> {
    const { index } = fullTopicKey;
    const [mhash, _] = await this.msgPool.getTopic(index, ocm);
    if (mhash === undefined) {
      return INVALID_KEY;
    }
    return { keyType: KeyType.link, mhash, isGet };
  }

  private async getFullNext(
    fullLinkKey: Readonly<FullLinkKey>,
    isGet: Readonly<IsGet>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<LineKey>> {
    const link = await this.linkPool.getLink(fullLinkKey, ocm);
    if (link.invalid) {
      return INVALID_KEY;
    }
    return {
      keyType: KeyType.link,
      mhash: fullLinkKey.isGet == IsGet.parent ? link.parent : link.child,
      isGet,
    };
  }

  private async getUserNext(
    fullKey: Readonly<FullUserlikeKey>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<LineKey>> {
    if (fullKey.fullKeyType === FullKeyType.user) {
      return { keyType: KeyType.userchild, parentUser: fullKey.userId };
    }
    const link = await this.linkPool.getUserLink(
      { keyType: KeyType.user, userId: fullKey.parentUser },
      fullKey.index,
      ocm,
    );
    if (link.invalid) {
      return INVALID_KEY;
    }
    return { keyType: KeyType.link, isGet: IsGet.child, mhash: link.child };
  }

  async getParent(
    fullKey: Readonly<FullKey>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<LineKey>> {
    if (fullKey.fullKeyType === FullKeyType.invalid) {
      return asLineKey(fullKey);
    } else if (fullKey.fullKeyType === FullKeyType.user) {
      return INVALID_KEY;
    } else if (fullKey.fullKeyType === FullKeyType.userchild) {
      return { keyType: KeyType.user, userId: fullKey.parentUser };
    } else if (fullKey.fullKeyType === FullKeyType.direct) {
      return {
        keyType: KeyType.link,
        mhash: fullKey.mhash,
        isGet: IsGet.parent,
      };
    }
    if (fullKey.fullKeyType === FullKeyType.topic) {
      return this.getTopicNext(fullKey, IsGet.parent, ocm);
    }
    return this.getFullNext(fullKey, IsGet.parent, ocm);
  }

  async getChild(
    fullKey: Readonly<FullKey>,
    ocm: OnCacheMiss,
  ): Promise<Readonly<LineKey>> {
    if (fullKey.fullKeyType === FullKeyType.invalid) {
      return asLineKey(fullKey);
    } else if (fullKey.fullKeyType === FullKeyType.direct) {
      return {
        keyType: KeyType.link,
        mhash: fullKey.mhash,
        isGet: IsGet.child,
      };
    } else if (fullKey.fullKeyType === FullKeyType.topic) {
      return await this.getTopicNext(fullKey, IsGet.child, ocm);
    } else if (
      fullKey.fullKeyType === FullKeyType.user ||
      fullKey.fullKeyType === FullKeyType.userchild
    ) {
      return await this.getUserNext(fullKey, ocm);
    } else {
      return await this.getFullNext(fullKey, IsGet.child, ocm);
    }
  }

  clearCache(): void {
    this.msgPool.clearCache();
    this.linkPool.clearCache();
  }
} // CommentGraph
