import { ApiProvider, DEFAULT_API } from './api';
import {
  BATCH_DELAY,
  DEFAULT_BLOCK_SIZE,
  DEFAULT_COMMENT_POOL_SIZE,
  DEFAULT_LINE_SIZE,
  DEFAULT_LINK_CACHE_SIZE,
  DEFAULT_LINK_POOL_SIZE,
  DEFAULT_TOPIC_POOL_SIZE,
} from './constants';
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
  UserId,
  UserKey,
} from './keys';
import LRU from './LRU';
import { assertTrue, BlockLoader, BlockResponse, errHnd, num } from './util';

export type ReadyCB = () => void;
export type NotifyContentCB = (
  mhash: Readonly<MHash> | undefined,
  content: Readonly<string>,
) => void;
export type NotifyHashCB = (mhash: Readonly<MHash> | undefined) => void;
export type NotifyLinkCB = (link: Readonly<Link>) => void;
export type NextCB = (next: Readonly<LineKey>) => void;

export type CGSettings = {
  maxCommentPoolSize?: Readonly<number>;
  maxTopicSize?: Readonly<number>;
  maxLinkPoolSize?: Readonly<number>;
  maxLinkCache?: Readonly<number>;
  maxLineSize?: Readonly<number>;
  maxUserCache?: Readonly<number>;
  maxUserLineSize?: Readonly<number>;
  blockSize?: Readonly<number>;
};

class CommentPool {
  private readonly api: ApiProvider;
  private readonly pool: LRU<Readonly<MHash>, Readonly<string>>;
  private readonly hashQueue: Set<Readonly<MHash>>;
  private readonly inFlight: Set<Readonly<MHash>>;
  private readonly listeners: Map<Readonly<MHash>, NotifyContentCB[]>;
  private readonly topics: BlockLoader<
    AdjustedLineIndex,
    [Readonly<MHash>, Readonly<string>]
  >;

  private active: boolean;

  constructor(
    api: ApiProvider,
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
    ): Promise<
      BlockResponse<AdjustedLineIndex, [Readonly<MHash>, Readonly<string>]>
    > {
      const { topics, next } = await api.topic(num(offset), limit);
      const entries = Object.entries(topics) as [MHash, string][];
      const topicMap = new Map(entries);
      const values: [Readonly<MHash>, Readonly<string>][] = Array.from(
        topicMap.keys(),
      )
        .sort()
        .map((mhash) => {
          const topic = topicMap.get(mhash);
          assertTrue(topic !== undefined, `topic does not exist ${mhash}`);
          return [mhash, topic];
        });
      return { values, next: adj(next) };
    }

    this.topics = new BlockLoader(loading, maxTopicSize, blockSize);
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
        .then((obj) => {
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
    if (res !== undefined) {
      return [mhash, res];
    }
    this.waitFor(mhash, notify);
    this.hashQueue.add(mhash);
    this.fetchMessages();
    return undefined;
  }

  getTopic(
    index: Readonly<AdjustedLineIndex>,
    notify: NotifyContentCB,
  ): Readonly<[Readonly<MHash>, Readonly<string>]> | undefined {
    return this.topics.get(index, ([mhash, content]) =>
      notify(mhash, content),
    );
  }

  clearCache(): void {
    this.topics.clear();
    this.pool.clear();
  }
} // CommentPool

class LinkLookup {
  private readonly loader: Readonly<BlockLoader<AdjustedLineIndex, Link>>;

  constructor(
    api: Readonly<ApiProvider>,
    linkKey: Readonly<LinkKey>,
    maxLineSize: Readonly<number>,
    blockSize: Readonly<number>,
  ) {
    async function loading(
      offset: Readonly<AdjustedLineIndex>,
      limit: number,
    ): Promise<BlockResponse<AdjustedLineIndex, Link>> {
      const { links, next } = await api.link(linkKey, num(offset), limit);
      return { values: links, next: adj(next) };
    }

    this.loader = new BlockLoader(loading, maxLineSize, blockSize);
  }

  retrieveLink(
    index: Readonly<AdjustedLineIndex>,
    notify: NotifyLinkCB,
  ): void {
    this.loader.retrieve(index, notify);
  }

  getLink(
    index: Readonly<AdjustedLineIndex>,
    notify: NotifyLinkCB,
  ): Link | undefined {
    return this.loader.get(index, notify);
  }
} // LinkLookup

class LinkPool {
  private readonly api: Readonly<ApiProvider>;
  private readonly maxLineSize: Readonly<number>;
  private readonly maxUserLineSize: Readonly<number>;
  private readonly blockSize: Readonly<number>;
  private readonly linkCache: LRU<
    Readonly<[Readonly<MHash>, Readonly<MHash>]>,
    Readonly<Link>
  >;

  private readonly pool: LRU<Readonly<LinkKey>, LinkLookup>;
  private readonly userLinks: LRU<
    Readonly<UserId>,
    BlockLoader<AdjustedLineIndex, Link>
  >;

  constructor(
    api: Readonly<ApiProvider>,
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

  private getLine(linkKey: Readonly<LinkKey>): Readonly<LinkLookup> {
    let res = this.pool.get(linkKey);
    if (res === undefined) {
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
  ): Readonly<BlockLoader<AdjustedLineIndex, Link>> {
    const { userId } = key;
    let res = this.userLinks.get(userId);
    const api = this.api;

    async function loading(
      offset: Readonly<AdjustedLineIndex>,
      limit: number,
    ): Promise<BlockResponse<AdjustedLineIndex, Link>> {
      const { links, next } = await api.userLink(key, num(offset), limit);
      return { values: links, next: adj(next) };
    }

    if (res === undefined) {
      res = new BlockLoader(loading, this.maxUserLineSize, this.blockSize);
      this.userLinks.set(userId, res);
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
    const { mhash, isGet } = fullLinkKey;
    const line = this.getLine({
      keyType: KeyType.link,
      mhash,
      isGet,
    });
    line.retrieveLink(fullLinkKey.index, this.constructNotify(notify));
  }

  getLink(
    fullLinkKey: Readonly<FullLinkKey>,
    notify: NotifyLinkCB,
  ): Link | undefined {
    const { mhash, isGet } = fullLinkKey;
    const line = this.getLine({
      keyType: KeyType.link,
      mhash,
      isGet,
    });
    return line.getLink(fullLinkKey.index, this.constructNotify(notify));
  }

  getUserLink(
    key: Readonly<UserKey>,
    index: Readonly<AdjustedLineIndex>,
    notify: NotifyLinkCB,
  ): Link | undefined {
    const userLine = this.getUserLine(key);
    return userLine.get(index, notify);
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
            user,
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
    this.userLinks.clear();
  }
} // LinkPool

export default class CommentGraph {
  private readonly msgPool: Readonly<CommentPool>;
  private readonly linkPool: Readonly<LinkPool>;

  constructor(api?: Readonly<ApiProvider>, settings?: Readonly<CGSettings>) {
    const actualApi = api ?? /* istanbul ignore next */ DEFAULT_API;
    const config = settings ?? {};
    this.msgPool = new CommentPool(
      actualApi,
      config.maxCommentPoolSize ?? DEFAULT_COMMENT_POOL_SIZE,
      config.maxTopicSize ?? DEFAULT_TOPIC_POOL_SIZE,
      config.blockSize ?? DEFAULT_BLOCK_SIZE,
    );
    this.linkPool = new LinkPool(
      actualApi,
      config.maxLinkPoolSize ?? DEFAULT_LINK_POOL_SIZE,
      config.maxLinkCache ?? DEFAULT_LINK_CACHE_SIZE,
      config.maxLineSize ?? DEFAULT_LINE_SIZE,
      config.maxUserCache ?? DEFAULT_LINK_CACHE_SIZE,
      config.maxUserLineSize ?? DEFAULT_LINE_SIZE,
      config.blockSize ?? DEFAULT_BLOCK_SIZE,
    );
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
    return this.msgPool.getTopic(index, notify);
  }

  private getMessageFromLink(
    link: Readonly<Link>,
    isGet: Readonly<IsGet>,
    notifyOnHit: boolean,
    notify: NotifyContentCB,
  ): readonly [Readonly<MHash> | undefined, Readonly<string>] | undefined {
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
    const mhash = isGet === IsGet.parent ? link.parent : link.child;
    const res = this.msgPool.getMessage(mhash, notify);
    if (notifyOnHit && res !== undefined) {
      notify(...res);
    }
    return res;
  }

  private getUserMessage(
    key: Readonly<FullUserlikeKey>,
    notify: NotifyContentCB,
  ): readonly [Readonly<MHash> | undefined, Readonly<string>] | undefined {
    if (key.fullKeyType === FullKeyType.user) {
      return [`${key.userId}` as MHash, `${key.userId}`];
    }
    const { parentUser, index } = key;

    const notifyLink: NotifyLinkCB = (link) => {
      this.getMessageFromLink(link, IsGet.child, true, notify);
    };

    const res = this.linkPool.getUserLink(
      { keyType: KeyType.user, userId: parentUser },
      index,
      notifyLink,
    );
    if (res === undefined) {
      return undefined;
    }
    return this.getMessageFromLink(res, IsGet.child, false, notify);
  }

  private getFullLinkMessage(
    fullLinkKey: Readonly<FullLinkKey>,
    notify: NotifyContentCB,
  ): readonly [Readonly<MHash> | undefined, Readonly<string>] | undefined {
    const notifyLink: NotifyLinkCB = (link) => {
      this.getMessageFromLink(link, fullLinkKey.isGet, true, notify);
    };

    const link = this.linkPool.getLink(fullLinkKey, notifyLink);
    if (link === undefined) {
      return undefined;
    }
    return this.getMessageFromLink(link, fullLinkKey.isGet, false, notify);
  }

  getMessage(
    fullKey: Readonly<FullKey>,
    notify: NotifyContentCB,
  ): readonly [Readonly<MHash> | undefined, Readonly<string>] | undefined {
    if (fullKey.fullKeyType === FullKeyType.invalid) {
      return [undefined, '[invalid]'];
    }
    if (
      fullKey.fullKeyType === FullKeyType.user ||
      fullKey.fullKeyType === FullKeyType.userchild
    ) {
      return this.getUserMessage(fullKey, notify);
    }
    if (fullKey.fullKeyType === FullKeyType.direct) {
      return this.getMessageByHash(fullKey, notify);
    }
    if (fullKey.fullKeyType === FullKeyType.topic) {
      return this.getTopicMessage(fullKey, notify);
    }
    return this.getFullLinkMessage(fullKey, notify);
  }

  getHash(fullKey: Readonly<FullKey>, notify: NotifyHashCB): void {
    if (fullKey.fullKeyType === FullKeyType.invalid) {
      notify(undefined);
    } else if (
      fullKey.fullKeyType === FullKeyType.user ||
      fullKey.fullKeyType === FullKeyType.userchild
    ) {
      notify(undefined);
    } else if (fullKey.fullKeyType === FullKeyType.direct) {
      notify(fullKey.mhash);
    } else if (fullKey.fullKeyType === FullKeyType.topic) {
      const topic = this.msgPool.getTopic(fullKey.index, (res, _) => {
        notify(res);
      });
      if (topic !== undefined) {
        notify(topic[0]);
      }
    } else {
      const notifyLink: NotifyLinkCB = (link) => {
        if (link.invalid) {
          notify(undefined);
        } else {
          notify(fullKey.isGet === IsGet.parent ? link.parent : link.child);
        }
      };
      const link = this.linkPool.getLink(fullKey, notifyLink);
      if (link !== undefined) {
        notifyLink(link);
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
          fullKeyType: FullKeyType.link,
          mhash,
          isGet: isTop ? IsGet.parent : IsGet.child,
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

    const res = this.msgPool.getTopic(index, (mhash, _) => {
      getTopicNextLink(mhash, true);
    });
    if (res === undefined) {
      return undefined;
    }
    const [curMHash, _] = res;
    return getTopicNextLink(curMHash, false);
  }

  private getNextLinkFromLink(
    link: Readonly<Link>,
    isGet: Readonly<IsGet>,
    isTop: boolean,
    nextIndex: Readonly<AdjustedLineIndex>,
    notifyOnHit: boolean,
    notify: NotifyLinkCB,
  ): Link | undefined {
    if (link.invalid) {
      if (notifyOnHit) {
        notify(link);
      }
      return link;
    }
    const topKey: Readonly<FullLinkKey> = {
      fullKeyType: FullKeyType.link,
      mhash: isGet === IsGet.parent ? link.parent : link.child,
      isGet: isTop ? IsGet.parent : IsGet.child,
      index: nextIndex,
    };
    const res = this.linkPool.getLink(topKey, (topLink) => {
      notify(topLink);
    });
    if (notifyOnHit && res !== undefined) {
      notify(res);
    }
    return res;
  }

  private getFullNextLink(
    fullLinkKey: Readonly<FullLinkKey>,
    nextIndex: Readonly<AdjustedLineIndex>,
    isTop: boolean,
    notify: NotifyLinkCB,
  ): Link | undefined {
    const isGet = fullLinkKey.isGet;

    const notifyLink: NotifyLinkCB = (link) => {
      this.getNextLinkFromLink(link, isGet, isTop, nextIndex, true, notify);
    };

    const link = this.linkPool.getLink(fullLinkKey, notifyLink);
    if (link === undefined) {
      return undefined;
    }
    return this.getNextLinkFromLink(
      link,
      isGet,
      isTop,
      nextIndex,
      false,
      notify,
    );
  }

  private getUserBottomLink(
    fullKey: Readonly<FullUserlikeKey>,
    childIndex: Readonly<AdjustedLineIndex>,
    notify: NotifyLinkCB,
  ): Readonly<Link> | undefined {
    if (fullKey.fullKeyType === FullKeyType.user) {
      const { userId } = fullKey;
      return this.linkPool.getUserLink(
        { keyType: KeyType.user, userId },
        childIndex,
        notify,
      );
    }

    const notifyLink: NotifyLinkCB = (link) => {
      this.getNextLinkFromLink(
        link,
        IsGet.child,
        false,
        childIndex,
        true,
        notify,
      );
    };

    const res = this.linkPool.getUserLink(
      { keyType: KeyType.user, userId: fullKey.parentUser },
      fullKey.index,
      notifyLink,
    );
    if (res === undefined) {
      return undefined;
    }
    return this.getNextLinkFromLink(
      res,
      IsGet.child,
      false,
      childIndex,
      false,
      notify,
    );
  }

  getTopLink(
    fullKey: Readonly<FullIndirectKey>,
    parentIndex: Readonly<AdjustedLineIndex>,
    notify: NotifyLinkCB,
  ): Readonly<Link> | undefined {
    if (
      fullKey.fullKeyType === FullKeyType.invalid ||
      fullKey.fullKeyType === FullKeyType.user
    ) {
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
        notify,
      );
    }
    if (fullKey.fullKeyType === FullKeyType.topic) {
      return this.getTopicNextLink(fullKey, parentIndex, true, notify);
    }
    return this.getFullNextLink(fullKey, parentIndex, true, notify);
  }

  getBottomLink(
    fullKey: Readonly<FullIndirectKey>,
    childIndex: Readonly<AdjustedLineIndex>,
    notify: NotifyLinkCB,
  ): Readonly<Link> | undefined {
    if (fullKey.fullKeyType === FullKeyType.invalid) {
      return INVALID_LINK;
    }
    if (fullKey.fullKeyType === FullKeyType.topic) {
      return this.getTopicNextLink(fullKey, childIndex, false, notify);
    }
    if (
      fullKey.fullKeyType === FullKeyType.user ||
      fullKey.fullKeyType === FullKeyType.userchild
    ) {
      return this.getUserBottomLink(fullKey, childIndex, notify);
    }
    return this.getFullNextLink(fullKey, childIndex, false, notify);
  }

  private getTopicNext(
    fullTopicKey: Readonly<FullTopicKey>,
    isGet: Readonly<IsGet>,
    notify: NextCB,
  ): void {
    const { index } = fullTopicKey;

    const getTopicNext = (mhash: Readonly<MHash> | undefined): void => {
      if (mhash === undefined) {
        const res = INVALID_KEY;
        notify(res);
        return;
      }
      const res: LineKey = { keyType: KeyType.link, mhash, isGet };
      notify(res);
    };

    const res = this.msgPool.getTopic(index, (mhash, _) => {
      getTopicNext(mhash);
    });
    if (res !== undefined) {
      const [curMHash, _] = res;
      getTopicNext(curMHash);
    }
  }

  private getFullNext(
    fullLinkKey: Readonly<FullLinkKey>,
    isGet: Readonly<IsGet>,
    notify: NextCB,
  ): void {
    const notifyLink: NotifyLinkCB = (link) => {
      if (link.invalid) {
        const res = INVALID_KEY;
        notify(res);
        return;
      }
      const res: LineKey = {
        keyType: KeyType.link,
        mhash: fullLinkKey.isGet == IsGet.parent ? link.parent : link.child,
        isGet,
      };
      notify(res);
    };

    this.linkPool.retrieveLink(fullLinkKey, notifyLink);
  }

  private getUserNext(
    fullKey: Readonly<FullUserlikeKey>,
    notify: NextCB,
  ): void {
    if (fullKey.fullKeyType === FullKeyType.user) {
      notify({ keyType: KeyType.userchild, parentUser: fullKey.userId });
      return;
    }

    const notifyLink: NotifyLinkCB = (link) => {
      if (link.invalid) {
        notify(INVALID_KEY);
        return;
      }
      notify({ keyType: KeyType.link, isGet: IsGet.child, mhash: link.child });
    };

    const res = this.linkPool.getUserLink(
      { keyType: KeyType.user, userId: fullKey.parentUser },
      fullKey.index,
      notifyLink,
    );
    if (res !== undefined) {
      notifyLink(res);
    }
  }

  getParent(fullKey: Readonly<FullKey>, callback: NextCB): void {
    if (fullKey.fullKeyType === FullKeyType.invalid) {
      callback(asLineKey(fullKey));
    } else if (fullKey.fullKeyType === FullKeyType.user) {
      callback(INVALID_KEY);
    } else if (fullKey.fullKeyType === FullKeyType.userchild) {
      callback({ keyType: KeyType.user, userId: fullKey.parentUser });
    } else if (fullKey.fullKeyType === FullKeyType.direct) {
      callback({
        keyType: KeyType.link,
        mhash: fullKey.mhash,
        isGet: IsGet.parent,
      });
    } else if (fullKey.fullKeyType === FullKeyType.topic) {
      this.getTopicNext(fullKey, IsGet.parent, callback);
    } else {
      this.getFullNext(fullKey, IsGet.parent, callback);
    }
  }

  getChild(fullKey: Readonly<FullKey>, callback: NextCB): void {
    if (fullKey.fullKeyType === FullKeyType.invalid) {
      callback(asLineKey(fullKey));
    } else if (fullKey.fullKeyType === FullKeyType.direct) {
      callback({
        keyType: KeyType.link,
        mhash: fullKey.mhash,
        isGet: IsGet.child,
      });
    } else if (fullKey.fullKeyType === FullKeyType.topic) {
      this.getTopicNext(fullKey, IsGet.child, callback);
    } else if (
      fullKey.fullKeyType === FullKeyType.user ||
      fullKey.fullKeyType === FullKeyType.userchild
    ) {
      this.getUserNext(fullKey, callback);
    } else {
      this.getFullNext(fullKey, IsGet.child, callback);
    }
  }

  clearCache(): void {
    this.msgPool.clearCache();
    this.linkPool.clearCache();
  }
} // CommentGraph
