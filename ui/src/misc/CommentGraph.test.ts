import CommentGraph, {
  AdjustedLineIndex,
  asLineKey,
  equalLineKey,
  equalLineKeys,
  FullKey,
  INVALID_FULL_KEY,
  INVALID_KEY,
  LineKey,
  Link,
  MHash,
  NextCB,
  NotifyContentCB,
  NotifyLinkCB,
  toFullKey,
  TOPIC_KEY,
  ValidLink,
} from './CommentGraph';
import { advancedGraph, simpleGraph } from './TestGraph';
import { assertTrue, range } from './util';

// FIXME not using fake timers for now as they don't work well with async
// jest.useFakeTimers();

function asTopicKey(index: number): FullKey {
  return {
    topic: true,
    index: index as AdjustedLineIndex,
  };
}

function asFullKey(
  hash: string,
  isGetParent: boolean,
  index: number,
): FullKey {
  return {
    mhash: hash as MHash,
    isGetParent,
    index: index as AdjustedLineIndex,
  };
}

function toLineKey(hash: string, isGetParent: boolean): LineKey {
  return {
    mhash: hash as MHash,
    isGetParent,
  };
}

type Callback<T extends any[]> = (...args: T) => void;

async function execute<A extends any[], T extends any[], R>(
  fun: (notify: Callback<T>, ...args: A) => R | undefined,
  args: A,
  callback: Callback<T>,
  convertDirect: ((res: R) => T) | undefined,
  alwaysExpectCall?: boolean,
): Promise<boolean> {
  const marker = jest.fn();
  return new Promise((resolve, reject) => {
    const cb: Callback<T> = (...cbArgs) => {
      try {
        callback(...cbArgs);
        resolve(true);
      } catch (e) {
        reject(e);
      }
    };
    const notify: Callback<T> = (...cbArgs) => {
      marker();
      if (convertDirect === undefined) {
        cb(...cbArgs);
      }
    };
    expect(marker).not.toBeCalled();
    const res = fun(notify, ...args);
    if (!alwaysExpectCall) {
      expect(marker).not.toBeCalled();
    }
    if (convertDirect === undefined) {
      assertTrue(res === undefined);
    } else {
      assertTrue(res !== undefined);
      cb(...convertDirect(res));
    }
    // console.log('runAllTimers');
    // jest.runAllTimers();
  }).then(() => {
    if (convertDirect === undefined || alwaysExpectCall) {
      expect(marker).toBeCalled();
      expect(marker).toHaveBeenCalledTimes(1);
    } else {
      expect(marker).not.toBeCalled();
    }
    return true;
  });
}

const convertMessage = (res: string): [undefined, string] => {
  return [undefined, res];
};
const checkMessage = (
  mhash: string | undefined,
  content?: string,
): NotifyContentCB => {
  return (otherMhash, otherContent) => {
    if (mhash !== undefined) {
      expect(otherMhash).toEqual(mhash);
    } else {
      expect(otherMhash).toBe(undefined);
    }
    expect(otherContent).toEqual(
      content !== undefined ? content : `msg: ${mhash}`,
    );
  };
};
const convertLink = (link: Link): [Link] => {
  return [link];
};
const validLink = (cb: (vlink: ValidLink) => void): ((link: Link) => void) => {
  return (link) => {
    assertTrue(!link.invalid);
    cb(link);
  };
};
const checkLink = (parent: string, child: string): ((link: Link) => void) => {
  return validLink((link) => {
    expect(link.parent).toEqual(parent);
    expect(link.child).toEqual(child);
  });
};
const invalidLink = (): ((link: Link) => void) => {
  return (link) => {
    assertTrue(!!link.invalid);
  };
};
const checkNext = (lineKey: Readonly<LineKey>): NextCB => {
  return (child) => {
    assertTrue(equalLineKey(child, lineKey));
  };
};
const toArgs = (
  fullKey: FullKey,
  nextIx: number,
): [FullKey, AdjustedLineIndex] => {
  return [fullKey, nextIx as AdjustedLineIndex];
};

const createGetTopLink = (
  pool: CommentGraph,
): ((
  notify: NotifyLinkCB,
  fullKey: Readonly<FullKey>,
  parentIndex: AdjustedLineIndex,
) => Link | undefined) => {
  return (notify, fullKey, parentIndex) =>
    pool.getTopLink(fullKey, parentIndex, notify);
};

const createGetBottomLink = (
  pool: CommentGraph,
): ((
  notify: NotifyLinkCB,
  fullKey: Readonly<FullKey>,
  childIndex: AdjustedLineIndex,
) => Link | undefined) => {
  return (notify, fullKey, childIndex) =>
    pool.getBottomLink(fullKey, childIndex, notify);
};

const createGetMessage = (
  pool: CommentGraph,
): ((
  notify: NotifyContentCB,
  fullKey: Readonly<FullKey>,
) => string | undefined) => {
  return (notify, fullKey) => pool.getMessage(fullKey, notify);
};

const createGetParent = (
  pool: CommentGraph,
): ((notify: NextCB, fullKey: Readonly<FullKey>) => undefined) => {
  return (notify, fullKey) => pool.getParent(fullKey, notify) as undefined;
};

const createGetChild = (
  pool: CommentGraph,
): ((notify: NextCB, fullKey: Readonly<FullKey>) => undefined) => {
  return (notify, fullKey) => pool.getChild(fullKey, notify) as undefined;
};

test('simple test comment graph', async () => {
  const pool = new CommentGraph(simpleGraph().getApiProvider());
  const getMessage = createGetMessage(pool);

  await execute(getMessage, [asTopicKey(0)], checkMessage('a'), undefined);
  await execute(
    getMessage,
    [asTopicKey(1)],
    checkMessage(undefined, 'msg: h'),
    convertMessage,
  );
  await execute(
    getMessage,
    [asTopicKey(-1)],
    checkMessage(undefined, '[unavailable]'),
    convertMessage,
  );
  await execute(
    getMessage,
    [asTopicKey(2)],
    checkMessage(undefined, '[unavailable]'),
    convertMessage,
  );
  await execute(
    getMessage,
    [asFullKey('a', false, 0)],
    checkMessage('b'),
    undefined,
  );
  await execute(
    getMessage,
    [asFullKey('a', false, 2)],
    checkMessage('d'),
    undefined,
  );
  await execute(
    getMessage,
    [asFullKey('a', false, 2)],
    checkMessage(undefined, 'msg: d'),
    convertMessage,
  );
  await execute(
    getMessage,
    [asFullKey('a', false, 4)],
    checkMessage('f'),
    undefined,
  );
  await execute(
    getMessage,
    [asFullKey('a', false, 5)],
    checkMessage(undefined, '[deleted]'),
    convertMessage,
  );
});

test('simple bulk message reading', async () => {
  const pool = new CommentGraph(simpleGraph().getApiProvider());
  const getMessage = createGetMessage(pool);

  const hashes = ['b', 'c', 'd', 'e', 'f'];
  const contents = hashes.map((el) => `msg: ${el}`);
  await Promise.all(
    range(5).map((ix) => {
      return execute(
        getMessage,
        [asFullKey('a', false, ix)],
        checkMessage(hashes[ix]),
        undefined,
      );
    }),
  );
  await Promise.all(
    range(5).map((ix) => {
      return execute(
        getMessage,
        [asFullKey('a', false, ix)],
        checkMessage(undefined, contents[ix]),
        convertMessage,
      );
    }),
  );
  await execute(
    getMessage,
    [asFullKey('b', true, 0)],
    checkMessage('a'),
    undefined,
  );
  await execute(
    getMessage,
    [asFullKey('b', true, 1)],
    checkMessage('g'),
    undefined,
  );
  await execute(
    getMessage,
    [asFullKey('b', true, 2)],
    checkMessage(undefined, '[deleted]'),
    convertMessage,
  );
  pool.clearCache();
  await execute(
    getMessage,
    [asFullKey('b', true, 2)],
    checkMessage(undefined, '[deleted]'),
    undefined,
  );
  await execute(
    getMessage,
    [INVALID_FULL_KEY],
    checkMessage(undefined, '[invalid]'),
    convertMessage,
  );
});

test('topic comment graph', async () => {
  const pool = new CommentGraph(advancedGraph().getApiProvider());
  const getTopLink = createGetTopLink(pool);

  await execute(
    getTopLink,
    toArgs(asTopicKey(0), 0),
    checkLink('a1', 'a2'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asTopicKey(0), 0),
    checkLink('a1', 'a2'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asTopicKey(1), 0),
    checkLink('a1', 'b2'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asTopicKey(1), 1),
    checkLink('b4', 'b2'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asTopicKey(1), 2),
    invalidLink(),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asTopicKey(2), 0),
    invalidLink(),
    convertLink,
  );
  pool.clearCache();
  await execute(
    getTopLink,
    toArgs(asTopicKey(2), 1),
    invalidLink(),
    undefined,
  );
});

test('parent comment graph', async () => {
  const pool = new CommentGraph(advancedGraph().getApiProvider());
  const getTopLink = createGetTopLink(pool);
  await execute(
    getTopLink,
    toArgs(asFullKey('d2', true, 0), 0),
    checkLink('a5', 'a1'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('b4', true, 1), 0),
    checkLink('a1', 'b2'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('b4', true, 1), 1),
    checkLink('b4', 'b2'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('b4', true, 0), 0),
    checkLink('a2', 'a3'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('b4', true, 0), 1),
    invalidLink(),
    convertLink,
  );
});

test('cache edge cases for comment graph', async () => {
  const pool = new CommentGraph(advancedGraph().getApiProvider());
  const getTopLink = createGetTopLink(pool);
  const getBottomLink = createGetBottomLink(pool);
  const getMessage = createGetMessage(pool);
  await execute(
    getTopLink,
    toArgs(asFullKey('a4', false, 0), 0),
    checkLink('a4', 'a5'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('a1', false, 1), 0),
    checkLink('a1', 'b2'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('a1', false, 1), 1),
    checkLink('b4', 'b2'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('a3', false, 0), 0),
    checkLink('a3', 'a4'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('a3', false, 1), 0),
    checkLink('a3', 'b4'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('a3', false, 1), 1),
    checkLink('b2', 'b4'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('b2', false, 0), 0),
    checkLink('a3', 'b4'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('b2', false, 0), 0),
    checkLink('a3', 'b4'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('b2', false, 0), 1),
    checkLink('b2', 'b4'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('b2', false, 0), 2),
    invalidLink(),
    convertLink,
  );
  pool.clearCache();
  await execute(
    getTopLink,
    toArgs(asFullKey('b2', false, 1), 0),
    invalidLink(),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('b2', false, 1), 0),
    invalidLink(),
    convertLink,
  );
  pool.clearCache();
  await execute(
    getTopLink,
    toArgs(asFullKey('b4', false, 0), 1),
    checkLink('b4', 'b2'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asTopicKey(1), 1),
    checkLink('b4', 'b2'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asFullKey('a5', true, 0), 0),
    checkLink('a3', 'a4'),
    undefined,
  );
  await execute(
    getMessage,
    [asFullKey('a4', true, 0)],
    checkMessage('a3'),
    undefined,
  );
  await execute(
    getMessage,
    [asFullKey('a2', false, 0)],
    checkMessage('a3'),
    undefined,
  );

  pool.clearCache();
  await execute(
    getBottomLink,
    toArgs(asTopicKey(0), 0),
    checkLink('a2', 'a3'),
    undefined,
  );
  await execute(
    getBottomLink,
    toArgs(asTopicKey(0), 0),
    checkLink('a2', 'a3'),
    convertLink,
  );
  pool.clearCache();
  await execute(
    getBottomLink,
    toArgs(asTopicKey(0), 1),
    invalidLink(),
    undefined,
  );
  await execute(
    getBottomLink,
    toArgs(asTopicKey(0), 1),
    invalidLink(),
    convertLink,
  );
  await execute(
    getBottomLink,
    toArgs(asTopicKey(1), 0),
    checkLink('b2', 'b4'),
    undefined,
  );
  pool.clearCache();
  await execute(getMessage, [asTopicKey(1)], checkMessage('b2'), undefined);
  await execute(
    getBottomLink,
    toArgs(asTopicKey(-1), 0),
    invalidLink(),
    convertLink,
  );
  await execute(
    getBottomLink,
    toArgs(asTopicKey(2), 0),
    invalidLink(),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(INVALID_FULL_KEY, 0),
    invalidLink(),
    convertLink,
  );
  await execute(
    getBottomLink,
    toArgs(INVALID_FULL_KEY, 0),
    invalidLink(),
    convertLink,
  );
});

test('child comment graph', async () => {
  const pool = new CommentGraph(advancedGraph().getApiProvider());
  const getBottomLink = createGetBottomLink(pool);

  await execute(
    getBottomLink,
    toArgs(asFullKey('a1', false, 1), 0),
    checkLink('b2', 'b4'),
    undefined,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('b4', true, 1), 0),
    checkLink('b2', 'b4'),
    undefined,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('b4', true, 0), 0),
    checkLink('a3', 'a4'),
    undefined,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('b4', true, 0), 1),
    checkLink('a3', 'b4'),
    convertLink,
  );

  pool.clearCache();
  await execute(
    getBottomLink,
    toArgs(asFullKey('a5', false, 0), 0),
    checkLink('a1', 'a2'),
    undefined,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('a5', false, 0), 1),
    checkLink('a1', 'b2'),
    convertLink,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('a5', false, 0), 2),
    checkLink('a1', 'c2'),
    convertLink,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('a5', false, 0), 3),
    checkLink('a1', 'd2'),
    convertLink,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('a5', false, 0), 4),
    invalidLink(),
    convertLink,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('a5', false, 0), -1),
    invalidLink(),
    undefined,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('c2', true, 0), 1),
    checkLink('a1', 'b2'),
    undefined,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('c2', true, 1), 1),
    invalidLink(),
    convertLink,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('c2', true, 0), 2),
    checkLink('a1', 'c2'),
    convertLink,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('a2', false, 0), 0),
    checkLink('a3', 'a4'),
    undefined,
  );
  await execute(
    getBottomLink,
    toArgs(asFullKey('a2', false, 0), 1),
    checkLink('a3', 'b4'),
    convertLink,
  );
});

test('get parent / child comment graph', async () => {
  const pool = new CommentGraph(advancedGraph().getApiProvider());
  const getParent = createGetParent(pool);
  const getChild = createGetChild(pool);

  await execute(
    getChild,
    [asFullKey('a2', false, 0)],
    checkNext(toLineKey('a3', false)),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asFullKey('c2', true, 0)],
    checkNext(toLineKey('a1', false)),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asFullKey('c2', false, 0)],
    checkNext(INVALID_KEY),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asFullKey('a1', false, 0)],
    checkNext(toLineKey('a2', false)),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asFullKey('a1', false, 1)],
    checkNext(toLineKey('b2', false)),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asFullKey('a1', false, 2)],
    checkNext(toLineKey('c2', false)),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asFullKey('a1', false, 2)],
    checkNext(toLineKey('c2', false)),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asFullKey('a1', false, 4)],
    checkNext(INVALID_KEY),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asFullKey('b2', true, 0)],
    checkNext(toLineKey('a1', true)),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asFullKey('b2', true, 1)],
    checkNext(toLineKey('b4', true)),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asFullKey('b4', true, 0)],
    checkNext(toLineKey('a3', false)),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asFullKey('b4', true, 0)],
    checkNext(toLineKey('a3', false)),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asFullKey('b4', true, 0)],
    checkNext(toLineKey('a3', false)),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asFullKey('a1', false, 1)],
    checkNext(toLineKey('b2', true)),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asFullKey('a1', false, 0)],
    checkNext(toLineKey('a2', true)),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asFullKey('b4', true, 1)],
    checkNext(toLineKey('b2', true)),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asFullKey('b4', true, 0)],
    checkNext(toLineKey('a3', true)),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asFullKey('b4', true, 0)],
    checkNext(toLineKey('a3', true)),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asFullKey('b4', true, 2)],
    checkNext(INVALID_KEY),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asFullKey('b4', true, 1)],
    checkNext(toLineKey('b2', false)),
    undefined,
    true,
  );
});

test('get parent / child of topic', async () => {
  const pool = new CommentGraph(advancedGraph().getApiProvider());
  const getParent = createGetParent(pool);
  const getChild = createGetChild(pool);

  await execute(
    getChild,
    [asTopicKey(0)],
    checkNext(toLineKey('a2', false)),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asTopicKey(1)],
    checkNext(toLineKey('b2', false)),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asTopicKey(0)],
    checkNext(toLineKey('a2', true)),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asTopicKey(1)],
    checkNext(toLineKey('b2', true)),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asTopicKey(1)],
    checkNext(toLineKey('b2', true)),
    undefined,
    true,
  );
  await execute(
    getParent,
    [asTopicKey(2)],
    checkNext(INVALID_KEY),
    undefined,
    true,
  );
  await execute(
    getChild,
    [asTopicKey(-1)],
    checkNext(INVALID_KEY),
    undefined,
    true,
  );
  await execute(
    getChild,
    [INVALID_FULL_KEY],
    checkNext(INVALID_KEY),
    undefined,
    true,
  );
  await execute(
    getParent,
    [INVALID_FULL_KEY],
    checkNext(INVALID_KEY),
    undefined,
    true,
  );
  await Promise.all([
    execute(
      getParent,
      [asTopicKey(0)],
      checkNext(toLineKey('a2', true)),
      undefined,
      true,
    ),
    execute(
      getParent,
      [asTopicKey(0)],
      checkNext(toLineKey('a2', true)),
      undefined,
      true,
    ),
    execute(
      getParent,
      [asTopicKey(1)],
      checkNext(toLineKey('b2', true)),
      undefined,
      true,
    ),
    execute(
      getChild,
      [asTopicKey(0)],
      checkNext(toLineKey('a2', false)),
      undefined,
      true,
    ),
    execute(
      getChild,
      [asTopicKey(0)],
      checkNext(toLineKey('a2', false)),
      undefined,
      true,
    ),
    execute(
      getChild,
      [asTopicKey(1)],
      checkNext(toLineKey('b2', false)),
      undefined,
      true,
    ),
  ]);
});

test('line keys', async () => {
  assertTrue(
    equalLineKeys(
      [
        INVALID_KEY,
        TOPIC_KEY,
        { mhash: 'a' as MHash, isGetParent: false },
        { mhash: 'a' as MHash, isGetParent: true },
      ],
      [
        { invalid: true },
        { topic: true },
        { mhash: 'a' as MHash, isGetParent: false },
        { mhash: 'a' as MHash, isGetParent: true },
      ],
    ),
  );
  const comp: [Readonly<LineKey>, Readonly<LineKey>][] = [
    [INVALID_KEY, TOPIC_KEY],
    [INVALID_KEY, { mhash: 'a' as MHash, isGetParent: false }],
    [TOPIC_KEY, INVALID_KEY],
    [TOPIC_KEY, { mhash: 'a' as MHash, isGetParent: false }],
    [toLineKey('a', false), toLineKey('a', true)],
    [toLineKey('a', true), toLineKey('a', false)],
    [toLineKey('b', false), toLineKey('a', false)],
    [toLineKey('b', true), toLineKey('a', true)],
    [toLineKey('c', false), toLineKey('a', true)],
    [toLineKey('c', true), toLineKey('a', false)],
  ];
  assertTrue(
    comp.every((cur) => {
      const [a, b] = cur;
      return !equalLineKey(a, b);
    }),
  );
  assertTrue(!equalLineKeys([], [INVALID_KEY]));
  assertTrue(!equalLineKeys([TOPIC_KEY], [INVALID_KEY]));
  expect(toFullKey(toLineKey('a', true), -1 as AdjustedLineIndex)).toEqual(
    asFullKey('a', true, -1),
  );
  expect(toFullKey(toLineKey('a', false), 0 as AdjustedLineIndex)).toEqual(
    asFullKey('a', false, 0),
  );
  expect(toFullKey(toLineKey('c', true), 3 as AdjustedLineIndex)).toEqual(
    asFullKey('c', true, 3),
  );
  expect(toFullKey(TOPIC_KEY, -1 as AdjustedLineIndex)).toEqual(
    asTopicKey(-1),
  );
  expect(toFullKey(TOPIC_KEY, 5 as AdjustedLineIndex)).toEqual(asTopicKey(5));
  expect(toFullKey(TOPIC_KEY, 0 as AdjustedLineIndex)).toEqual(asTopicKey(0));
  expect(toFullKey(INVALID_KEY, 2 as AdjustedLineIndex)).toEqual(
    INVALID_FULL_KEY,
  );
  expect(
    asLineKey(toFullKey(toLineKey('f', false), 4 as AdjustedLineIndex)),
  ).toEqual(toLineKey('f', false));
  expect(
    asLineKey(toFullKey(toLineKey('g', true), 0 as AdjustedLineIndex)),
  ).toEqual(toLineKey('g', true));
  expect(asLineKey(toFullKey(TOPIC_KEY, 5 as AdjustedLineIndex))).toEqual(
    TOPIC_KEY,
  );
  expect(asLineKey(toFullKey(INVALID_KEY, 6 as AdjustedLineIndex))).toEqual(
    INVALID_KEY,
  );
});
