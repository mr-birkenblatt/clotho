import CommentGraph, {
  AdjustedLineIndex,
  FullKey,
  Link,
  MHash,
  NotifyContentCB,
  NotifyLinkCB,
  ValidLink,
} from './CommentGraph';
import TestGraph from './TestGraph';
import { assertTrue, range } from './util';

// FIXME not using fake timers for now as they don't work well with async
// jest.useFakeTimers();

function asTopicKey(index: number): FullKey {
  return {
    topic: true,
    index: index as AdjustedLineIndex,
  };
}

function asLinkKey(
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

type Callback<T extends any[]> = (...args: T) => void;

async function execute<A extends any[], T extends any[], R>(
  fun: (notify: Callback<T>, ...args: A) => R | undefined,
  args: A,
  callback: Callback<T>,
  convertDirect: ((res: R) => T) | undefined,
): Promise<boolean> {
  const marker = jest.fn();
  return new Promise((resolve) => {
    const cb: Callback<T> = (...cbArgs) => {
      callback(...cbArgs);
      resolve(true);
    };
    const notify: Callback<T> = (...cbArgs) => {
      marker();
      if (convertDirect === undefined) {
        cb(...cbArgs);
      }
    };
    const res = fun(notify, ...args);
    expect(marker).not.toBeCalled();
    if (convertDirect === undefined) {
      assertTrue(res === undefined);
    } else {
      assertTrue(res !== undefined);
      cb(...convertDirect(res));
    }
    // console.log('runAllTimers');
    // jest.runAllTimers();
  }).then(() => {
    if (convertDirect === undefined) {
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
const validLink = (
  cb: (vlink: ValidLink) => void,
): ((link: Link) => void) => {
  return (link) => {
    assertTrue(!link.invalid);
    cb(link);
  };
};
const checkLink = (
  parent: string,
  child: string,
): ((link: Link) => void) => {
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
const toArgs = (
  fullKey: FullKey,
  nextIx: number,
): [FullKey, AdjustedLineIndex] => {
  return [fullKey, nextIx as AdjustedLineIndex];
};

test('simple test comment graph', async () => {
  const graph = new TestGraph(3);
  graph.addLinks([
    ['a', 'b'],
    ['a', 'c'],
    ['a', 'd'],
    ['a', 'e'],
    ['a', 'f'],
    ['g', 'a'],
    ['g', 'b'],
    ['h', 'i'],
    ['h', 'j'],
  ]);
  graph.addTopics(['a', 'h']);
  const pool = new CommentGraph(graph.getApiProvider());
  const getMessage = (
    notify: NotifyContentCB,
    fullKey: FullKey,
  ): string | undefined => {
    return pool.getMessage(fullKey, notify);
  };

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
    [asLinkKey('a', false, 0)],
    checkMessage('b'),
    undefined,
  );
  await execute(
    getMessage,
    [asLinkKey('a', false, 2)],
    checkMessage('d'),
    undefined,
  );
  await execute(
    getMessage,
    [asLinkKey('a', false, 2)],
    checkMessage(undefined, 'msg: d'),
    convertMessage,
  );
  await execute(
    getMessage,
    [asLinkKey('a', false, 4)],
    checkMessage('f'),
    undefined,
  );
  await execute(
    getMessage,
    [asLinkKey('a', false, 5)],
    checkMessage(undefined, '[deleted]'),
    convertMessage,
  );
  const hashes = ['b', 'c', 'd', 'e', 'f'];
  const contents = hashes.map((el) => `msg: ${el}`);
  pool.clearCache();
  await Promise.all(
    range(5).map((ix) => {
      return execute(
        getMessage,
        [asLinkKey('a', false, ix)],
        checkMessage(hashes[ix]),
        undefined,
      );
    }),
  );
  await Promise.all(
    range(5).map((ix) => {
      return execute(
        getMessage,
        [asLinkKey('a', false, ix)],
        checkMessage(undefined, contents[ix]),
        convertMessage,
      );
    }),
  );
  await execute(
    getMessage,
    [asLinkKey('b', true, 0)],
    checkMessage('a'),
    undefined,
  );
  await execute(
    getMessage,
    [asLinkKey('b', true, 1)],
    checkMessage('g'),
    undefined,
  );
  await execute(
    getMessage,
    [asLinkKey('b', true, 2)],
    checkMessage(undefined, '[deleted]'),
    convertMessage,
  );
  pool.clearCache();
  await execute(
    getMessage,
    [asLinkKey('b', true, 2)],
    checkMessage(undefined, '[deleted]'),
    undefined,
  );
});

test('parent / child comment graph', async () => {
  const graph = new TestGraph(3);
  graph.addLinks([
    ['a1', 'a2'],
    ['a1', 'b2'],
    ['a1', 'c2'],
    ['a1', 'd2'],
    ['a2', 'a3'],
    ['a3', 'a4'],
    ['a3', 'b4'],
    ['a4', 'a5'],
    ['a5', 'a1'],
    ['b4', 'b2'],
    ['b2', 'b4'],
  ]);
  graph.addTopics(['a2', 'b2']);
  const pool = new CommentGraph(graph.getApiProvider());

  const getTopLink = (
    notify: NotifyLinkCB,
    fullKey: Readonly<FullKey>,
    parentIndex: AdjustedLineIndex,
  ): Link | undefined => {
    return pool.getTopLink(fullKey, parentIndex, notify);
  };
  const getBottomLink = (
    notify: NotifyLinkCB,
    fullKey: Readonly<FullKey>,
    childIndex: AdjustedLineIndex,
  ): Link | undefined => {
    return pool.getBottomLink(fullKey, childIndex, notify);
  };
  const getMessage = (
    notify: NotifyContentCB,
    fullKey: FullKey,
  ): string | undefined => {
    return pool.getMessage(fullKey, notify);
  };

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

  await execute(
    getTopLink,
    toArgs(asLinkKey('d2', true, 0), 0),
    checkLink('a5', 'a1'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('b4', true, 1), 0),
    checkLink('a1', 'b2'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('b4', true, 1), 1),
    checkLink('b4', 'b2'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('b4', true, 0), 0),
    checkLink('a2', 'a3'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('b4', true, 0), 1),
    invalidLink(),
    convertLink,
  );

  await execute(
    getTopLink,
    toArgs(asLinkKey('a4', false, 0), 0),
    checkLink('a4', 'a5'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('a1', false, 1), 0),
    checkLink('a1', 'b2'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('a1', false, 1), 1),
    checkLink('b4', 'b2'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('a3', false, 0), 0),
    checkLink('a3', 'a4'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('a3', false, 1), 0),
    checkLink('a3', 'b4'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('a3', false, 1), 1),
    checkLink('b2', 'b4'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('b2', false, 0), 0),
    checkLink('a3', 'b4'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('b2', false, 0), 0),
    checkLink('a3', 'b4'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('b2', false, 0), 1),
    checkLink('b2', 'b4'),
    convertLink,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('b2', false, 0), 2),
    invalidLink(),
    convertLink,
  );
  pool.clearCache();
  await execute(
    getTopLink,
    toArgs(asLinkKey('b2', false, 1), 0),
    invalidLink(),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asLinkKey('b2', false, 1), 0),
    invalidLink(),
    convertLink,
  );
  pool.clearCache();
  await execute(
    getTopLink,
    toArgs(asLinkKey('b4', false, 0), 1),
    checkLink('b4', 'b2'),
    undefined,
  );
  await execute(
    getTopLink,
    toArgs(asTopicKey(1), 1),
    checkLink('b4', 'b2'),
    undefined,
  );
  console.log('hi');
  await execute(
    getTopLink,
    toArgs(asLinkKey('a5', true, 0), 0),
    checkLink('a3', 'a4'),
    undefined,
    );
  await execute(
    getMessage,
    [asLinkKey('a4', true, 0)],
    checkMessage('a3'),
    undefined,
  );
  await execute(
    getMessage,
    [asLinkKey('a2', false, 0)],
    checkMessage('a3'),
    undefined,
  );
  console.log('ih');

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
  await execute(
    getMessage,
    [asTopicKey(1)],
    checkMessage('b2'),
    undefined,
  );
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
  // ['a1', 'a2'],
  // ['a1', 'b2'],
  // ['a1', 'c2'],
  // ['a1', 'd2'],
  // ['a2', 'a3'],
  // ['a3', 'a4'],
  // ['a3', 'b4'],
  // ['a4', 'a5'],
  // ['a5', 'a1'],
  // ['b4', 'b2'],
  // ['b2', 'b4'],

  // await execute(
  //   getBottomLink,
  //   toArgs(asLinkKey('a1', false, 1), 0),
  //   checkLink('b2', 'b4'),
  //   undefined,
  // );
  // await execute(
  //   getBottomLink,
  //   toArgs(asLinkKey('b4', true, 0), 0),
  //   checkLink('b2', 'b4'),
  //   undefined,
  // );
});
