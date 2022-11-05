import CommentGraph, {
  AdjustedLineIndex,
  FullKey,
  Link,
  MHash,
  NotifyContentCB,
  NotifyLinkCB,
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

  const getMessage = (notify: NotifyContentCB, fullKey: FullKey): string | undefined => {
    return pool.getMessage(fullKey, notify);
  };
  const convertMessage = (res: string): [undefined, string] => {
    return [undefined, res];
  };

  await execute(
    getMessage,
    [asTopicKey(0)],
    (mhash, content) => {
      expect(mhash).toEqual('a');
      expect(content).toEqual('msg: a');
    },
    undefined,
  );
  await execute(
    getMessage,
    [asTopicKey(1)],
    (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('msg: h');
    },
    convertMessage,
  );
  await execute(
    getMessage,
    [asTopicKey(-1)],
    (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('[unavailable]');
    },
    convertMessage,
  );
  await execute(
    getMessage,
    [asTopicKey(2)],
    (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('[unavailable]');
    },
    convertMessage,
  );
  await execute(
    getMessage,
    [asLinkKey('a', false, 0)],
    (mhash, content) => {
      expect(mhash).toEqual('b');
      expect(content).toEqual('msg: b');
    },
    undefined,
  );
  await execute(
    getMessage,
    [asLinkKey('a', false, 2)],
    (mhash, content) => {
      expect(mhash).toEqual('d');
      expect(content).toEqual('msg: d');
    },
    undefined,
  );
  await execute(
    getMessage,
    [asLinkKey('a', false, 2)],
    (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('msg: d');
    },
    convertMessage,
  );
  await execute(
    getMessage,
    [asLinkKey('a', false, 4)],
    (mhash, content) => {
      expect(mhash).toEqual('f');
      expect(content).toEqual('msg: f');
    },
    undefined,
  );
  await execute(
    getMessage,
    [asLinkKey('a', false, 5)],
    (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('[deleted]');
    },
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
        (mhash, content) => {
          expect(mhash).toEqual(hashes[ix]);
          expect(content).toEqual(contents[ix]);
        },
        undefined,
      );
    }),
  );
  await Promise.all(
    range(5).map((ix) => {
      return execute(
        getMessage,
        [asLinkKey('a', false, ix)],
        (mhash, content) => {
          expect(mhash).toBe(undefined);
          expect(content).toEqual(contents[ix]);
        },
        convertMessage,
      );
    }),
  );
  await execute(
    getMessage,
    [asLinkKey('b', true, 0)],
    (mhash, content) => {
      expect(mhash).toBe('a');
      expect(content).toEqual('msg: a');
    },
    undefined,
  );
  await execute(
    getMessage,
    [asLinkKey('b', true, 1)],
    (mhash, content) => {
      expect(mhash).toBe('g');
      expect(content).toEqual('msg: g');
    },
    undefined,
  );
  await execute(
    getMessage,
    [asLinkKey('b', true, 2)],
    (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('[deleted]');
    },
    convertMessage,
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
    parentIndex: AdjustedLineIndex): Link | undefined => {
      return pool.getTopLink(fullKey, parentIndex, notify);
    };
  const convertTopLink = (link: Link): [Link] => {return [link];};

  await execute(
    getTopLink,
    [asTopicKey(0), 0 as AdjustedLineIndex],
    (link) => {
      assertTrue(!!link.valid);
      expect(link.parent).toEqual('a1');
      expect(link.child).toEqual('a2');
    },
    undefined,
  );
  await execute(
    getTopLink,
    [asTopicKey(1), 0 as AdjustedLineIndex],
    (link) => {
      assertTrue(!!link.valid);
      expect(link.parent).toEqual('a1');
      expect(link.child).toEqual('b2');
    },
    undefined,
  );
  await execute(
    getTopLink,
    [asTopicKey(1), 1 as AdjustedLineIndex],
    (link) => {
      assertTrue(!!link.valid);
      expect(link.parent).toEqual('b4');
      expect(link.child).toEqual('b2');
    },
    convertTopLink,
  );
  await execute(
    getTopLink,
    [asTopicKey(1), 2 as AdjustedLineIndex],
    (link) => {
      assertTrue(!link.valid);
    },
    convertTopLink,
  );
});
