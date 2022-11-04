import CommentGraph, {
  AdjustedLineIndex,
  FullKey,
  MHash,
} from './CommentGraph';
import TestGraph from './TestGraph';
import { assertTrue } from './util';

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

type Callback = (key: Readonly<MHash> | undefined, value: string) => void;

async function execute(
  expectWait: boolean,
  pool: CommentGraph,
  fun: (fullKey: Readonly<FullKey>, notify: Callback) => string | undefined,
  fullKey: FullKey,
  callback: Callback,
): Promise<boolean> {
  const marker = jest.fn();
  return new Promise((resolve) => {
    const cb: Callback = (key, value) => {
      callback(key, value);
      resolve(true);
    };
    const notify: Callback = (key, value) => {
      marker();
      if (expectWait) {
        cb(key, value);
      }
    };
    const res = fun.call(pool, fullKey, notify);
    expect(marker).not.toBeCalled();
    if (expectWait) {
      assertTrue(res === undefined);
    } else {
      assertTrue(res !== undefined);
      cb(undefined, res);
    }
    // console.log('runAllTimers');
    // jest.runAllTimers();
  }).then(() => {
    if (expectWait) {
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
  await execute(
    true,
    pool,
    pool.getMessage,
    asTopicKey(0),
    (mhash, content) => {
      expect(mhash).toEqual('a');
      expect(content).toEqual('msg: a');
    },
  );
  await execute(
    false,
    pool,
    pool.getMessage,
    asTopicKey(1),
    (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('msg: h');
    },
  );
  await execute(
    false,
    pool,
    pool.getMessage,
    asTopicKey(-1),
    (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('[unavailable]');
    },
  );
  await execute(
    false,
    pool,
    pool.getMessage,
    asTopicKey(2),
    (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('[unavailable]');
    },
  );
  await execute(
    true,
    pool,
    pool.getMessage,
    asLinkKey('a', false, 0),
    (mhash, content) => {
      expect(mhash).toEqual('b');
      expect(content).toEqual('msg: b');
    },
  );
  await execute(
    true,
    pool,
    pool.getMessage,
    asLinkKey('a', false, 2),
    (mhash, content) => {
      expect(mhash).toEqual('d');
      expect(content).toEqual('msg: d');
    },
  );
  await execute(
    false,
    pool,
    pool.getMessage,
    asLinkKey('a', false, 2),
    (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('msg: d');
    },
  );
  await execute(
    true,
    pool,
    pool.getMessage,
    asLinkKey('a', false, 4),
    (mhash, content) => {
      expect(mhash).toEqual('f');
      expect(content).toEqual('msg: f');
    },
  );
  await execute(
    false,
    pool,
    pool.getMessage,
    asLinkKey('a', false, 5),
    (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('[deleted]');
    },
  );
});
