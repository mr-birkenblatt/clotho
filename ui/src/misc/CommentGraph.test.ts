import CommentGraph, {
  AdjustedLineIndex,
  FullKey,
  MHash,
} from './CommentGraph';
import TestGraph from './TestGraph';
import { assertTrue } from './util';

jest.useFakeTimers();

function asTopicKey(index: number): FullKey {
  return {
    topic: true,
    index: index as AdjustedLineIndex,
  };
}

function asLinkKey(hash: string, isGetParent: boolean, index: number): FullKey {
  return {
    mhash: hash as MHash,
    isGetParent,
    index: index as AdjustedLineIndex,
  }
}

function expectWait(res: any): void {
  assertTrue(res === undefined);
}

test('simple test comment graph', () => {
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
  // TODO use timer api and count function calls
  expectWait(
    pool.getMessage(asTopicKey(0), (mhash, content) => {
      expect(mhash).toEqual('a');
      expect(content).toEqual('msg: a');
    }),
  );
  expectWait(
    pool.getMessage(asTopicKey(1), (mhash, content) => {
      expect(mhash).toEqual('h');
      expect(content).toEqual('msg: h');
    }),
  );
  expectWait(
    pool.getMessage(asTopicKey(-1), (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('[unavailable]');
    }),
  );
  expectWait(
    pool.getMessage(asTopicKey(2), (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('[unavailable]');
    }),
  );
  expectWait(
    pool.getMessage(asLinkKey('a', false, 0), (mhash, content) => {
      expect(mhash).toEqual('b');
      expect(content).toEqual('msg: ba');
    }),
  );
  expectWait(
    pool.getMessage(asLinkKey('a', false, 2), (mhash, content) => {
      expect(mhash).toEqual('d');
      expect(content).toEqual('msg: d');
    }),
  );
  expectWait(
    pool.getMessage(asLinkKey('a', false, 4), (mhash, content) => {
      expect(mhash).toEqual('f');
      expect(content).toEqual('msg: f');
    }),
  );
  expectWait(
    pool.getMessage(asLinkKey('a', false, 5), (mhash, content) => {
      expect(mhash).toBe(undefined);
      expect(content).toEqual('[unavailablee]');
    }),
  );
});
