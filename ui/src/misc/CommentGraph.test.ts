import CommentGraph, { AdjustedLineIndex, FullKey, MHash } from './CommentGraph';
import TestGraph from './TestGraph';
import { assertFail, assertTrue } from './util';

function asTopicKey(index: number): FullKey {
  return {
    topic: true,
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
  expectWait(pool.getMessage(asTopicKey(0), (mhash, content) => {expect(mhash).toEqual('a'); expect(content).toEqual('msg: a')}));
  expectWait(pool.getMessage(asTopicKey(1), (mhash, content) => {expect(mhash).toEqual('h'); expect(content).toEqual('msg: h')}));
  expectWait(pool.getMessage(asTopicKey(2), (mhash, content) => {expect(mhash).toBe(undefined); expect(content).toEqual('[unavailable]')}));
});
