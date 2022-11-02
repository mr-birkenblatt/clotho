import TestGraph, { asMHashSet } from './TestGraph';

test('test graph', () => {
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
  const api = graph.getApiProvider();
  api.topic().then((topics) => {
    expect(topics.topics).toEqual({ a: 'msg: a', h: 'msg: h' });
  });
  console.log(graph);
  api.read(asMHashSet(['a', 'b', 'c', 'd'])).then((read) => {
    expect(read.messages).toEqual({ a: 'msg: a', b: 'msg: b', c: 'msg: c' });
    expect(read.skipped).toEqual(['d']);
  });
});
