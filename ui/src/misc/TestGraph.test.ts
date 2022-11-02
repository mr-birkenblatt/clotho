import { MHash } from './CommentGraph';
import TestGraph from './TestGraph';

function asMHashSet(arr: string[]): Set<MHash> {
  return new Set(arr as MHash[]);
}

function asLinkKey(
  hash: string,
  isGetParent: boolean,
): { mhash: MHash; isGetParent: boolean } {
  return { mhash: hash as MHash, isGetParent };
}

type TestLink = {
  parent: Readonly<MHash>;
  child: Readonly<MHash>;
  user: Readonly<string> | undefined;
  first: Readonly<number>;
  votes: { [key: string]: number };
};

function getParentHashs(links: readonly TestLink[]): string[] {
  return links.map((l) => l.parent as unknown as string);
}

function getChildHashs(links: readonly TestLink[]): string[] {
  return links.map((l) => l.child as unknown as string);
}

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
  api.read(asMHashSet(['a', 'b', 'c', 'd'])).then((read) => {
    expect(read.messages).toEqual({ a: 'msg: a', b: 'msg: b', c: 'msg: c' });
    expect(read.skipped).toEqual(['d']);
  });
  api.read(asMHashSet(['a', 'b', 'c'])).then((read) => {
    expect(read.messages).toEqual({ a: 'msg: a', b: 'msg: b', c: 'msg: c' });
    expect(read.skipped).toEqual([]);
  });
  api.read(asMHashSet(['d'])).then((read) => {
    expect(read.messages).toEqual({ d: 'msg: d' });
    expect(read.skipped).toEqual([]);
  });
  api.link(asLinkKey('a', false), 0, 5).then((link) => {
    expect(getParentHashs(link.links)).toEqual(['a', 'a', 'a']);
    expect(getChildHashs(link.links)).toEqual(['b', 'c', 'd']);
    expect(link.next).toBe(3);
  });
  api.link(asLinkKey('a', false), 3, 5).then((link) => {
    expect(getParentHashs(link.links)).toEqual(['a', 'a']);
    expect(getChildHashs(link.links)).toEqual(['e', 'f']);
    expect(link.next).toBe(0);
  });
  api.link(asLinkKey('g', false), 0, 1).then((link) => {
    expect(getParentHashs(link.links)).toEqual(['g']);
    expect(getChildHashs(link.links)).toEqual(['a']);
    expect(link.next).toBe(1);
  });
  api.link(asLinkKey('g', false), 1, 1).then((link) => {
    expect(getParentHashs(link.links)).toEqual(['g']);
    expect(getChildHashs(link.links)).toEqual(['b']);
    expect(link.next).toBe(0);
  });
  api.link(asLinkKey('a', true), 0, 5).then((link) => {
    expect(getParentHashs(link.links)).toEqual(['g']);
    expect(getChildHashs(link.links)).toEqual(['a']);
    expect(link.next).toBe(0);
  });
  api.link(asLinkKey('b', true), 0, 5).then((link) => {
    expect(getParentHashs(link.links)).toEqual(['a', 'g']);
    expect(getChildHashs(link.links)).toEqual(['b', 'b']);
    expect(link.next).toBe(0);
  });
});
