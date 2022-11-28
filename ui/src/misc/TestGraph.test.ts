import {
  fromMHash,
  IsGet,
  KeyType,
  MHash,
  UserId,
  Votes,
} from './CommentGraph';
import { simpleGraph } from './TestGraph';

function asMHashSet(arr: string[]): Set<MHash> {
  return new Set(arr as MHash[]);
}

function asLinkKey(
  hash: string,
  isGetParent: boolean,
): {
  keyType: KeyType.link;
  mhash: Readonly<MHash>;
  isGet: Readonly<IsGet>;
} {
  return {
    keyType: KeyType.link,
    mhash: hash as MHash,
    isGet: isGetParent ? IsGet.parent : IsGet.child,
  };
}

type TestLink = {
  parent: Readonly<MHash>;
  child: Readonly<MHash>;
  user: Readonly<UserId> | undefined;
  first: Readonly<number>;
  votes: Votes;
};

function getParentHashs(links: readonly TestLink[]): string[] {
  return links.map((l) => fromMHash(l.parent));
}

function getChildHashs(links: readonly TestLink[]): string[] {
  return links.map((l) => fromMHash(l.child));
}

test('test messages', () => {
  const api = simpleGraph().getApiProvider();
  api.topic(0, 3).then((topics) => {
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
  api.read(asMHashSet(['foo'])).then((read) => {
    expect(read.messages).toEqual({ foo: '[missing]' });
    expect(read.skipped).toEqual([]);
  });
});

test('test links', () => {
  const api = simpleGraph().getApiProvider();
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
  api.link(asLinkKey('foo', true), 0, 5).then((link) => {
    expect(getParentHashs(link.links)).toEqual([]);
    expect(getChildHashs(link.links)).toEqual([]);
    expect(link.next).toBe(0);
  });
});
