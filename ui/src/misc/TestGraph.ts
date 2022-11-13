import { ApiProvider, MHash, Votes } from './CommentGraph';
import { assertTrue, str } from './util';

export const simpleGraph = (): TestGraph => {
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
  return graph;
};

export const advancedGraph = (): TestGraph => {
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
  return graph;
};

export default class TestGraph {
  private readonly apiLimit: number;
  private readonly topics: string[];
  private readonly messages: { [key: string]: string };
  private readonly children: { [key: string]: string[] };
  private readonly parents: { [key: string]: string[] };

  constructor(apiLimit: number) {
    this.apiLimit = apiLimit;
    this.topics = [];
    this.messages = {};
    this.children = {};
    this.parents = {};
  }

  private addTopic(hash: string): void {
    this.addMessage(hash);
    this.topics.push(hash);
  }

  private addMessage(hash: string): void {
    assertTrue(!hash.startsWith('msg:'));
    this.messages[hash] = `msg: ${hash}`;
  }

  private addLink(from: string, to: string): void {
    this.addMessage(from);
    this.addMessage(to);
    if (this.children[from] === undefined) {
      this.children[from] = [];
    }
    this.children[from].push(to);
    if (this.parents[to] === undefined) {
      this.parents[to] = [];
    }
    this.parents[to].push(from);
  }

  addTopics(hashs: string[]): void {
    hashs.forEach((el) => this.addTopic(el));
  }

  addLinks(links: [string, string][]): void {
    links.forEach(([from, to]) => this.addLink(from, to));
  }

  getApiProvider(): ApiProvider {
    return {
      topic: async () => {
        const entries: [string, string][] = this.topics.map((el) => [
          el,
          this.messages[el],
        ]);
        return {
          topics: Object.fromEntries(entries),
        };
      },
      read: async (hashes) => {
        const ms: Readonly<MHash[]> = Array.from(hashes)
          .sort()
          .map((e) => e as MHash);
        const msgArr: string[] = ms.slice(0, this.apiLimit);
        const skipped: Readonly<MHash[]> = ms.slice(this.apiLimit, undefined);
        const messages: Readonly<{ [key: string]: string }> =
          Object.fromEntries(
            msgArr.map((el) => [el, this.messages[el] ?? '[missing]']),
          );
        return { messages, skipped };
      },
      link: async (linkKey, offset, limit) => {
        const { mhash, isGetParent } = linkKey;
        const map = isGetParent ? this.parents : this.children;
        const arr = map[mhash as MHash] ?? [];
        const ret = arr.slice(offset, offset + Math.min(limit, this.apiLimit));
        const endIx = offset + ret.length;
        const next = endIx === arr.length ? 0 : endIx;
        const links = ret.map((other) => ({
          parent: (isGetParent ? other : mhash) as MHash,
          child: (isGetParent ? mhash : other) as MHash,
          user: 'abc',
          first: 123,
          votes: { up: 1 },
        }));
        return { links, next };
      },
      singleLink: async (parent, child) => {
        const children = this.children[parent as MHash] ?? [];
        const exists = children.some((cur) => cur === str(child));
        const votes: Votes = exists ? { up: 1 } : {};
        return {
          parent,
          child,
          user: exists ? 'abc' : undefined,
          first: exists ? 123 : 999,
          votes,
        };
      },
    };
  }
} // TestGraph