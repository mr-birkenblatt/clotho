import { ApiProvider, Link, MHash } from './CommentGraph';
import { assertTrue } from './util';

export function asMHashSet(arr: string[]): Set<MHash> {
  return new Set(arr as MHash[]);
}

export function asLinkKey(hash: string, isGetParent: boolean): {mhash: MHash, isGetParent: boolean} {
  return {mhash: hash as MHash, isGetParent};
}

type TestLink = {
  parent: Readonly<MHash>;
  child: Readonly<MHash>;
  user: Readonly<string> | undefined;
  first: Readonly<number>;
  votes: {[key: string]: number};
};

export function getParentHashs(links: readonly TestLink[]): string[] {
  return links.map(l => (l.parent as unknown) as string);
}

export function getChildHashs(links: readonly TestLink[]): string[] {
  return links.map(l => (l.child as unknown) as string);
}

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
          Object.fromEntries(msgArr.map((el) => [el, this.messages[el]]));
        return { messages, skipped };
      },
      link: async (linkKey, offset, limit) => {
        const { mhash, isGetParent } = linkKey;
        const map = isGetParent ? this.parents : this.children;
        const arr = map[mhash as MHash];
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
    };
  }
} // TestGraph
