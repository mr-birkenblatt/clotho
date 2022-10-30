import GenericLoader, { ResultCB, ReadyCB, ContentCB } from './GenericLoader';
import LRU from './LRU';
import { errHnd, json, range, union } from './util';

const URL_PREFIX = `${window.location.origin}/api`;

type ApiTopic = {
  topics: { [key: string]: string };
};

type ApiRead = {
  messages: { [key: string]: string };
  skipped: string[];
};

type LinkResponse = {
  parent: string;
  child: string;
  user: string | undefined;
  first: number;
  votes: Votes;
};

type ApiLinkList = {
  links: LinkResponse[];
  next: number;
};

export type Votes = { [key: string]: number };

export type Link = {
  // FIXME actually figure out the hashes
  // parent: string;
  // child: string;
  mhash: string;
  first: number;
  msg: string | undefined;
  user: string;
  votes: Votes;
};

export default class ContentLoader {
  topics: Map<string, string> | undefined;
  msgs: LRU<string, string>;
  parentLines: GenericLoader<Link>;
  childLines: GenericLoader<Link>;

  constructor() {
    this.topics = undefined;
    this.msgs = new LRU(100);
    this.parentLines = new GenericLoader(10, this.fetchParentLine);
    this.childLines = new GenericLoader(10, this.fetchChildLine);
    this.fetchTopics();
  }

  fetchTopics(): void {
    console.log('topic');
    fetch(`${URL_PREFIX}/topic`)
      .then(json)
      .then((obj: ApiTopic) => {
        const { topics } = obj;
        this.topics = new Map(Object.entries(topics));
        console.log('topic done');
      })
      .catch(errHnd);
  }

  fetchParentLine = (
    name: string,
    offset: number,
    limit: number,
    resultCb: ResultCB<Link>,
  ): void => {
    this.fetchLine(true, name, offset, limit, resultCb);
  };

  fetchChildLine = (
    name: string,
    offset: number,
    limit: number,
    resultCb: ResultCB<Link>,
  ): void => {
    this.fetchLine(false, name, offset, limit, resultCb);
  };

  fetchMsg(objs: Link[], readyCb: ReadyCB): void {
    const mapping = new Map<string, Link>();
    objs.forEach((obj) => {
      mapping.set(obj.mhash, obj);
    });
    console.log('read', objs);
    fetch(`${URL_PREFIX}/read`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        hashes: objs.map((obj) => obj.mhash),
      }),
    })
      .then(json)
      .then((obj: ApiRead) => {
        const { messages, skipped } = obj;
        Object.entries(messages).forEach((cur) => {
          const [mhash, content] = cur;
          const entry = mapping.get(mhash);
          if (entry !== undefined) {
            entry.msg = content;
          } else {
            console.warn(`${mhash} was not requested: ${content}`);
          }
          this.msgs.set(mhash, content);
        });
        if (skipped.length > 0) {
          this.fetchMsg(
            skipped.reduce((res: Link[], cur) => {
              const entry = mapping.get(cur);
              if (entry === undefined) {
                console.warn(
                  `skipped contains superfluous elements: ${skipped}`,
                );
              } else {
                res.push(entry);
              }
              return res;
            }, []),
            readyCb,
          );
        } else {
          console.log('read done', objs);
          readyCb();
        }
      })
      .catch(errHnd);
  }

  fetchLine(
    isGetParent: boolean,
    name: string,
    offset: number,
    limit: number,
    resultCb: ResultCB<Link>,
  ): void {
    const isHash = name[0] !== '!';
    const lineNum = isHash ? 0 : Math.floor(+name.slice(1) / 1000);
    const padResult = (curRes: Map<number, Link>) => {
      range(limit).forEach((ix) => {
        const pos = offset + ix;
        if (!curRes.has(pos)) {
          curRes.set(pos, {
            mhash: `!${pos + 1000 * lineNum}`,
            first: 0,
            msg: `no data ${pos}`,
            user: 'nouser',
            votes: {},
          });
        }
      });
      return curRes;
    };
    if (!isHash) {
      if (lineNum === 0) {
        if (isGetParent) {
          resultCb(padResult(new Map()));
        } else {
          if (this.topics === undefined) {
            setTimeout(() => {
              this.fetchLine(isGetParent, name, offset, limit, resultCb);
            }, 100);
          } else {
            const topics = this.topics;
            const tres = Array.from(topics.keys())
              .sort()
              .slice(offset, offset + limit);
            resultCb(
              padResult(
                new Map(
                  tres.map((thash, index) => {
                    return [
                      index,
                      {
                        mhash: thash,
                        first: 0,
                        msg: topics.get(thash),
                        user: 'nouser',
                        votes: {},
                      },
                    ];
                  }),
                ),
              ),
            );
          }
        }
      } else {
        resultCb(padResult(new Map()));
      }
      return;
    }
    const query = isGetParent
      ? {
          child: name,
        }
      : {
          parent: name,
        };
    console.log(isGetParent ? 'parents' : 'children', name, offset, limit);
    fetch(`${URL_PREFIX}/${isGetParent ? 'parents' : 'children'}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...query,
        offset,
        limit,
        scorer: 'best',
      }),
    })
      .then(json)
      .then((obj: ApiLinkList) => {
        const { links, next } = obj;
        const res = new Map<number, Link>(
          links.map((link, ix) => {
            const mhash = isGetParent ? link.parent : link.child;
            const res: Link = {
              mhash,
              first: link.first,
              msg: this.msgs.get(mhash),
              user: link.user || 'nouser',
              votes: link.votes,
            };
            return [ix + offset, res];
          }),
        );
        const finalize = () => {
          if (links.length < limit && next > 0 && links.length > 0) {
            this.fetchLine(
              isGetParent,
              name,
              next,
              limit - links.length,
              (rec: Map<number, Link>) => {
                resultCb(padResult(union(res, rec)));
              },
            );
          } else {
            resultCb(padResult(res));
          }
        };
        const objs: Link[] = [];
        Array.from(res.values()).forEach((cur) => {
          if (cur.mhash[0] === '!' || cur.msg !== undefined) {
            return;
          }
          objs.push(cur);
        });
        if (objs.length > 0) {
          this.fetchMsg(objs, finalize);
        } else {
          finalize();
        }
      })
      .catch(errHnd);
  }

  getItem<R>(
    isGetParent: boolean,
    name: string,
    index: number,
    contentCb: ContentCB<Link, R>,
    readyCb: ReadyCB,
  ): R {
    const loader = isGetParent ? this.parentLines : this.childLines;
    return loader.get(name, index, contentCb, readyCb);
  }

  getChildLine = (
    name: string,
    index: number,
    callback: (child: string) => void,
  ): void => {
    this.childLines.with(name, index, (link) => {
      callback(link.mhash);
    });
  };

  getParentLine = (
    name: string,
    index: number,
    callback: (parent: string) => void,
  ): void => {
    this.parentLines.with(name, index, (link) => {
      callback(link.mhash);
    });
  };

  getLink = (
    isGetParent: boolean,
    name: string,
    index: number,
    readyCb: ReadyCB,
  ): Link | undefined => {
    const loader = isGetParent ? this.parentLines : this.childLines;
    return loader.get(
      name,
      index,
      (ready, content) => (ready ? content : undefined),
      readyCb,
    );
  };
} // ContentLoader