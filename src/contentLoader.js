import GenericLoader from "./genericLoader";
import WeakValueMap from "./WeakValueMap";

const URL_PREFIX = `${window.location.origin}/api`;


function errHnd(e) {
  console.error(e);
}


function json(resp) {
  return resp.json();
}


export default class ContentLoader {
  constructor() {
    this.topics = null;
    this.msgs = new WeakValueMap(100);
    this.parentLines = new GenericLoader(10, this.fetchParentLine);
    this.childLines = new GenericLoader(10, this.fetchChildLine);
    this.fetchTopics();
  }

  fetchTopics() {
    fetch(`${URL_PREFIX}/topic`).then(json).then((obj) => {
      const { topics } = obj;
      this.topics = topics;
    }).catch(errHnd);
  }

  fetchParentLine = (name, offset, limit, resultCb) => {
    return this.fetchLine(true, name, offset, limit, resultCb);
  }

  fetchChildLine = (name, offset, limit, resultCb) => {
    return this.fetchLine(false, name, offset, limit, resultCb);
  }

  fetchMsg(objs, readyCb) {
    const mapping = {};
    objs.forEach((obj) => {
      mapping[obj.mhash] = obj;
    });
    fetch(`${URL_PREFIX}/read`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        "hashes": objs.map((obj => obj.mhash)),
      })
    }).then(json).then((obj) => {
      const { messages, skipped } = obj;
      Object.keys(messages).forEach((mhash) => {
        mapping[mhash].msg = messages[mhash];
        this.msgs.set(mhash, messages[mhash]);
      });
      if (skipped.length > 0) {
        this.fetchMsg(skipped.map(cur => mapping[cur]), readyCb);
      } else {
        readyCb();
      }
    }).catch(errHnd);
  }

  fetchLine(isGetParent, name, offset, limit, resultCb) {
    const isHash = name[0] !== '!';
    const lineNum = isHash? 0 : Math.floor(+name.slice(1) / 1000);
    const padResult = (curRes) => {
      [...Array(limit).keys()].forEach((ix) => {
        const pos = offset + ix;
        if (curRes[pos] === undefined) {
          curRes[pos] = {
            mhash: `!${pos + 1000 * lineNum}`,
            first: 0,
            msg: `no data ${pos}`,
            user: "nouser",
            votes: {},
          };
        }
      });
      return curRes;
    };
    if (!isHash) {
      if (lineNum === 0) {
        if (isGetParent) {
          resultCb(padResult({}));
        } else {
          if (this.topics === null) {
            setTimeout(() => {
              this.fetchLine(isGetParent, name, offset, limit, resultCb);
            }, 100);
          } else {
            const tres = Object.keys(this.topics).slice(offset, offset + limit);
            resultCb(padResult(tres.map((thash) => {
              return {
                mhash: thash,
                first: 0,
                msg: tres[thash],
                user: "nouser",
                votes: {},
              };
            })));
          }
        }
      } else {
        resultCb(padResult({}));
      }
      return;
    }
    const query = isGetParent ? {
      "child": name,
    } : {
      "parent": name,
    };
    fetch(`${URL_PREFIX}/${isGetParent ? 'parents' : 'children'}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...query,
        "offset": offset,
        "limit": limit,
        "scorer": "best",
      })
    }).then(json).then((obj) => {
      const { links, next } = obj;
      const res = {};
      links.forEach((link, ix) => {
        const mhash = isGetParent ? link.parent : link.child;
        res[ix + offset] = {
          mhash,
          first: link.first,
          msg: this.msgs.get(mhash),
          user: link.user,
          votes: link.votes,
        };
      });
      const finalize = () => {
        if (links.length < limit && next > 0 && links.length > 0) {
          getChildren(name, next, limit - links.length, (rec) => {
            resultCb(padResult({...res, ...rec}));
          });
        } else {
          resultCb(padResult(res));
        }
      };
      const objs = [];
      Object.keys(res).forEach((cur) => {
        if (cur.mhash[0] === '!' || cur.msg !== null) {
          return;
        }
        objs.push(cur);
      });
      if (objs.length > 0) {
        this.fetchMsg(objs, finalize);
      } else {
        finalize();
      }
    }).catch(errHnd);
  }

  getItem(isParent, name, index, contentCb, readyCb) {

  }

  getChild(lineName, cb) {

  }

  getParent(lineName, cb) {

  }

  getLinkInfo(
      parentLineName, childLineName, parentIndex, childIndex, readyCb) {
    return [
      {key: 'up', count: 2},
      {key: 'down', count: 1},
    ];
  }
} // ContentLoader
