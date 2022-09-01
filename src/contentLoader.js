import GenericLoader from "./genericLoader";
import WeakValueMap from "./WeakValueMap";

const URL_PREFIX = `${window.location.origin}/api`;


function errHnd(e) {
  console.error(e);
}


function json(resp) {
  return resp.json();
}


// function getChild(name, cb) {
//   console.log("child", name);
//   fetch(`${URL_PREFIX}/children`, {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//     },
//     body: JSON.stringify({
//       "parent": name,
//       "offset": 0,
//       "limit": 1,
//       "scorer": "best",
//     })
//   }).then((resp) => resp.json()).then((obj) => {
//     const { links } = obj;
//     cb(links[0].child);
//   }).catch((e) => {
//     console.error(e);
//   });
// }


// function getParent(name, cb) {
//   console.log("parent", name);
//   fetch(`${URL_PREFIX}/parents`, {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//     },
//     body: JSON.stringify({
//       "child": name,
//       "offset": 0,
//       "limit": 1,
//       "scorer": "best",
//     })
//   }).then((resp) => resp.json()).then((obj) => {
//     const { links } = obj;
//     cb(links[0].parent);
//   }).catch((e) => {
//     console.error(e);
//   });
// }


// function getChildren(name, offset, limit, cb) {
//   fetch(`${URL_PREFIX}/children`, {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//     },
//     body: JSON.stringify({
//       "parent": name,
//       "offset": offset,
//       "limit": limit,
//       "scorer": "best",
//     })
//   }).then((resp) => resp.json()).then((obj) => {
//     const { links, next } = obj;
//     const res = {};
//     links.forEach((link, ix) => {
//       res[ix + offset] = `**name**: ${link.child} _ix_: ${ix + offset}`;
//     });
//     if (links.length < limit) {
//       if (next > 0 && links.length > 0) {
//         getChildren(name, next, limit - links.length, (rec) => {
//           cb({...res, ...rec});
//         });
//         return;
//       } else {
//         [...Array(limit - links.length).keys()].forEach((ix) => {
//           const pos = offset + links.length + ix;
//           res[pos] = `no data ${pos}`;
//         });
//       }
//     }
//     cb(res);
//   }).catch((e) => {
//     console.error(e);
//   });
// }


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

  fetchLine(isGetParent, name, offset, limit, resultCb) {
    const isHash = name[0] !== '!';
    const lineNum = isHash? 0 : +name.slice(1);
    const padResult = (curRes) => {
      [...Array(limit).keys()].forEach((ix) => {
        const pos = offset + ix;
        if (curRes[pos] === undefined) {
          curRes[pos] = {
            child: isGetParent ? name : `!${pos + lineNum}`,
            first: 0,
            parent: isGetParent ? `!${pos + lineNum}` : name,
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
                child: thash,
                first: 0,
                parent: name,
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

  }

  //// function getParents(name, offset, limit, cb) {
//   fetch(`${URL_PREFIX}/parents`, {
//     method: 'POST',
//     headers: {
//       'Content-Type': 'application/json',
//     },
//     body: JSON.stringify({
//       "child": name,
//       "offset": offset,
//       "limit": limit,
//       "scorer": "best",
//     })
//   }).then((resp) => resp.json()).then((obj) => {
//     const { links, next } = obj;
//     const res = {};
//     links.forEach((link, ix) => {
//       res[offset + ix] = `**name**: ${link.parent} _ix_: ${ix + offset}`;
//     });
//     if (links.length < limit) {
//       if (next > 0 && links.length > 0) {
//         getParents(name, next, limit - links.length, (rec) => {
//           cb({...res, ...rec});
//         });
//         return;
//       } else {
//         [...Array(limit - links.length).keys()].forEach((ix) => {
//           const pos = offset + links.length + ix;
//           res[pos] = `no data ${pos}`;
//         });
//       }
//     }
//     cb(res);
//   }).catch((e) => {
//     console.error(e);
//   });
// }

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
