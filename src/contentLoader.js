import WeakValueMap from "./WeakValueMap.js";

export default class ContentLoader {
  constructor(blockSize, loadCb) {
    this.blockSize = blockSize;
    this.loadCb = loadCb;
    this.lines = new WeakValueMap(10);
    this.activeLoads = new Set();
  }

  getLine(name) {
    let res = this.lines[name];
    if (!res) {
      res = new WeakValueMap(100);
      this.lines[name] = res;
    }
    return res;
  }

  get(name, index, contentCb, readyCb) {
    const line = this.getLine(name);
    const res = line.get(index);
    if (res) {
      return contentCb(true, res);
    }
    const block = Math.floor(index / this.blockSize);
    const blockName = `${name}-${block}`;
    if (!this.activeLoads.has(blockName)) {
      setTimeout(() => {
        const offset = block * this.blockSize;
        this.loadCb(name, offset, this.blockSize, (arr) => {
          Object.keys(arr).forEach((ix) => {
            line.set(ix, arr[ix]);
          });
          this.activeLoads.delete(blockName);
          readyCb();
        });
      }, 0);
      this.activeLoads.add(blockName);
    }
    return contentCb(false, null);
  }

  unloadLine(name) {
    if (this.lines[name]) {
      delete this.lines[name];
    }
  }
} // ContentLoader