export default class WeakValueMap {
  constructor(maxValues = 500, softLimit = undefined) {
    this.obj = {};
    this.times = {};
    this.maxValues = maxValues;
    this.softLimit = softLimit !== undefined
      ? softLimit : Math.max(1, Math.floor(maxValues * 0.9));
    if (this.softLimit > this.maxValues) {
      throw Error(`softLimit=${this.softLimit} > maxValues=${this.maxValues}`);
    }
  }

  now() {
    return performance.now()
  }

  get(key) {
    const value = this.obj[key];
    if (value) {
      this.times[key] = this.now();
    }
    return value;
  }

  set(key, value) {
    const len = Object.keys(this.obj).length;
    if (len > this.maxValues) {
      const excess = len - this.softLimit;
      Object.keys(this.obj).sort((a, b) => {
        return this.times[a] - this.times[b];
      }).slice(0, excess).forEach((key) => {
        delete this.obj[key];
        delete this.times[key];
        // console.log(`remove ${key}`);
      });
    }
    this.obj[key] = value;
    this.times[key] = this.now();
  }

  update(key, cb) {
    this.set(key, cb(this.get(key)));
  }
} // WeakValueMap
