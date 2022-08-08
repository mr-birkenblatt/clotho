export default class WeakValueMap {
  constructor(maxValues = 500) {
    this.obj = {};
    this.times = {};
    this.maxValues = maxValues;
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
    const excess = Object.keys(this.obj).length - this.maxValues;
    if (excess > 0) {
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
