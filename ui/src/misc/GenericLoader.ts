import LRU from './LRU';
import { assertTrue } from './util';

export type ResultCB<V> = (arr: Map<number, V>) => void;
export type ContentCB<V, R> = (ready: boolean, content: V | undefined) => R;
export type ValueCB<V> = (content: V) => void;
export type ReadyCB = () => void;
export type ItemCB<V, R> = (
  isGetParent: boolean,
  name: string,
  index: number,
  contentCb: ContentCB<V, R>,
  readyCb: ReadyCB,
) => R;
type LoadCB<V> = (
  name: string,
  offset: number,
  size: number,
  resultCb: ResultCB<V>,
) => void;

export default class GenericLoader<V> {
  blockSize: number;
  loadCb: LoadCB<V>;
  lines: LRU<string, LRU<number, V>>;
  activeLoads: Set<string>;

  constructor(blockSize: number, loadCb: LoadCB<V>) {
    this.blockSize = blockSize;
    this.loadCb = loadCb;
    this.lines = new LRU(10);
    this.activeLoads = new Set<string>();
  }

  getLine(name: string): LRU<number, V> {
    let res = this.lines.get(name);
    if (res === undefined) {
      res = new LRU<number, V>(100);
      this.lines.set(name, res);
    }
    return res;
  }

  get<R>(
    name: string,
    index: number,
    contentCb: ContentCB<V, R>,
    readyCb: ReadyCB,
  ): R {
    const line = this.getLine(name);
    const res = line.get(index);
    if (res !== undefined) {
      return contentCb(true, res);
    }
    const block = Math.floor(index / this.blockSize);
    const blockName = `${name}-${block}`;
    if (!this.activeLoads.has(blockName)) {
      setTimeout(() => {
        const offset = block * this.blockSize;
        this.loadCb(name, offset, this.blockSize, (arr) => {
          arr.forEach((v, ix) => {
            line.set(ix, v);
          });
          this.activeLoads.delete(blockName);
          readyCb();
        });
      }, 0);
      this.activeLoads.add(blockName);
    }
    return contentCb(false, undefined);
  }

  with(name: string, index: number, valueCb: ValueCB<V>) {
    const line = this.getLine(name);
    const res = line.get(index);
    if (res !== undefined) {
      valueCb(res);
      return;
    }
    const block = Math.floor(index / this.blockSize);
    const blockName = `${name}-${block}`;
    if (!this.activeLoads.has(blockName)) {
      setTimeout(() => {
        const offset = block * this.blockSize;
        this.loadCb(name, offset, this.blockSize, (arr) => {
          arr.forEach((v, ix) => {
            line.set(ix, v);
          });
          this.activeLoads.delete(blockName);
          const res = line.get(index);
          assertTrue(res !== undefined);
          valueCb(res);
        });
      }, 0);
      this.activeLoads.add(blockName);
    }
  }

  unloadLine(name: string): void {
    this.lines.delete(name);
  }
} // GenericLoader
