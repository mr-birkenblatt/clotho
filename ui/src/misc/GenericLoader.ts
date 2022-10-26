import LRU from './LRU.js';

type ResultCB<V> = (arr: Map<number, V>) => void;
type ContentCB<V> = (ready: boolean, content: V | undefined) => void;
type ReadyCB = () => void;
type LoadCB<V> = (
    name: string,
    offset: number,
    size: number,
    resultCb: ResultCB<V>
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

    get(
        name: string,
        index: number,
        contentCb: ContentCB<V>,
        readyCb: ReadyCB
    ) {
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

    unloadLine(name: string) {
        this.lines.delete(name);
    }
} // GenericLoader
