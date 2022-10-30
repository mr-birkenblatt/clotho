export function range(from: number, to?: number): number[] {
  if (to === undefined) {
    to = from;
    from = 0;
  }
  return Array.from(Array(to - from).keys()).map((cur) => from + cur);
}

export function union<K, V>(left: Map<K, V>, right: Map<K, V>): Map<K, V> {
  return new Map(
    Array.from(left.entries()).concat(Array.from(right.entries())),
  );
}

export function errHnd(e: any): void {
  console.error(e);
}

export function json(resp: Response): Promise<any> {
  return resp.json();
}

export function toJson(obj: any): string {
  return JSON.stringify(obj, (_key, value) => {
    if (value instanceof Set) {
      value = Array.from(value.keys());
    }
    return value;
  });
}

export function assertTrue(value: boolean): asserts value {
  if (!value) {
    throw new Error('assertion was not true');
  }
}

export function assertEqual<T>(
  actual: unknown,
  expected: T,
): asserts actual is T {
  if (actual !== expected) {
    throw new Error(`actual:${actual} !== expected:${expected}`);
  }
}

export function assertNotEqual(actual: unknown, expected: unknown): void {
  if (actual === expected) {
    throw new Error(`actual:${actual} === expected:${expected}`);
  }
}

function safeStringify(obj: any): string {
  return JSON.stringify(
    obj,
    (_k, value) => {
      if (
        value instanceof Object &&
        !(value instanceof Array) &&
        !(value instanceof Date) &&
        !(value instanceof Function)
      ) {
        value = Object.fromEntries(
          Object.keys(value)
            .sort()
            .map((k) => [k, value[k]]),
        );
      }
      return value;
    },
    '',
  );
}

export class SafeMap<K, V> {
  private readonly mapValues: Map<string, V>;
  private readonly mapKeys: Map<string, K>;

  constructor(
    entries?: Iterable<readonly [K, V]> | ArrayLike<readonly [K, V]>,
  ) {
    if (entries !== undefined) {
      const es: (readonly [K, V])[] = Array.from(entries);
      this.mapValues = new Map(es.map((e) => [this.key(e[0]), e[1]]));
      this.mapKeys = new Map(es.map((e) => [this.key(e[0]), e[0]]));
    } else {
      this.mapValues = new Map();
      this.mapKeys = new Map();
    }
  }

  private key(key: Readonly<K>): string {
    return safeStringify(key);
  }

  clear(): void {
    this.mapValues.clear();
    this.mapKeys.clear();
  }

  delete(key: Readonly<K>): boolean {
    const k = this.key(key);
    this.mapKeys.delete(k);
    return this.mapValues.delete(k);
  }

  forEach(
    callbackfn: (value: V, key: K, map: this) => void,
    thisArg?: any,
  ): void {
    this.mapValues.forEach((value, key) => {
      const k = this.mapKeys.get(key);
      assertTrue(k !== undefined);
      callbackfn.call(thisArg, value, k, this);
    }, this);
  }

  get(key: Readonly<K>): V | undefined {
    const k = this.key(key);
    return this.mapValues.get(k);
  }

  has(key: Readonly<K>): boolean {
    const k = this.key(key);
    return this.mapValues.has(k);
  }

  set(key: Readonly<K>, value: V): this {
    const k = this.key(key);
    this.mapKeys.set(k, key);
    this.mapValues.set(k, value);
    return this;
  }

  get size(): number {
    return this.mapValues.size;
  }

  keys(): Iterator<K> {
    return this.mapKeys.values();
  }

  values(): Iterator<V> {
    return this.mapValues.values();
  }

  entries(): Iterator<[K, V]> {
    const res: [K, V][] = Array.from(this.mapValues.entries()).map((entry) => {
      const [key, value] = entry;
      const k = this.mapKeys.get(key);
      assertTrue(k !== undefined);
      return [k, value];
    });
    return res.values();
  }
} // SafeMap

export class SafeSet<V> {
  private readonly setValues: Map<string, V>;

  constructor(entries?: Iterable<V> | ArrayLike<V>) {
    if (entries !== undefined) {
      const es: V[] = Array.from(entries);
      this.setValues = new Map(es.map((e) => [this.key(e), e]));
    } else {
      this.setValues = new Map();
    }
  }

  private key(value: Readonly<V>): string {
    return safeStringify(value);
  }

  clear(): void {
    this.setValues.clear();
  }

  delete(value: Readonly<V>): boolean {
    const v = this.key(value);
    return this.setValues.delete(v);
  }

  forEach(
    callbackfn: (value: V, value2: V, set: this) => void,
    thisArg?: any,
  ): void {
    this.setValues.forEach((value) => {
      callbackfn.call(thisArg, value, value, this);
    }, this);
  }

  has(value: Readonly<V>): boolean {
    const v = this.key(value);
    return this.setValues.has(v);
  }

  add(value: V): this {
    const v = this.key(value);
    this.setValues.set(v, value);
    return this;
  }

  get size(): number {
    return this.setValues.size;
  }

  values(): Iterator<V> {
    return this.setValues.values();
  }
} // SafeSet