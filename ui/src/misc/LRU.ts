import { strict as assert } from "node:assert";

class LRUEntry<K, V> {
  parent: LRU<K, V>;
  prev: LRUEntry<K, V>;
  next: LRUEntry<K, V>;
  key: K;
  item: V;

  constructor(parent: LRU<K, V>, key: K, item: V) {
    this.parent = parent;
    this.prev = this;
    this.next = this;
    this.key = key;
    this.item = item;
  }
} // LRUEntry

export default class LRU<K, V> {
  head: LRUEntry<K, V> | undefined;
  tail: LRUEntry<K, V> | undefined;
  objs: Map<K, LRUEntry<K, V>>;
  maxSize: number;

  constructor(maxSize: number) {
    this.maxSize = Math.max(maxSize, 3);
    this.head = undefined;
    this.tail = undefined;
    this.objs = new Map();
  }

  set(key: K, value: V): V | undefined {
    const entry = this.objs.get(key);
    if (entry !== undefined) {
      const res = entry.item;
      entry.item = value;
      return res;
    }
    const elem = new LRUEntry(this, key, value);
    this.objs.set(key, elem);
    const head = this.head;
    this.head = elem;
    if (head !== undefined) {
      elem.next = head;
      head.prev = elem;
    } else {
      assert.equal(this.tail, undefined);
      this.tail = elem;
    }
    const tail = this.tail;
    if (tail !== undefined && this.objs.size > this.maxSize) {
      assert.equal(tail.next, tail);
      assert.notEqual(tail.prev, tail);
      this.tail = tail.prev;
      tail.prev = tail;
      assert.equal(this.tail.next, tail);
      this.tail.next = this.tail;
      this.objs.delete(tail.key);
    }
    return undefined;
  }

  get(key: K): V | undefined {
    const entry = this.objs.get(key);
    if (entry === undefined) {
      return undefined;
    }
    assert.ok(this.head !== undefined);
    const head = this.head;
    assert.equal(head.prev, head);
    if (head !== entry) {
      const prev = entry.prev;
      assert.notEqual(prev, entry);
      assert.equal(prev.next, entry);
      const next = entry.next;
      if (next === entry) {
        assert.equal(this.tail, entry);
        this.tail = prev;
        prev.next = prev;
      } else {
        assert.equal(next.prev, entry);
        prev.next = next;
        next.prev = prev;
      }
      entry.prev = entry;
      entry.next = head;
      head.prev = entry;
      this.head = entry;
    }
    return entry.item;
  }

  keys(): K[] {
    let cur = this.head;
    if (cur === undefined) {
      return [];
    }
    const arr: K[] = [];
    while (cur.next !== cur) {
      arr.push(cur.key);
      cur = cur.next;
      assert.ok(arr.length <= this.objs.size);
    }
    arr.push(cur.key);
    return arr;
  }

  values(): V[] {
    let cur = this.head;
    if (cur === undefined) {
      return [];
    }
    const arr: V[] = [];
    while (cur.next !== cur) {
      arr.push(cur.item);
      cur = cur.next;
      assert.ok(arr.length <= this.objs.size);
    }
    arr.push(cur.item);
    return arr;
  }
} // LRU
