import { AssertionError } from 'assert';

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

export function assertTrue(value: boolean): asserts value {
  if (!value) {
    throw new AssertionError({ message: 'assertion was not true' });
  }
}

export function assertEqual<T>(
  actual: unknown,
  expected: T,
): asserts actual is T {
  if (actual !== expected) {
    throw new AssertionError({
      message: `${actual} !== ${expected}`,
      actual,
      expected,
    });
  }
}

export function assertNotEqual(actual: unknown, expected: unknown): void {
  if (actual === expected) {
    throw new AssertionError({
      message: `${actual} === ${expected}`,
      actual,
      expected,
    });
  }
}
