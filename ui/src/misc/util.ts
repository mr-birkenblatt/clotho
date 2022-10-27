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
