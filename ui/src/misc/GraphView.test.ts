import CommentGraph, {
  asFullKey,
  asTopicKey,
  FullKey,
  INVALID_FULL_KEY,
  INVALID_LINK,
  MHash,
} from './CommentGraph';
import {
  Cell,
  consistentLinks,
  equalView,
  GraphView,
  progressView,
  scrollBottomHorizontal,
  scrollTopHorizontal,
} from './GraphView';
import { advancedGraph } from './TestGraph';
import { assertFail, assertTrue, safeStringify } from './util';

async function execute(
  graph: CommentGraph,
  view: Readonly<GraphView> | undefined,
  expected: Readonly<GraphView>,
  expectedTransitions: number,
): Promise<Readonly<GraphView>> {
  assertTrue(view !== undefined);
  const marker = jest.fn();
  const transition: (
    view: Readonly<GraphView>,
    resolve: (
      value: Readonly<GraphView> | PromiseLike<Readonly<GraphView>>,
    ) => void,
    reject: (reason: any) => void,
  ) => void = (view, resolve, reject) => {
    const res = progressView(graph, view, (newView) => {
      marker();
      transition(newView, resolve, reject);
    });
    if (res !== undefined) {
      expect(marker).toBeCalledTimes(expectedTransitions);
      if (equalView(view, expected) && consistentLinks(view, console.warn)) {
        resolve(res);
      } else {
        reject(`${safeStringify(view)} !== ${safeStringify(expected)}`);
      }
    }
  };

  return new Promise((resolve, reject) => {
    transition(view, resolve, reject);
  });
}

function cellFromString(key: string, fullKey: FullKey): Cell {
  return {
    fullKey,
    mhash: key as MHash,
    content: `msg: ${key}`,
  };
}

function invalidCell(): Cell {
  return {
    invalid: true,
    fullKey: INVALID_FULL_KEY,
    content: '[invalid]',
    topLink: INVALID_LINK,
  };
}

function buildFullView(
  top: [string, FullKey],
  middleTop: [
    [string, FullKey] | undefined,
    [string, FullKey],
    [string, FullKey] | undefined,
  ],
  middleBottom: [
    [string, FullKey] | undefined,
    [string, FullKey],
    [string, FullKey] | undefined,
  ],
  bottom: [string, FullKey],
): Readonly<GraphView> {
  return {
    top: cellFromString(...top),
    topLeft:
      middleTop[0] !== undefined
        ? cellFromString(...middleTop[0])
        : invalidCell(),
    centerTop: cellFromString(...middleTop[1]),
    topRight:
      middleTop[2] !== undefined
        ? cellFromString(...middleTop[2])
        : invalidCell(),
    bottomLeft:
      middleBottom[0] !== undefined
        ? cellFromString(...middleBottom[0])
        : invalidCell(),
    centerBottom: cellFromString(...middleBottom[1]),
    bottomRight:
      middleBottom[2] !== undefined
        ? cellFromString(...middleBottom[2])
        : invalidCell(),
    bottom: cellFromString(...bottom),
  };
}

function asCell(fullKey: Readonly<FullKey>): Readonly<Cell> {
  return { fullKey };
}

test('test graph view init', async () => {
  const graph = new CommentGraph(advancedGraph().getApiProvider());

  const initGraph = await execute(
    graph,
    { centerTop: asCell(asTopicKey(0)) },
    buildFullView(
      ['a1', asFullKey('a2', true, 0)],
      [undefined, ['a2', asTopicKey(0)], ['b2', asTopicKey(1)]],
      [undefined, ['a3', asFullKey('a2', false, 0)], undefined],
      ['a4', asFullKey('a3', false, 0)],
    ),
    13,
  );
  assertTrue(initGraph !== undefined);

  assertTrue(
    equalView(
      progressView(graph, { centerTop: invalidCell() }, (_) => {
        assertFail('no progress should happen');
      }),
      { centerTop: invalidCell() },
    ),
  );

  assertTrue(scrollTopHorizontal(initGraph, false) === undefined);
  assertTrue(scrollBottomHorizontal(initGraph, false) === undefined);
  assertTrue(scrollBottomHorizontal(initGraph, true) === undefined);

  await execute(
    graph,
    scrollTopHorizontal(initGraph, true),
    buildFullView(
      ['a1', asFullKey('a2', true, 0)],
      [
        ['a2', asTopicKey(0)],
        ['b2', asTopicKey(1)],
        ['b2', asTopicKey(2)],
      ],
      [undefined, ['a3', asFullKey('a2', false, 0)], undefined],
      ['a4', asFullKey('a3', false, 0)],
    ),
    1,
  );
  // ['a1', 'a2'],
  // ['a1', 'b2'],
  // ['a1', 'c2'],
  // ['a1', 'd2'],
  // ['a2', 'a3'],
  // ['a3', 'a4'],
  // ['a3', 'b4'],
  // ['a4', 'a5'],
  // ['a5', 'a1'],
  // ['b4', 'b2'],
  // ['b2', 'b4'],
  // addTopics(['a2', 'b2'])
});
