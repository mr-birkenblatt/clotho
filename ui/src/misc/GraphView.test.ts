import CommentGraph, {
  adj,
  asDirectKey,
  asTopicKey,
  FullIndirectKey,
  FullKey,
  FullKeyType,
  INVALID_FULL_KEY,
  INVALID_LINK,
  IsGet,
  MHash,
} from './CommentGraph';
import {
  Cell,
  consistentLinks,
  equalView,
  GraphView,
  initView,
  progressView,
  scrollBottomHorizontal,
  scrollTopHorizontal,
  scrollVertical,
} from './GraphView';
import { advancedGraph, InfGraph } from './TestGraph';
import { assertFail, assertTrue, LoggerCB, safeStringify } from './util';

function asFullKey(
  hash: Readonly<string>,
  isGetParent: boolean,
  index: number,
): Readonly<FullIndirectKey> {
  return {
    fullKeyType: FullKeyType.link,
    mhash: hash as MHash,
    isGet: isGetParent ? IsGet.parent : IsGet.child,
    index: adj(index),
  };
}

async function execute(
  graph: CommentGraph,
  view: Readonly<GraphView> | undefined,
  expected: Readonly<GraphView>,
  expectedTransitions: number,
  logger?: LoggerCB,
): Promise<Readonly<GraphView>> {
  assertTrue(view !== undefined, 'view is not set');
  let transitionCount = 0;
  const transition: (
    view: Readonly<GraphView>,
    resolve: (
      value: Readonly<GraphView> | PromiseLike<Readonly<GraphView>>,
    ) => void,
    reject: (reason: any) => void,
  ) => void = (view, resolve, reject) => {
    const res = progressView(
      graph,
      view,
      (newView) => {
        transitionCount += 1;
        transition(newView, resolve, reject);
      },
      logger,
    );
    if (res !== undefined) {
      if (transitionCount !== expectedTransitions) {
        reject(`${transitionCount} !== expected ${expectedTransitions}`);
      }
      if (
        equalView(view, expected, console.warn) &&
        consistentLinks(view, console.warn)
      ) {
        resolve(res);
      } else {
        reject(
          `${safeStringify(view)} !== expected ${safeStringify(expected)}`,
        );
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
  bottom: [string, FullKey] | undefined,
  topSkip: string | undefined,
  bottomSkip: string | undefined,
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
    bottom: bottom !== undefined ? cellFromString(...bottom) : invalidCell(),
    topSkip: topSkip !== undefined ? (topSkip as MHash) : undefined,
    bottomSkip: bottomSkip !== undefined ? (bottomSkip as MHash) : undefined,
  };
}

test('test graph view init', async () => {
  const graph = new CommentGraph(advancedGraph().getApiProvider(), {
    maxCommentPoolSize: 100,
    maxTopicSize: 100,
    maxLinkPoolSize: 100,
    maxLinkCache: 100,
    maxLineSize: 100,
    maxUserCache: 100,
    maxUserLineSize: 100,
    blockSize: 10,
  });

  const initGraph = await execute(
    graph,
    initView(undefined),
    buildFullView(
      ['a1', asFullKey('a2', true, 0)],
      [undefined, ['a2', asTopicKey(0)], ['b2', asTopicKey(1)]],
      [undefined, ['a3', asFullKey('a2', false, 0)], undefined],
      ['a4', asFullKey('a3', false, 0)],
      undefined,
      undefined,
    ),
    13,
  );
  assertTrue(initGraph !== undefined, 'initGraph is undefined');

  assertTrue(
    equalView(
      progressView(graph, { centerTop: invalidCell() }, (_) => {
        assertFail('no progress should happen');
      }),
      { centerTop: invalidCell() },
    ),
    'should be invalid',
  );

  assertTrue(
    scrollTopHorizontal(initGraph, false) === undefined,
    `${safeStringify(scrollTopHorizontal(initGraph, false))}`,
  );
  assertTrue(
    scrollBottomHorizontal(initGraph, false) === undefined,
    `${safeStringify(scrollBottomHorizontal(initGraph, false))}`,
  );
  assertTrue(
    scrollBottomHorizontal(initGraph, true) === undefined,
    `${safeStringify(scrollBottomHorizontal(initGraph, true))}`,
  );

  await execute(
    graph,
    scrollTopHorizontal(initGraph, true),
    buildFullView(
      ['a1', asFullKey('b2', true, 0)],
      [['a2', asTopicKey(0)], ['b2', asTopicKey(1)], undefined],
      [
        undefined,
        ['a3', asDirectKey('a3')],
        ['b4', asFullKey('b2', false, 0)],
      ],
      ['a4', asFullKey('a3', false, 0)],
      undefined,
      'a3',
    ),
    8,
  );

  const a1Graph = await execute(
    graph,
    initView('a1' as MHash),
    buildFullView(
      ['a5', asFullKey('a1', true, 0)],
      [undefined, ['a1', asDirectKey('a1')], undefined],
      [
        undefined,
        ['a2', asFullKey('a1', false, 0)],
        ['b2', asFullKey('a1', false, 1)],
      ],
      ['a3', asFullKey('a2', false, 0)],
      'a1',
      undefined,
    ),
    13,
  );

  assertTrue(a1Graph !== undefined, 'a1Graph is undefined');
  assertTrue(
    scrollTopHorizontal(a1Graph, false) === undefined,
    `${safeStringify(scrollTopHorizontal(a1Graph, false))}`,
  );
  assertTrue(
    scrollBottomHorizontal(a1Graph, false) === undefined,
    `${safeStringify(scrollBottomHorizontal(a1Graph, false))}`,
  );

  const a1BRGraph = await execute(
    graph,
    scrollBottomHorizontal(a1Graph, true),
    buildFullView(
      ['a5', asFullKey('a1', true, 0)],
      [undefined, ['a1', asDirectKey('a1')], ['b4', asFullKey('b2', true, 1)]],
      [
        ['a2', asFullKey('a1', false, 0)],
        ['b2', asFullKey('a1', false, 1)],
        ['c2', asFullKey('a1', false, 2)],
      ],
      ['b4', asFullKey('b2', false, 0)],
      'a1',
      undefined,
    ),
    6,
  );

  const a1BRBRGraph = await execute(
    graph,
    scrollBottomHorizontal(a1BRGraph, true),
    buildFullView(
      ['a5', asFullKey('a1', true, 0)],
      [undefined, ['a1', asDirectKey('a1')], undefined],
      [
        ['b2', asFullKey('a1', false, 1)],
        ['c2', asFullKey('a1', false, 2)],
        ['d2', asFullKey('a1', false, 3)],
      ],
      undefined,
      'a1',
      undefined,
    ),
    6,
  );

  assertTrue(
    scrollVertical(a1BRBRGraph, false) === undefined,
    `${safeStringify(scrollVertical(a1BRBRGraph, false))}`,
  );

  await execute(
    graph,
    scrollBottomHorizontal(a1BRGraph, false),
    buildFullView(
      ['a5', asFullKey('a1', true, 0)],
      [undefined, ['a1', asDirectKey('a1')], undefined],
      [
        undefined,
        ['a2', asFullKey('a1', false, 0)],
        ['b2', asFullKey('a1', false, 1)],
      ],
      ['a3', asFullKey('a2', false, 0)],
      'a1',
      undefined,
    ),
    6,
  );

  const a1BRTRGraph = await execute(
    graph,
    scrollTopHorizontal(a1BRGraph, true),
    buildFullView(
      ['a3', asFullKey('b4', true, 0)],
      [['a1', asDirectKey('a1')], ['b4', asFullKey('b2', true, 1)], undefined],
      [undefined, ['b2', asDirectKey('b2')], undefined],
      ['b4', asFullKey('b2', false, 0)],
      'a1',
      'b2',
    ),
    8,
  );

  const a1BRTRUGraph = await execute(
    graph,
    scrollVertical(a1BRTRGraph, true),
    buildFullView(
      ['a2', asFullKey('a3', true, 0)],
      [
        undefined,
        ['a3', asFullKey('b4', true, 0)],
        ['b2', asFullKey('b4', true, 1)],
      ],
      [['a1', asDirectKey('a1')], ['b4', asFullKey('b2', true, 1)], undefined],
      ['b2', asDirectKey('b2')],
      undefined,
      'a1',
    ),
    6,
  );

  const a1BRTRUTRGraph = await execute(
    graph,
    scrollTopHorizontal(a1BRTRUGraph, true),
    buildFullView(
      ['a1', asFullKey('b2', true, 0)],
      [
        ['a3', asFullKey('b4', true, 0)],
        ['b2', asFullKey('b4', true, 1)],
        undefined,
      ],
      [undefined, ['b4', asDirectKey('b4')], undefined],
      ['b2', asDirectKey('b2')],
      undefined,
      'b4',
    ),
    8,
  );

  await execute(
    graph,
    scrollVertical(a1BRTRUTRGraph, false),
    buildFullView(
      ['b2', asFullKey('b4', true, 1)],
      [undefined, ['b4', asDirectKey('b4')], undefined],
      [undefined, ['b2', asDirectKey('b2')], undefined],
      ['b4', asFullKey('b2', false, 0)],
      'b4',
      'b2',
    ),
    6,
  );

  const a1BRTRUTRTLGraph = await execute(
    graph,
    scrollTopHorizontal(a1BRTRUTRGraph, false),
    buildFullView(
      ['a2', asFullKey('a3', true, 0)],
      [
        undefined,
        ['a3', asFullKey('b4', true, 0)],
        ['b2', asFullKey('b4', true, 1)],
      ],
      [
        undefined,
        ['b4', asDirectKey('b4')],
        ['a4', asFullKey('a3', false, 0)],
      ],
      ['b2', asDirectKey('b2')],
      undefined,
      'b4',
    ),
    8,
  );

  const a1BRTRUTRTLBRGraph = await execute(
    graph,
    scrollBottomHorizontal(a1BRTRUTRTLGraph, true),
    buildFullView(
      ['a2', asFullKey('a3', true, 0)],
      [undefined, ['a3', asDirectKey('a3')], undefined],
      [
        ['b4', asDirectKey('b4')],
        ['a4', asFullKey('a3', false, 0)],
        undefined,
      ],
      ['a5', asFullKey('a4', false, 0)],
      'a3',
      'b4',
    ),
    6,
  );

  assertTrue(
    scrollTopHorizontal(a1BRTRUTRTLBRGraph, true) === undefined,
    `${safeStringify(scrollTopHorizontal(a1BRTRUTRTLBRGraph, true))}`,
  );

  await execute(
    graph,
    initView('a5' as MHash, 'a1' as MHash),
    buildFullView(
      ['a4', asFullKey('a5', true, 0)],
      [undefined, ['a5', asDirectKey('a5')], undefined],
      [undefined, ['a1', asDirectKey('a1')], undefined],
      ['a2', asFullKey('a1', false, 0)],
      'a5',
      'a1',
    ),
    13,
  );
});

test('test graph infinite', async () => {
  const graph = new CommentGraph(new InfGraph(3).getApiProvider(), {
    maxCommentPoolSize: 100,
    maxTopicSize: 100,
    maxLinkPoolSize: 100,
    maxLinkCache: 100,
    maxLineSize: 100,
    maxUserCache: 100,
    maxUserLineSize: 100,
    blockSize: 10,
  });

  // FIXME: direct key left replacement
  // FIXME: user key
  await execute(
    graph,
    initView(undefined),
    buildFullView(
      ['`0', asFullKey('a0', true, 0)],
      [undefined, ['a0', asTopicKey(0)], ['a1', asTopicKey(1)]],
      [
        undefined,
        ['b0', asFullKey('a0', false, 0)],
        ['b1', asFullKey('a0', false, 1)],
      ],
      ['c0', asFullKey('b0', false, 0)],
      undefined,
      undefined,
    ),
    13,
  );
});
