import { Token, UserId } from '../api/types';
import CommentGraph from './CommentGraph';
import {
  adj,
  AdjustedLineIndex,
  equalFullKey,
  FullKey,
  FullKeyType,
  INVALID_FULL_KEY,
  IsGet,
  Link,
  MHash,
  toFullKey,
  TOPIC_KEY,
} from './keys';
import {
  amend,
  assertFail,
  debugJSON,
  LoggerCB,
  maybeLog,
  num,
  OnCacheMiss,
} from '../misc/util';

export type Cell = {
  fullKey: Readonly<FullKey>;
  mhash?: Readonly<MHash>;
  content?: string;
  topLink?: Readonly<Link>;
  invalid?: boolean;
};

export type GraphView = {
  centerTop: Readonly<Cell>;
  centerBottom?: Readonly<Cell>;
  topLeft?: Readonly<Cell>;
  topRight?: Readonly<Cell>;
  bottomLeft?: Readonly<Cell>;
  bottomRight?: Readonly<Cell>;
  top?: Readonly<Cell>;
  bottom?: Readonly<Cell>;
  topSkip?: Readonly<MHash>;
  bottomSkip?: Readonly<MHash>;
};

function cell(
  mhash: Readonly<MHash>,
  isGet: IsGet,
  index: Readonly<AdjustedLineIndex>,
): Readonly<Cell> {
  return {
    fullKey: { fullKeyType: FullKeyType.link, mhash, isGet, index },
  };
}

function invalidCell(): Readonly<Cell> {
  return { fullKey: INVALID_FULL_KEY };
}

function directCell(mhash: Readonly<MHash>): Readonly<Cell> {
  return { fullKey: { fullKeyType: FullKeyType.direct, mhash } };
}

function topicCell(index: Readonly<AdjustedLineIndex>): Readonly<Cell> {
  return { fullKey: toFullKey(TOPIC_KEY, index) };
}

function userCell(userId: Readonly<UserId>): Readonly<Cell> {
  return { fullKey: { fullKeyType: FullKeyType.user, userId } };
}

function userChildCell(
  parentUser: Readonly<UserId>,
  index: Readonly<AdjustedLineIndex>,
): Readonly<Cell> {
  return {
    fullKey: { fullKeyType: FullKeyType.userchild, parentUser, index },
  };
}

export function initView(
  top: Readonly<MHash> | undefined,
  bottom?: Readonly<MHash>,
): Readonly<GraphView> {
  const topCell: Readonly<Cell> =
    top !== undefined ? directCell(top) : topicCell(adj(0));
  const bottomCell: Readonly<Cell> | undefined =
    bottom !== undefined ? directCell(bottom) : undefined;
  return {
    centerTop: topCell,
    centerBottom: bottomCell,
    topSkip: top,
    bottomSkip: bottom,
  };
}

// FIXME use
// ts-unused-exports:disable-next-line
export function initUserView(userId: Readonly<UserId>): Readonly<GraphView> {
  return {
    centerTop: userCell(userId),
  };
}

// FIXME use
// ts-unused-exports:disable-next-line
export function removeAllLinks(
  view: Readonly<GraphView>,
): Readonly<GraphView> {
  const {
    centerTop,
    centerBottom,
    topLeft,
    topRight,
    bottomLeft,
    bottomRight,
    top,
    bottom,
    ...rest
  } = view;
  return {
    centerTop: removeLink(centerTop),
    centerBottom: removeLink(centerBottom),
    topLeft: removeLink(topLeft),
    topRight: removeLink(topRight),
    bottomLeft: removeLink(bottomLeft),
    bottomRight: removeLink(bottomRight),
    top: removeLink(top),
    bottom: removeLink(bottom),
    ...rest,
  };
}

async function getCellContent(
  graph: CommentGraph,
  cell: Readonly<Cell>,
  ocm: OnCacheMiss,
): Promise<Readonly<Cell>> {
  const [mhash, content] = await graph.getMessage(cell.fullKey, ocm);
  const res = mhash !== undefined ? { mhash } : { invalid: true };
  return {
    ...cell,
    ...res,
    content,
  };
}

async function getTopLink(
  graph: CommentGraph,
  cell: Readonly<Cell>,
  parent: Readonly<FullKey>,
  token: Readonly<Token> | undefined,
  ocm: OnCacheMiss,
): Promise<Readonly<Cell>> {
  const link = await graph.getSingleLink(parent, cell.fullKey, token, ocm);
  return {
    ...cell,
    topLink: link,
  };
}

async function getNextCell(
  graph: CommentGraph,
  sameLevel: Readonly<FullKey>,
  otherLevel: Readonly<MHash>,
  skip: Readonly<MHash> | undefined,
  isGet: IsGet,
  isIncrease: boolean,
  ocm: OnCacheMiss,
  logger: LoggerCB,
): Promise<Readonly<Cell>> {
  const log = maybeLog(logger, 'nextCell');
  const fullKeyType = sameLevel.fullKeyType;
  const change = isIncrease ? 1 : -1;
  if (fullKeyType === FullKeyType.invalid) {
    log('invalid');
    return getCellContent(graph, invalidCell(), ocm);
  }
  if (fullKeyType === FullKeyType.user) {
    log('user');
    return getCellContent(graph, invalidCell(), ocm);
  }

  const getIndexedContent = async (
    oldIndex: Readonly<AdjustedLineIndex>,
    skip: Readonly<MHash> | undefined,
    initCell: (index: Readonly<AdjustedLineIndex>) => Readonly<Cell>,
  ): Promise<Readonly<Cell>> => {
    const index = adj(num(oldIndex) + change);
    if (num(index) === -1 && skip !== undefined) {
      log('direct -1');
      return getCellContent(graph, directCell(skip), ocm);
    }
    if (num(index) < 0) {
      log('invalid negative');
      return getCellContent(graph, invalidCell(), ocm);
    }
    const res = await getCellContent(graph, initCell(index), ocm);
    if (skip === undefined || res.mhash !== skip) {
      log(`no skip:${skip}`);
      return res;
    }
    log(`skip at ${index}`);
    return getIndexedContent(index, undefined, initCell);
  };

  if (fullKeyType === FullKeyType.userchild) {
    log('userchild');
    return getIndexedContent(sameLevel.index, skip, (index) =>
      userChildCell(sameLevel.parentUser, index),
    );
  }
  if (fullKeyType === FullKeyType.topic) {
    log('topic');
    return getIndexedContent(sameLevel.index, skip, (index) =>
      topicCell(index),
    );
  }
  if (fullKeyType === FullKeyType.link) {
    log('link');
    return getIndexedContent(sameLevel.index, skip, (index) =>
      cell(otherLevel, isGet, index),
    );
  }

  if (fullKeyType === FullKeyType.direct) {
    log('direct');
    if (!isIncrease) {
      return getCellContent(graph, invalidCell(), ocm);
    }
    return getIndexedContent(adj(-1), skip, (index) =>
      cell(otherLevel, isGet, index),
    );
  }
  /* istanbul ignore next */
  assertFail(`unkown FullKeyType: ${fullKeyType}`);
}

export async function progressView(
  graph: CommentGraph,
  view: Readonly<GraphView>,
  token: Readonly<Token> | undefined,
  ocm?: OnCacheMiss,
  logger?: LoggerCB,
): Promise<Readonly<{ view: Readonly<GraphView>; change: boolean }>> {
  const log = maybeLog(logger, 'progress');
  if (view.centerTop.invalid) {
    log('invalid');
    return { view, change: false };
  }
  if (
    view.centerTop.content === undefined ||
    view.centerTop.mhash === undefined
  ) {
    log('centerTop content');
    return {
      view: {
        ...view,
        centerTop: await getCellContent(graph, view.centerTop, ocm),
      },
      change: true,
    };
  }
  if (
    view.centerBottom === undefined ||
    view.centerBottom.mhash === undefined ||
    view.centerBottom.content === undefined
  ) {
    log('centerBottom content');
    return {
      view: {
        ...view,
        centerBottom: await getCellContent(
          graph,
          view.centerBottom !== undefined
            ? view.centerBottom
            : cell(view.centerTop.mhash, IsGet.child, adj(0)),
          ocm,
        ),
      },
      change: true,
    };
  }
  if (view.centerBottom.topLink === undefined) {
    log('centerBottom topLink');
    return {
      view: {
        ...view,
        centerBottom: await getTopLink(
          graph,
          view.centerBottom,
          view.centerTop.fullKey,
          token,
          ocm,
        ),
      },
      change: true,
    };
  }
  if (view.topRight === undefined) {
    log('centerTop neighbor right');
    return {
      view: {
        ...view,
        topRight: await getNextCell(
          graph,
          view.centerTop.fullKey,
          view.centerBottom.mhash,
          view.topSkip,
          IsGet.parent,
          true,
          ocm,
          log,
        ),
      },
      change: true,
    };
  }
  if (view.bottomRight === undefined) {
    log('centerBottom neighbor right');
    return {
      view: {
        ...view,
        bottomRight: await getNextCell(
          graph,
          view.centerBottom.fullKey,
          view.centerTop.mhash,
          view.bottomSkip,
          IsGet.child,
          true,
          ocm,
          log,
        ),
      },
      change: true,
    };
  }
  if (view.topLeft === undefined) {
    log('centerTop neighbor left');
    return {
      view: {
        ...view,
        topLeft: await getNextCell(
          graph,
          view.centerTop.fullKey,
          view.centerBottom.mhash,
          view.topSkip,
          IsGet.parent,
          false,
          ocm,
          log,
        ),
      },
      change: true,
    };
  }
  if (view.bottomLeft === undefined) {
    log('centerBottom neighbor left');
    return {
      view: {
        ...view,
        bottomLeft: await getNextCell(
          graph,
          view.centerBottom.fullKey,
          view.centerTop.mhash,
          view.bottomSkip,
          IsGet.child,
          false,
          ocm,
          log,
        ),
      },
      change: true,
    };
  }
  if (view.top === undefined) {
    log('top content');
    return {
      view: {
        ...view,
        top: await getCellContent(
          graph,
          cell(view.centerTop.mhash, IsGet.parent, adj(0)),
          ocm,
        ),
      },
      change: true,
    };
  }
  if (view.bottom === undefined) {
    log('bottom content');
    return {
      view: {
        ...view,
        bottom: await getCellContent(
          graph,
          cell(view.centerBottom.mhash, IsGet.child, adj(0)),
          ocm,
        ),
      },
      change: true,
    };
  }
  if (view.centerTop.topLink === undefined) {
    log('centerTop topLink');
    return {
      view: {
        ...view,
        centerTop: await getTopLink(
          graph,
          view.centerTop,
          view.top.fullKey,
          token,
          ocm,
        ),
      },
      change: true,
    };
  }
  if (view.bottom.topLink === undefined) {
    log('bottom topLink');
    return {
      view: {
        ...view,
        bottom: await getTopLink(
          graph,
          view.bottom,
          view.centerBottom.fullKey,
          token,
          ocm,
        ),
      },
      change: true,
    };
  }
  if (view.bottomRight.topLink === undefined) {
    log('bottomRight topLink');
    return {
      view: {
        ...view,
        bottomRight: await getTopLink(
          graph,
          view.bottomRight,
          view.centerTop.fullKey,
          token,
          ocm,
        ),
      },
      change: true,
    };
  }
  if (view.bottomLeft.topLink === undefined) {
    log('bottomLeft topLink');
    return {
      view: {
        ...view,
        bottomLeft: await getTopLink(
          graph,
          view.bottomLeft,
          view.centerTop.fullKey,
          token,
          ocm,
        ),
      },
      change: true,
    };
  }
  log('no change');
  return { view, change: false };
}

function removeLink<T extends Readonly<Cell> | undefined>(
  cell: T,
): Readonly<Cell> | (undefined extends T ? undefined : never) {
  if (cell === undefined) {
    return undefined as undefined extends T ? undefined : never;
  }
  const { topLink: _, ...rest }: Readonly<Cell> = cell;
  return { ...rest };
}

export enum Direction {
  UpRight = 'UpRight',
  DownLeft = 'DownLeft',
}

export type NavigationCB = (
  view: Readonly<GraphView>,
  direction: Direction,
) => Readonly<GraphView> | undefined;

type HNavigationCB = (
  view: Readonly<GraphView>,
  direction: HDirection,
) => Readonly<GraphView> | undefined;

type VNavigationCB = (
  view: Readonly<GraphView>,
  direction: VDirection,
) => Readonly<GraphView> | undefined;

export function horizontal(cb: HNavigationCB): NavigationCB {
  return (view, direction) =>
    cb(
      view,
      direction === Direction.UpRight ? HDirection.Right : HDirection.Left,
    );
}

export function vertical(cb: VNavigationCB): NavigationCB {
  return (view, direction) =>
    cb(
      view,
      direction === Direction.UpRight ? VDirection.Up : VDirection.Down,
    );
}

export enum HDirection {
  Right = 'Right',
  Left = 'Left',
}

export enum VDirection {
  Up = 'Up',
  Down = 'Down',
}

export function scrollVertical(
  view: Readonly<GraphView>,
  direction: VDirection,
): Readonly<GraphView> | undefined {
  if (direction === VDirection.Up) {
    if (view.top === undefined || view.top.invalid) {
      return undefined;
    }
    return {
      top: undefined,
      centerTop: view.top,
      centerBottom: view.centerTop,
      bottom: view.centerBottom,
      topLeft: undefined,
      topRight: undefined,
      bottomLeft: view.topLeft,
      bottomRight: view.topRight,
      topSkip:
        view.top.fullKey.fullKeyType === FullKeyType.direct
          ? view.top.mhash
          : undefined,
      bottomSkip: view.topSkip,
    };
  } else {
    if (view.bottom === undefined || view.bottom.invalid) {
      return undefined;
    }
    if (view.centerBottom === undefined || view.centerBottom.invalid) {
      return undefined;
    }
    return {
      top: removeLink(view.centerTop),
      centerTop: view.centerBottom,
      centerBottom: view.bottom,
      bottom: undefined,
      topLeft: removeLink(view.bottomLeft),
      topRight: removeLink(view.bottomRight),
      bottomLeft: undefined,
      bottomRight: undefined,
      topSkip: view.bottomSkip,
      bottomSkip:
        view.bottom !== undefined &&
        view.bottom.fullKey.fullKeyType === FullKeyType.direct
          ? view.bottom.mhash
          : undefined,
    };
  }
}

function convertToDirect<T extends Readonly<Cell> | undefined>(
  cell: T,
): T | (undefined extends T ? undefined : never) {
  if (cell === undefined || cell.mhash === undefined) {
    return undefined as undefined extends T ? undefined : never;
  }
  return {
    ...cell,
    ...directCell(cell.mhash),
  };
}

export function scrollTopHorizontal(
  view: Readonly<GraphView>,
  direction: HDirection,
): Readonly<GraphView> | undefined {
  const centerBottom = removeLink(convertToDirect(view.centerBottom));
  if (direction == HDirection.Right) {
    if (view.topRight === undefined || view.topRight.invalid) {
      return undefined;
    }
    return {
      top: undefined,
      centerTop: removeLink(view.topRight),
      centerBottom: centerBottom,
      bottom: centerBottom !== undefined ? view.bottom : undefined,
      topLeft: removeLink(view.centerTop),
      topRight: undefined,
      bottomLeft: undefined,
      bottomRight: undefined,
      topSkip: view.topSkip,
      bottomSkip: centerBottom !== undefined ? centerBottom.mhash : undefined,
    };
  } else {
    if (view.topLeft === undefined || view.topLeft.invalid) {
      return undefined;
    }
    return {
      top: undefined,
      centerTop: removeLink(view.topLeft),
      centerBottom: centerBottom,
      bottom: centerBottom !== undefined ? view.bottom : undefined,
      topLeft: undefined,
      topRight: removeLink(view.centerTop),
      bottomLeft: undefined,
      bottomRight: undefined,
      topSkip: view.topSkip,
      bottomSkip: centerBottom !== undefined ? centerBottom.mhash : undefined,
    };
  }
}

export function scrollBottomHorizontal(
  view: Readonly<GraphView>,
  direction: HDirection,
): Readonly<GraphView> | undefined {
  const centerTop = convertToDirect(view.centerTop);
  if (direction === HDirection.Right) {
    if (view.bottomRight === undefined || view.bottomRight.invalid) {
      return undefined;
    }
    return {
      top: view.top,
      centerTop,
      centerBottom: view.bottomRight,
      bottom: undefined,
      topLeft: undefined,
      topRight: undefined,
      bottomLeft: view.centerBottom,
      bottomRight: undefined,
      topSkip: centerTop.mhash,
      bottomSkip: view.bottomSkip,
    };
  } else {
    if (view.bottomLeft === undefined || view.bottomLeft.invalid) {
      return undefined;
    }
    return {
      top: view.top,
      centerTop,
      centerBottom: view.bottomLeft,
      bottom: undefined,
      topLeft: undefined,
      topRight: undefined,
      bottomLeft: undefined,
      bottomRight: view.centerBottom,
      topSkip: centerTop.mhash,
      bottomSkip: view.bottomSkip,
    };
  }
}

function equalCell(
  cell: Readonly<Cell> | undefined,
  expected: Readonly<Cell> | undefined,
  logger?: LoggerCB,
): boolean {
  const log = maybeLog(logger, 'equalCell:');
  if (cell === undefined && expected === undefined) {
    return true;
  }
  if (cell === undefined || expected === undefined) {
    log(`cell:${debugJSON(cell)} !== expected:${debugJSON(expected)}`);
    return false;
  }
  if (cell.invalid && expected.invalid) {
    return true;
  }
  if (cell.invalid || expected.invalid) {
    log(
      `cell.invalid:${debugJSON(cell)}`,
      '!==',
      `expected.invalid:${debugJSON(expected)}`,
    );
    return false;
  }
  if (!equalFullKey(cell.fullKey, expected.fullKey, amend(log, 'fullKey'))) {
    return false;
  }
  if (cell.mhash !== expected.mhash) {
    log(`cell.mhash:${cell.mhash} !== expected.mhash:${expected.mhash}`);
    return false;
  }
  // NOTE: topLink is a cache
  if (cell.content === expected.content) {
    return true;
  }
  log(`cell.content:${cell.content} !== expected.content:${expected.content}`);
  return false;
}

export function equalView(
  view: Readonly<GraphView> | undefined,
  expected: Readonly<GraphView> | undefined,
  logger?: LoggerCB,
): boolean {
  const log = maybeLog(logger, 'equalView:');
  if (view === undefined && expected === undefined) {
    return true;
  }
  if (view === undefined || expected === undefined) {
    log(`view:${debugJSON(view)} !== expected:${debugJSON(expected)}`);
    return false;
  }
  if (view.topSkip !== expected.topSkip) {
    log(
      `view.topSkip:${debugJSON(view.topSkip)}`,
      '!==',
      `expected.topSkip:${debugJSON(expected.topSkip)}`,
    );
    return false;
  }
  if (view.bottomSkip !== expected.bottomSkip) {
    log(
      `view.bottomSkip:${debugJSON(view.bottomSkip)}`,
      '!==',
      `expected.bottomSkip:${debugJSON(expected.bottomSkip)}`,
    );
    return false;
  }
  if (
    !equalCell(view.centerTop, expected.centerTop, amend(log, 'centerTop'))
  ) {
    return false;
  }
  if (
    !equalCell(
      view.centerBottom,
      expected.centerBottom,
      amend(log, 'centerBottom'),
    )
  ) {
    return false;
  }
  if (!equalCell(view.topLeft, expected.topLeft, amend(log, 'topLeft'))) {
    return false;
  }
  if (!equalCell(view.topRight, expected.topRight, amend(log, 'topRight'))) {
    return false;
  }
  if (
    !equalCell(view.bottomLeft, expected.bottomLeft, amend(log, 'bottomLeft'))
  ) {
    return false;
  }
  if (
    !equalCell(
      view.bottomRight,
      expected.bottomRight,
      amend(log, 'bottomRight'),
    )
  ) {
    return false;
  }
  if (!equalCell(view.top, expected.top, amend(log, 'top'))) {
    return false;
  }
  return equalCell(view.bottom, expected.bottom, amend(log, 'bottom'));
}

function checkLink(
  cell: Readonly<Cell>,
  other: Readonly<Cell> | undefined,
  logger: LoggerCB,
): boolean {
  if (other === undefined) {
    if (cell.topLink === undefined) {
      return true;
    }
    logger(`cell:${debugJSON(cell)} !== other:${debugJSON(other)}`);
    return false;
  }
  if (cell.invalid && cell.topLink !== undefined && cell.topLink.invalid) {
    return true;
  }
  if (cell.topLink === undefined || cell.topLink.invalid) {
    logger(`invalid topLink: ${debugJSON(cell.topLink)}`);
    return false;
  }
  if (cell.mhash === undefined || other.mhash === undefined) {
    logger(`undefined hash cell:${cell.mhash} other:${other.mhash}`);
    return false;
  }
  const topLink = cell.topLink;
  if (topLink.child === cell.mhash && topLink.parent === other.mhash) {
    return true;
  }
  logger(
    'mismatching link:',
    debugJSON(topLink),
    `!== parent:${other.mhash} child:${cell.mhash}`,
  );
  return false;
}

export function consistentLinks(
  view: Readonly<GraphView>,
  logger?: LoggerCB,
): boolean {
  const log = maybeLog(logger, 'consistentLinks:');
  if (view.top === undefined) {
    log('top is undefined');
    return false;
  }
  if (!checkLink(view.centerTop, view.top, amend(log, '(centerTop->top)'))) {
    return false;
  }
  if (!checkLink(view.top, undefined, amend(log, '(top->undefined)'))) {
    return false;
  }
  if (view.topLeft === undefined) {
    log('topLeft is undefined');
    return false;
  }
  if (
    !checkLink(view.topLeft, undefined, amend(log, '(topLeft->undefined)'))
  ) {
    return false;
  }
  if (view.topRight === undefined) {
    log('topRight is undefined');
    return false;
  }
  if (
    !checkLink(view.topRight, undefined, amend(log, '(topRight->undefined)'))
  ) {
    return false;
  }
  if (view.centerBottom === undefined) {
    log('centerBottom is undefined');
    return false;
  }
  if (
    !checkLink(
      view.centerBottom,
      view.centerTop,
      amend(log, '(centerBottom->centerTop)'),
    )
  ) {
    return false;
  }
  if (view.bottomLeft === undefined) {
    log('bottomLeft is undefined');
    return false;
  }
  if (
    !checkLink(
      view.bottomLeft,
      view.centerTop,
      amend(log, '(bottomLeft->centerTop)'),
    )
  ) {
    return false;
  }
  if (view.bottomRight === undefined) {
    log('bottomRight is undefined');
    return false;
  }
  if (
    !checkLink(
      view.bottomRight,
      view.centerTop,
      amend(log, '(bottomRight->centerTop)'),
    )
  ) {
    return false;
  }
  if (view.bottom === undefined) {
    log('bottom is undefined');
    return false;
  }
  return checkLink(
    view.bottom,
    view.centerBottom,
    amend(log, '(bottom->centerBottom)'),
  );
}
