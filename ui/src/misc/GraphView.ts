import CommentGraph, {
  adj,
  AdjustedLineIndex,
  equalFullKey,
  FullKey,
  FullKeyType,
  INVALID_FULL_KEY,
  IsGet,
  Link,
  MHash,
  NotifyContentCB,
  toFullKey,
  TOPIC_KEY,
  UserId,
} from './CommentGraph';
import { amend, LoggerCB, maybeLog, num, safeStringify } from './util';

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

/* istanbul ignore next */
export function equalCell(
  cell: Readonly<Cell> | undefined,
  expected: Readonly<Cell> | undefined,
  logger?: LoggerCB,
): boolean {
  const log = maybeLog(logger, 'equalCell:');
  if (cell === undefined && expected === undefined) {
    return true;
  }
  if (cell === undefined || expected === undefined) {
    log(`cell:${safeStringify(cell)} !== expected:${safeStringify(expected)}`);
    return false;
  }
  if (cell.invalid && expected.invalid) {
    return true;
  }
  if (cell.invalid || expected.invalid) {
    log(
      `cell.invalid:${safeStringify(cell)}`,
      '!==',
      `expected.invalid:${safeStringify(expected)}`,
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

/* istanbul ignore next */
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
    log(`view:${safeStringify(view)} !== expected:${safeStringify(expected)}`);
    return false;
  }
  if (view.topSkip !== expected.topSkip) {
    log(
      `view.topSkip:${safeStringify(view.topSkip)}`,
      '!==',
      `expected.topSkip:${safeStringify(expected.topSkip)}`,
    );
    return false;
  }
  if (view.bottomSkip !== expected.bottomSkip) {
    log(
      `view.bottomSkip:${safeStringify(view.bottomSkip)}`,
      '!==',
      `expected.bottomSkip:${safeStringify(expected.bottomSkip)}`,
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

/* istanbul ignore next */
function checkLink(
  cell: Readonly<Cell>,
  other: Readonly<Cell> | undefined,
  logger: LoggerCB,
): boolean {
  if (other === undefined) {
    if (cell.topLink === undefined) {
      return true;
    }
    logger(`cell:${safeStringify(cell)} !== other:${safeStringify(other)}`);
    return false;
  }
  if (cell.invalid && cell.topLink !== undefined && cell.topLink.invalid) {
    return true;
  }
  if (cell.topLink === undefined || cell.topLink.invalid) {
    logger(`invalid topLink: ${safeStringify(cell.topLink)}`);
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
    safeStringify(topLink),
    `!== parent:${other.mhash} child:${cell.mhash}`,
  );
  return false;
}

/* istanbul ignore next */
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

export type ViewUpdateCB = (view: Readonly<GraphView>) => void;
type CellUpdateCB = (cell: Readonly<Cell>) => void;

function cell(
  mhash: Readonly<MHash>,
  isGet: IsGet,
  index: Readonly<AdjustedLineIndex>,
): Readonly<Cell> {
  return {
    fullKey: { fullKeyType: FullKeyType.link, mhash, isGet, index },
  };
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

// function getCellHash(
//   graph: CommentGraph,
//   cell: Readonly<Cell>,
//   updateCB: CellUpdateCB,
// ): void {
//   graph.getHash(cell.fullKey, (mhash) => {
//     const res =
//       mhash !== undefined ? { ...cell, mhash } : { ...cell, invalid: true };
//     updateCB(res);
//   });
// }

function getCellContent(
  graph: CommentGraph,
  cell: Readonly<Cell>,
  updateCB: CellUpdateCB,
) {
  const getMessage: NotifyContentCB = (mhash, content) => {
    const res = mhash !== undefined ? { mhash } : { invalid: true };
    updateCB({
      ...cell,
      ...res,
      content,
    });
  };
  const res = graph.getMessage(cell.fullKey, getMessage);
  if (res !== undefined) {
    getMessage(...res);
  }
}

function getTopLink(
  graph: CommentGraph,
  cell: Readonly<Cell>,
  parent: Readonly<FullKey>,
  updateCB: CellUpdateCB,
) {
  graph.getSingleLink(parent, cell.fullKey, (link) => {
    updateCB({
      ...cell,
      topLink: link,
    });
  });
}

function getNextCell(
  graph: CommentGraph,
  sameLevel: Readonly<FullKey>,
  otherLevel: Readonly<MHash>,
  skip: Readonly<MHash> | undefined,
  isTop: boolean,
  isIncrease: boolean,
  updateCB: CellUpdateCB,
): void {
  const index =
    sameLevel.fullKeyType === FullKeyType.direct ||
    sameLevel.fullKeyType === FullKeyType.invalid ||
    sameLevel.fullKeyType === FullKeyType.user
      ? adj(isIncrease ? 0 : -1)
      : adj(num(sameLevel.index) + (isIncrease ? 1 : -1));
  getCellContent(
    graph,
    num(index) === -1 && skip !== undefined
      ? directCell(skip)
      : num(index) < 0
      ? { fullKey: INVALID_FULL_KEY }
      : sameLevel.fullKeyType === FullKeyType.topic
      ? topicCell(index)
      : sameLevel.fullKeyType === FullKeyType.user
      ? userCell(sameLevel.userId)
      : sameLevel.fullKeyType === FullKeyType.userchild
      ? userChildCell(sameLevel.parentUser, index)
      : cell(otherLevel, isTop ? IsGet.parent : IsGet.child, index),
    (res) => {
      if (skip !== undefined && res.mhash === skip && num(index) >= 0) {
        const skipIndex = adj(num(index) + 1);
        getCellContent(
          graph,
          sameLevel.fullKeyType === FullKeyType.topic
            ? topicCell(skipIndex)
            : cell(otherLevel, isTop ? IsGet.parent : IsGet.child, skipIndex),
          updateCB,
        );
        return;
      }
      updateCB(res);
    },
  );
}

export function progressView(
  graph: CommentGraph,
  view: Readonly<GraphView>,
  updateCB: ViewUpdateCB,
  logger?: LoggerCB,
): Readonly<GraphView> | undefined {
  const log = maybeLog(logger, 'progress');
  if (view.centerTop.invalid) {
    log('invalid');
    return view;
  }
  if (
    view.centerTop.content === undefined ||
    view.centerTop.mhash === undefined
  ) {
    log('centerTop content');
    getCellContent(graph, view.centerTop, (cell) => {
      updateCB({ ...view, centerTop: cell });
    });
  } else if (
    view.centerBottom === undefined ||
    view.centerBottom.mhash === undefined ||
    view.centerBottom.content === undefined
  ) {
    log('centerBottom content');
    getCellContent(
      graph,
      view.centerBottom !== undefined
        ? view.centerBottom
        : cell(view.centerTop.mhash, IsGet.child, adj(0)),
      (cell) => {
        updateCB({ ...view, centerBottom: cell });
      },
    );
  } else if (view.centerBottom.topLink === undefined) {
    log('centerBottom topLink');
    getTopLink(graph, view.centerBottom, view.centerTop.fullKey, (cell) => {
      updateCB({ ...view, centerBottom: cell });
    });
  } else if (view.topRight === undefined) {
    log('centerTop neighbor right');
    getNextCell(
      graph,
      view.centerTop.fullKey,
      view.centerBottom.mhash,
      view.topSkip,
      true,
      true,
      (cell) => {
        updateCB({ ...view, topRight: cell });
      },
    );
  } else if (view.bottomRight === undefined) {
    log('centerBottom neighbor right');
    getNextCell(
      graph,
      view.centerBottom.fullKey,
      view.centerTop.mhash,
      view.bottomSkip,
      false,
      true,
      (cell) => {
        updateCB({ ...view, bottomRight: cell });
      },
    );
  } else if (view.topLeft === undefined) {
    log('centerTop neighbor left');
    getNextCell(
      graph,
      view.centerTop.fullKey,
      view.centerBottom.mhash,
      undefined,
      true,
      false,
      (cell) => {
        updateCB({ ...view, topLeft: cell });
      },
    );
  } else if (view.bottomLeft === undefined) {
    log('centerBottom neighbor left');
    getNextCell(
      graph,
      view.centerBottom.fullKey,
      view.centerTop.mhash,
      undefined,
      false,
      false,
      (cell) => {
        updateCB({ ...view, bottomLeft: cell });
      },
    );
  } else if (view.top === undefined) {
    log('top content');
    getCellContent(
      graph,
      cell(view.centerTop.mhash, IsGet.parent, adj(0)),
      (cell) => {
        updateCB({ ...view, top: cell });
      },
    );
  } else if (view.bottom === undefined) {
    log('bottom content');
    getCellContent(
      graph,
      cell(view.centerBottom.mhash, IsGet.child, adj(0)),
      (cell) => {
        updateCB({ ...view, bottom: cell });
      },
    );
  } else if (view.centerTop.topLink === undefined) {
    log('centerTop topLink');
    getTopLink(graph, view.centerTop, view.top.fullKey, (cell) => {
      updateCB({ ...view, centerTop: cell });
    });
  } else if (view.bottom.topLink === undefined) {
    log('bottom topLink');
    getTopLink(graph, view.bottom, view.centerBottom.fullKey, (cell) => {
      updateCB({ ...view, bottom: cell });
    });
  } else if (view.bottomRight.topLink === undefined) {
    log('bottomRight topLink');
    getTopLink(graph, view.bottomRight, view.centerTop.fullKey, (cell) => {
      updateCB({ ...view, bottomRight: cell });
    });
  } else if (view.bottomLeft.topLink === undefined) {
    log('bottomLeft topLink');
    getTopLink(graph, view.bottomLeft, view.centerTop.fullKey, (cell) => {
      updateCB({ ...view, bottomLeft: cell });
    });
  } else {
    log('no change');
    return view;
  }
  return undefined;
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

export function scrollVertical(
  view: Readonly<GraphView>,
  up: boolean,
): Readonly<GraphView> | undefined {
  if (up) {
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
  right: boolean,
): Readonly<GraphView> | undefined {
  const centerBottom = removeLink(convertToDirect(view.centerBottom));
  if (right) {
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
  right: boolean,
): Readonly<GraphView> | undefined {
  const centerTop = convertToDirect(view.centerTop);
  if (right) {
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
