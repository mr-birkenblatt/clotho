import React, { PureComponent } from 'react';
import { connect, ConnectedProps } from 'react-redux';
import styled from 'styled-components';
import {
  AdjustedLineIndex,
  equalLineKeys,
  FullKey,
  LineKey,
  Link,
  NextCB,
  ReadyCB,
  toFullKey,
} from '../misc/CommentGraph';
import { range } from '../misc/util';
import { RootState } from '../store';
import {
  constructKey,
  setVCurrentIx,
  addLine,
  focusV,
  VIndex,
  LineIndex,
  VArrIndex,
} from './LineStateSlice';

const Outer = styled.div`
  position: relative;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
`;

const NavButtonUp = styled.button<ButtonProps>`
  position: fixed;
  top: 0;
  right: 0;
  width: ${(props) => props.buttonSize}px;
  height: ${(props) => props.buttonSize}px;
  pointer-events: auto;
  opacity: 0.8;
`;

const NavButtonDown = styled.button<ButtonProps>`
  position: fixed;
  bottom: 0;
  right: 0;
  width: ${(props) => props.buttonSize}px;
  height: ${(props) => props.buttonSize}px;
  pointer-events: auto;
  opacity: 0.8;
`;

const Band = styled.div`
  width: 100%;
  height: 100%;
  background-color: green;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: scroll;

  &::-webkit-scrollbar {
    display: none;
  }

  -ms-overflow-style: none;
  scrollbar-width: none;
  scroll-snap-type: y mandatory;
`;

const Item = styled.div`
  width: 100%;
  height: auto;
  scroll-snap-align: start;
  background-color: ${(props: { isCurrent: boolean }) =>
    props.isCurrent ? 'blue' : 'cornflowerblue'};
`;

const ItemMid = styled.div`
  display: flex;
  width: 100%;
  height: 0;
  position: relative;
  top: 0;
  left: 0;
  align-items: center;
  justify-content: center;
  text-align: center;
  opacity: 0.8;
`;

type ButtonProps = {
  buttonSize: number;
};

export enum VPosType {
  AboveFocus,
  InFocus,
  BelowFocus,
}

export type LinkCB = (
  fullLinkKey: FullKey,
  parentIndex: AdjustedLineIndex,
  readyCb: ReadyCB,
) => Link | undefined;
export type ChildLineCB = (
  lineKey: LineKey,
  index: AdjustedLineIndex,
  callback: NextCB,
) => void;
export type ParentLineCB = (
  lineKey: LineKey,
  index: AdjustedLineIndex,
  callback: NextCB,
) => void;
export type VItemCB = (
  lineKey: LineKey | undefined,
  height: number,
  isViewUpdate: boolean,
  vPosType: VPosType,
) => JSX.Element | null;
export type RenderLinkCB = (
  link: Link,
  buttonSize: number,
  radius: number,
) => JSX.Element | null;

interface VerticalProps extends ConnectVertical {
  height: number;
  buttonSize: number;
  radius: number;
  getChildLine: ChildLineCB;
  getParentLine: ParentLineCB;
  getItem: VItemCB;
  getLink: LinkCB;
  renderLink: RenderLinkCB;
}

type EmptyVerticalProps = {
  offset: undefined;
  order: undefined;
  currentIx: undefined;
  correction: undefined;
};

type VerticalState = {
  itemCount: number;
  redraw: boolean;
  scrollInit: boolean;
  isScrolling: boolean;
  focusIx: VIndex;
  viewUpdate: boolean;
};

type EmptyVerticalState = {
  itemCount: undefined;
};

type RefCB = (instance: HTMLDivElement | null) => void;

class Vertical extends PureComponent<VerticalProps, VerticalState> {
  rootBox: React.RefObject<HTMLDivElement>;
  bandRef: React.RefObject<HTMLDivElement>;
  activeRefCbs: Map<VIndex, RefCB>;
  activeElements: Map<VIndex, HTMLDivElement>;
  awaitOrderChange: LineKey[] | undefined;
  awaitCurrentChange: VIndex | undefined;

  constructor(props: VerticalProps) {
    super(props);
    this.state = {
      itemCount: 4,
      redraw: false,
      scrollInit: false,
      isScrolling: false,
      focusIx: 0 as VIndex,
      viewUpdate: false,
    };
    this.rootBox = React.createRef();
    this.bandRef = React.createRef();
    this.activeRefCbs = new Map();
    this.activeElements = new Map();
    this.awaitOrderChange = undefined;
    this.awaitCurrentChange = undefined;
  }

  componentDidMount(): void {
    this.componentDidUpdate(
      {
        offset: undefined,
        order: undefined,
        currentIx: undefined,
        correction: undefined,
      },
      { itemCount: undefined },
    );
  }

  componentWillUnmount(): void {
    if (this.bandRef.current) {
      this.bandRef.current.removeEventListener('scroll', this.handleScroll);
    }
  }

  handleScroll = (): void => {
    const maybeBand = this.bandRef.current;
    if (maybeBand === null) {
      return;
    }
    const band = maybeBand;
    let startTime: number | undefined = undefined;
    const startScroll = band.scrollTop;
    const that = this;

    function checkScroll(time: number) {
      const isScrolling = band.scrollTop !== startScroll;
      if (startTime === undefined) {
        startTime = time;
      }
      const update = time - startTime >= 100 || isScrolling;
      if (update) {
        if (that.state.isScrolling !== isScrolling) {
          that.setState({ isScrolling });
        }
      } else {
        requestAnimationFrame(checkScroll);
      }
    }

    requestAnimationFrame(checkScroll);
  };

  debugString(): void {
    const { offset, order, currentIx, correction } = this.props;
    const { itemCount } = this.state;
    const rangeOrder = [0, order.length];
    const rangeArray = [correction + offset, correction + offset + itemCount];
    const adjIndex = this.getArrayIndex(currentIx);
    const minIx = Math.min(rangeOrder[0], rangeArray[0], adjIndex);
    const maxIx = Math.max(rangeOrder[1], rangeArray[1], adjIndex);
    const ord = range(minIx, maxIx).reduce((cur, val) => {
      const isOrd = val >= rangeOrder[0] && val < rangeOrder[1];
      return `${cur}${isOrd ? '#' : '_'}`;
    }, '');
    const arr = range(minIx, maxIx).reduce((cur, val) => {
      const isArr = val >= rangeArray[0] && val < rangeArray[1];
      return `${cur}${isArr ? '#' : '_'}`;
    }, '');
    const cur = range(minIx, maxIx).reduce((cur, val) => {
      const isCur = val === adjIndex;
      return `${cur}${isCur ? 'X' : '_'}`;
    }, '');
    console.group('VState');
    console.log(`ORD ${ord}`);
    console.log(`ARR ${arr}`);
    console.log(`CUR ${cur}`);
    console.groupEnd();
  }

  computeIx(): VIndex | undefined {
    type PosAndIndex = [undefined, undefined] | [number, VIndex];

    const out = Array.from(this.activeElements.entries()).reduce(
      (res: PosAndIndex, val: [VIndex, HTMLDivElement]): PosAndIndex => {
        const [realIx, element] = val;
        const bounds = element.getBoundingClientRect();
        if (bounds.top <= -50) {
          return res;
        }
        const top = bounds.top;
        if (res[0] === undefined || res[0] >= top) {
          return [top, realIx];
        }
        return res;
      },
      [undefined, undefined],
    );
    return out[1];
  }

  componentDidUpdate(
    prevProps: VerticalProps | EmptyVerticalProps,
    prevState: VerticalState | EmptyVerticalState,
  ): void {
    const {
      correction,
      currentIx,
      dispatch,
      focus,
      focusSmooth,
      getChildLine,
      getParentLine,
      order,
      offset,
    } = this.props;
    const { focusIx, isScrolling, scrollInit, itemCount } = this.state;
    let viewUpdate = this.state.viewUpdate;

    if (
      prevProps.currentIx !== currentIx ||
      prevProps.offset !== offset ||
      // FIXME identity might be enough
      !equalLineKeys(prevProps.order, order) ||
      prevProps.correction !== correction ||
      prevState.itemCount !== itemCount
    ) {
      this.debugString();
    }

    if (!scrollInit && this.bandRef.current) {
      this.bandRef.current.addEventListener('scroll', this.handleScroll, {
        passive: true,
      });
      this.setState({
        scrollInit: true,
      });
    }

    if (this.awaitOrderChange === undefined) {
      if (this.getArrayIndex(currentIx) >= order.length - 2) {
        const lastIx = (order.length - 1) as VIndex;
        console.log('request child line', lastIx);
        getChildLine(
          order[lastIx],
          this.getHIndexAdjusted(lastIx),
          (child) => {
            if (child === undefined) {
              console.warn('no child found!');
              return;
            }
            dispatch(
              addLine({
                lineKey: child,
                isBack: true,
              }),
            );
          },
        );
        this.awaitOrderChange = order;
      } else if (this.getArrayIndex(currentIx) <= 0) {
        const firstIx = 0 as VIndex;
        console.log('request parent line', firstIx);
        getParentLine(
          order[firstIx],
          this.getHIndexAdjusted(firstIx),
          (parent) => {
            if (parent === undefined) {
              console.warn('no parent found!');
              return;
            }
            dispatch(
              addLine({
                lineKey: parent,
                isBack: false,
              }),
            );
          },
        );
        this.awaitOrderChange = order;
      }
    }
    if (
      this.awaitOrderChange !== undefined &&
      !equalLineKeys(this.awaitOrderChange, order)
    ) {
      this.awaitOrderChange = undefined;
    }

    if (
      !isScrolling &&
      !viewUpdate &&
      this.awaitCurrentChange === undefined &&
      this.awaitOrderChange === undefined &&
      focus === focusIx
    ) {
      const computedIx = this.computeIx();
      const nextIx = (
        computedIx !== undefined && computedIx > currentIx
          ? currentIx + 1
          : currentIx - 1
      ) as VIndex;
      const computedLine = this.lineKey(nextIx);
      if (
        computedIx !== undefined &&
        computedLine !== undefined &&
        computedIx !== currentIx
      ) {
        console.log('update computedIx', computedIx, currentIx, nextIx);
        dispatch(
          setVCurrentIx({
            vIndex: nextIx,
            hIndex: this.getHIndexAdjusted(nextIx),
            lineKey: computedLine,
          }),
        );
        this.awaitCurrentChange = currentIx;
        this.setState({
          viewUpdate: true,
        });
        viewUpdate = true;
      }
    }
    if (
      this.awaitCurrentChange !== undefined &&
      this.awaitCurrentChange !== currentIx
    ) {
      this.awaitCurrentChange = undefined;
    }

    viewUpdate = this.checkViewReady(viewUpdate);

    if (
      this.awaitCurrentChange === undefined &&
      this.awaitOrderChange === undefined &&
      focus !== focusIx &&
      !viewUpdate
    ) {
      this.focus(focus, focusSmooth);
    }
  }

  checkViewReady(viewUpdate: boolean): boolean {
    console.log('checkViewReady before', viewUpdate);
    if (viewUpdate) {
      const refIxs = Array.from(this.activeElements.keys())
        .map((realIx) => this.getArrayIndex(realIx))
        .sort();
      const allReady = refIxs.every((corrIx, ix) => {
        if (corrIx !== ix) {
          console.log(
            corrIx,
            ix,
            Array.from(this.activeElements.keys()),
            refIxs,
            this.props.correction,
            this.props.offset,
            this.props.currentIx,
          );
          this.debugString();
        }
        return corrIx === ix;
      });
      if (allReady) {
        this.setState({
          viewUpdate: false,
        });
        viewUpdate = false;
      }
    }
    console.log('checkViewReady after', viewUpdate);
    return viewUpdate;
  }

  focus(focusIx: VIndex, smooth: boolean): void {
    const item = this.activeElements.get(focusIx);
    if (item !== undefined) {
      item.scrollIntoView({
        behavior: smooth ? 'smooth' : 'auto',
        block: 'start',
        inline: 'nearest',
      });
      this.setState({ focusIx });
    }
  }

  getRealIndex(index: number): VIndex {
    return (this.props.offset + index) as VIndex;
  }

  getArrayIndex(index: VIndex): VArrIndex {
    return (index + this.props.correction) as VArrIndex;
  }

  isValidArrayIndex(index: VIndex): boolean {
    const { order } = this.props;
    const arrIx = this.getArrayIndex(index);
    return arrIx < order.length || arrIx >= 0;
  }

  lineKey(index: VIndex | undefined): LineKey | undefined {
    if (index === undefined) {
      return undefined;
    }
    if (!this.isValidArrayIndex(index)) {
      return undefined;
    }
    const { order } = this.props;
    const correctedIndex = this.getArrayIndex(index);
    return order[correctedIndex];
  }

  getHIndex(index: VIndex): LineIndex {
    const { currentLineIxs } = this.props;
    const lineKey = this.lineKey(index);
    if (lineKey === undefined) {
      return 0 as LineIndex;
    }
    const key = constructKey(lineKey);
    const res = currentLineIxs[key];
    if (res === undefined) {
      return 0 as LineIndex;
    }
    return res;
  }

  getHIndexAdjusted(index: VIndex): AdjustedLineIndex {
    const { locks } = this.props;
    const lineKey = this.lineKey(index);
    if (lineKey === undefined) {
      return 0 as AdjustedLineIndex;
    }
    const key = constructKey(lineKey);
    const res = this.getHIndex(index);
    const locked = locks[key];
    if (locked && res < 0) {
      return locked.index;
    }
    const lockedIx = (
      locked && locked.skipItem ? locked.index : res + 1
    ) as AdjustedLineIndex;
    return (res + (lockedIx > res ? 0 : 1)) as AdjustedLineIndex;
  }

  getRefCb(realIx: VIndex): RefCB {
    return (element) => {
      console.log('ref', realIx, element);
      if (element !== null) {
        this.activeElements.set(realIx, element);
      } else {
        this.activeElements.delete(realIx);
      }
      const { viewUpdate } = this.state;
      this.checkViewReady(viewUpdate);
    };
  }

  handleUp = (event: React.MouseEvent<HTMLButtonElement>): void => {
    const { currentIx, dispatch } = this.props;
    dispatch(focusV({ focus: (currentIx - 1) as VIndex }));
    event.preventDefault();
  };

  handleDown = (event: React.MouseEvent<HTMLButtonElement>): void => {
    const { currentIx, dispatch } = this.props;
    dispatch(focusV({ focus: (currentIx + 1) as VIndex }));
    event.preventDefault();
  };

  requestRedraw = (): void => {
    console.groupCollapsed('V');
    console.log('request redraw V');
    console.trace();
    console.groupEnd();
    const { redraw } = this.state;
    this.setState({
      redraw: !redraw,
    });
  };

  render() {
    const {
      buttonSize,
      currentIx,
      getItem,
      getLink,
      height,
      radius,
      renderLink,
    } = this.props;
    const { itemCount, viewUpdate } = this.state;

    const render = (realIx: VIndex): JSX.Element | null => {
      const lineKey = this.lineKey(realIx);
      if (lineKey === undefined) {
        return null;
      }
      const link = getLink(
        toFullKey(lineKey, this.getHIndexAdjusted(realIx)),
        this.getHIndexAdjusted((realIx - 1) as VIndex),
        this.requestRedraw,
      );
      if (link === undefined) {
        return null;
      }
      const res = renderLink(link, buttonSize, radius);
      if (res === null) {
        return null;
      }
      return <ItemMid>{res}</ItemMid>;
    };

    const validRealIx = new Set(range(itemCount));
    Array.from(this.activeRefCbs.keys()).forEach((realIx) => {
      if (validRealIx.has(realIx)) {
        return;
      }
      this.activeRefCbs.delete(realIx);
    });

    return (
      <Outer ref={this.rootBox}>
        <Band ref={this.bandRef}>
          {range(itemCount).map((ix) => {
            const realIx = this.getRealIndex(ix);
            if (!this.isValidArrayIndex(realIx)) {
              return null;
            }
            let mRefCb = this.activeRefCbs.get(realIx);
            if (mRefCb === undefined) {
              mRefCb = this.getRefCb(realIx);
              this.activeRefCbs.set(realIx, mRefCb);
            }
            const refCb = mRefCb;
            const vPosType =
              currentIx === realIx
                ? VPosType.InFocus
                : currentIx < realIx
                ? VPosType.BelowFocus
                : VPosType.AboveFocus;
            return (
              <Item
                key={realIx}
                ref={refCb}
                isCurrent={currentIx === realIx}>
                {ix > 0 ? render(realIx) : null}
                {getItem(this.lineKey(realIx), height, viewUpdate, vPosType)}
              </Item>
            );
          })}
        </Band>
        <NavButtonUp
          buttonSize={buttonSize}
          onClick={this.handleUp}>
          ^
        </NavButtonUp>
        <NavButtonDown
          buttonSize={buttonSize}
          onClick={this.handleDown}>
          v
        </NavButtonDown>
      </Outer>
    );
  }
} // Vertical

const connector = connect((state: RootState) => ({
  correction: state.lineState.vCorrection,
  currentIx: state.lineState.vCurrentIx,
  currentLineIxs: state.lineState.currentLineIxs,
  focus: state.lineState.vFocus,
  focusSmooth: state.lineState.vFocusSmooth,
  offset: state.lineState.vOffset,
  order: state.lineState.vOrder,
  locks: state.lineState.locks,
}));
export default connector(Vertical);

type ConnectVertical = ConnectedProps<typeof connector>;
