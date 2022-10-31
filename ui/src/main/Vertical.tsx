import React, { PureComponent } from 'react';
import { connect, ConnectedProps } from 'react-redux';
import styled from 'styled-components';
import {
  AdjustedLineIndex,
  FullKey,
  LineKey,
  Link,
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

export type LinkCB = (
  fullLinkKey: FullKey,
  parentIndex: AdjustedLineIndex,
  readyCb: ReadyCB,
) => Link | undefined;
export type ChildLineCB = (
  lineKey: LineKey,
  index: AdjustedLineIndex,
  childIndex: AdjustedLineIndex,
  callback: (child: LineKey) => void,
) => void;
export type ParentLineCB = (
  lineKey: LineKey,
  index: AdjustedLineIndex,
  parentIndex: AdjustedLineIndex,
  callback: (parent: LineKey) => void,
) => void;
export type VItemCB = (lineKey: LineKey | undefined, height: number) => JSX.Element | null;
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

class Vertical extends PureComponent<VerticalProps, VerticalState> {
  rootBox: React.RefObject<HTMLDivElement>;
  bandRef: React.RefObject<HTMLDivElement>;
  activeRefs: Map<VIndex, React.RefObject<HTMLDivElement>>;
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
    this.activeRefs = new Map();
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
    const adjIndex = correction + currentIx;
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
    const { itemCount } = this.state;

    type PosAndIndex = undefined[] | [number, VIndex];

    const out = range(itemCount).reduce(
      (res: PosAndIndex, ix: number): PosAndIndex => {
        const realIx = this.getRealIndex(ix);
        const curRef = this.activeRefs.get(realIx);
        if (curRef === undefined) {
          return res;
        }
        const cur = curRef.current;
        if (cur === null) {
          return res;
        }
        const bounds = cur.getBoundingClientRect();
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
      prevProps.order !== order ||
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

    if (this.awaitOrderChange === null) {
      if (currentIx + correction >= order.length - 2) {
        const lastIx = (order.length - 1) as VIndex;
        const newIx = (lastIx + 1) as VIndex;
        getChildLine(
          order[lastIx],
          this.getHIndexAdjusted(lastIx),
          this.getHIndexAdjusted(newIx),
          (child) => {
            dispatch(
              addLine({
                lineKey: child,
                isBack: true,
              }),
            );
          },
        );
        this.awaitOrderChange = order;
      } else if (currentIx + correction <= 0) {
        const firstIx = 0 as VIndex;
        const newIx = (firstIx - 1) as VIndex;
        getParentLine(
          order[firstIx],
          this.getHIndexAdjusted(firstIx),
          this.getHIndexAdjusted(newIx),
          (parent) => {
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
      this.awaitOrderChange !== order
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
      const computedLine = this.lineKey(computedIx);
      if (computedIx !== undefined && computedLine !== undefined && computedIx !== currentIx) {
        dispatch(
          setVCurrentIx({
            vIndex: computedIx,
            hIndex: this.getHIndexAdjusted(computedIx),
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

    this.updateViews(prevProps, prevState);

    if (
      this.awaitCurrentChange === undefined &&
      this.awaitOrderChange === undefined &&
      focus !== focusIx &&
      !viewUpdate
    ) {
      this.focus(focus, focusSmooth);
    }
  }

  updateViews(
    prevProps: VerticalProps | EmptyVerticalProps,
    prevState: VerticalState | EmptyVerticalState,
  ): void {
    const { offset, order, currentIx } = this.props;
    const { itemCount, viewUpdate } = this.state;

    let newViewUpdate = false;
    if (
      prevProps.offset !== offset ||
      prevState.itemCount !== itemCount ||
      prevProps.order !== order ||
      prevProps.currentIx !== currentIx
    ) {
      Array.from(this.activeRefs.keys()).forEach((realIx) => {
        if (realIx < offset || realIx >= offset + itemCount) {
          this.activeRefs.delete(realIx);
          newViewUpdate = true;
        }
      });
      range(itemCount).forEach((ix) => {
        const realIx = this.getRealIndex(ix);
        if (!this.activeRefs.has(realIx)) {
          this.activeRefs.set(realIx, React.createRef());
          newViewUpdate = true;
        }
      });
    }
    if (newViewUpdate && !viewUpdate) {
      this.setState({
        viewUpdate: true,
      });
    } else if (viewUpdate) {
      const allReady = Array.from(this.activeRefs.entries()).reduce(
        (cur, val) => {
          val[1].current === null && console.log('ref missing', val);
          return cur && val[1].current !== null;
        },
        true,
      );
      if (allReady) {
        // NOTE: for debugging
        console.log(
          Array.from(this.activeRefs.values()).map((val) => val.current),
        );
        this.setState({
          viewUpdate: false,
        });
      } else {
        // NOTE: careful! can end in an infinite loop if elements are not
        // filled up correctly.
        setTimeout(() => {
          // this.requestRedraw();
        }, 100);
      }
    }
  }

  focus(focusIx: VIndex, smooth: boolean): void {
    const item = this.activeRefs.get(focusIx);
    if (item !== undefined && item.current !== null) {
      const curItem = item.current;
      curItem.scrollIntoView({
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

  lineKey(index: VIndex | undefined): LineKey | undefined {
    if (index === undefined) {
      return undefined;
    }
    const { order } = this.props;
    const correctedIndex = index + this.props.correction;
    if (correctedIndex < 0 || correctedIndex >= order.length) {
      return undefined;
    }
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
      correction,
      currentIx,
      getItem,
      getLink,
      height,
      order,
      radius,
      renderLink,
    } = this.props;
    const { itemCount } = this.state;

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

    return (
      <Outer ref={this.rootBox}>
        <Band ref={this.bandRef}>
          {range(itemCount).map((ix) => {
            const realIx = this.getRealIndex(ix);
            if (
              realIx + correction >= order.length ||
              realIx + correction < 0
            ) {
              return null;
            }
            return (
              <Item
                key={realIx}
                ref={this.activeRefs.get(realIx)}
                isCurrent={currentIx === realIx}>
                {ix > 0 ? render(realIx) : null}
                {getItem(this.lineKey(realIx), height)}
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
