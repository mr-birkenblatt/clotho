import React, { PureComponent } from 'react';
import { connect, ConnectedProps } from 'react-redux';
import styled from 'styled-components';
import { range } from '../misc/util';
import { RootState } from '../store';
import {
  constructKey,
  setVCurrentIx,
  addLine,
  focusV,
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

interface VerticalProps extends ConnectVertical {
  height: number;
  buttonSize: number;
  radius: number;
  getChildLine: (
    lineName: string,
    index: number,
    callback: (child: string) => void,
  ) => void;
  getParentLine: (
    lineName: string,
    index: number,
    callback: (parent: string) => void,
  ) => void;
  getItem: (
    isParent: boolean,
    lineName: string,
    height: number,
  ) => JSX.Element;
  getLink: (
    isParent: boolean,
    lineName: string,
    index: number,
    readyCb: ReadyCB,
  ) => Link | undefined;
  renderLink: (link: Link, buttonSize: number, radius: number) => JSX.Element;
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
  focusIx: number;
  viewUpdate: boolean;
};

type EmptyVerticalState = {
  itemCount: undefined;
};

class Vertical extends PureComponent<VerticalProps, VerticalState> {
  rootBox: React.RefObject<HTMLDivElement>;
  bandRef: React.RefObject<HTMLDivElement>;
  activeRefs: Map<number, React.RefObject<HTMLDivElement>>;
  awaitOrderChange: string[] | undefined;
  awaitCurrentChange: number | undefined;

  constructor(props: VerticalProps) {
    super(props);
    this.state = {
      itemCount: 4,
      redraw: false,
      scrollInit: false,
      isScrolling: false,
      focusIx: 0,
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
    // FIXME double check all range usages
    const ord = range(maxIx - minIx).reduce((cur, val) => {
      const isOrd =
        val + minIx >= rangeOrder[0] && val + minIx < rangeOrder[1];
      return `${cur}${isOrd ? '#' : '_'}`;
    }, '');
    const arr = range(maxIx - minIx).reduce((cur, val) => {
      const isArr =
        val + minIx >= rangeArray[0] && val + minIx < rangeArray[1];
      return `${cur}${isArr ? '#' : '_'}`;
    }, '');
    const cur = range(maxIx - minIx).reduce((cur, val) => {
      const isCur = val + minIx === adjIndex;
      return `${cur}${isCur ? 'X' : '_'}`;
    }, '');
    console.group('VState');
    console.log(`ORD ${ord}`);
    console.log(`ARR ${arr}`);
    console.log(`CUR ${cur}`);
    console.groupEnd();
  }

  computeIx(): number | undefined {
    const { itemCount } = this.state;
    const out = range(itemCount).reduce(
      (res: undefined[] | number[], ix: number) => {
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
        getChildLine(
          order[order.length - 1],
          this.getHIndex(order.length - 1, true),
          (child) => {
            dispatch(
              addLine({
                lineName: child,
                isBack: true,
              }),
            );
          },
        );
        this.awaitOrderChange = order;
      } else if (currentIx + correction <= 0) {
        getParentLine(order[0], this.getHIndex(0, true), (parent) => {
          dispatch(
            addLine({
              lineName: parent,
              isBack: false,
            }),
          );
        });
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
      if (computedIx !== undefined && computedIx !== currentIx) {
        dispatch(
          setVCurrentIx({
            vIndex: computedIx,
            hIndex: this.getHIndex(computedIx, false),
            isParent: this.isParent(computedIx),
            lineName: this.lineName(computedIx),
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

  focus(focusIx: number, smooth: boolean): void {
    const item = this.activeRefs.get(focusIx);
    if (item && item.current) {
      const curItem = item.current;
      curItem.scrollIntoView({
        behavior: smooth ? 'smooth' : 'auto',
        block: 'start',
        inline: 'nearest',
      });
      this.setState({ focusIx });
    }
  }

  getRealIndex(index: number): number {
    return this.props.offset + index;
  }

  isParent(index: number): boolean {
    return index <= this.props.currentIx;
  }

  lineName(index: number): string {
    return this.props.order[index + this.props.correction];
  }

  getHIndex(index: number, adjust: boolean): number {
    const { locks, currentLineIxs } = this.props;
    const key = constructKey(this.lineName(index));
    const res = currentLineIxs[key];
    if (res === undefined) {
      return 0;
    }
    if (!adjust) {
      return res;
    }
    const locked = locks[key];
    if (locked && res < 0) {
      return locked.index;
    }
    const lockedIx = locked && locked.skipItem ? locked.index : res + 1;
    return res + (lockedIx > res ? 0 : 1);
  }

  handleUp = (event: React.MouseEvent<HTMLButtonElement>): void => {
    const { currentIx, dispatch } = this.props;
    dispatch(focusV({ focus: currentIx - 1 }));
    event.preventDefault();
  };

  handleDown = (event: React.MouseEvent<HTMLButtonElement>): void => {
    const { currentIx, dispatch } = this.props;
    dispatch(focusV({ focus: currentIx + 1 }));
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

    const render = (realIx: number): JSX.Element | string => {
      const link = getLink(
        this.isParent(realIx),
        this.lineName(realIx),
        this.getHIndex(realIx, true),
        this.requestRedraw,
      );
      if (link === undefined) {
        return '...';
      }
      return renderLink(link, buttonSize, radius);
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
                {ix > 0 ? <ItemMid>{render(realIx)}</ItemMid> : null}
                {getItem(this.isParent(realIx), this.lineName(realIx), height)}
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
