import React, { PureComponent } from "react";
import { connect } from "react-redux";
import styled from "styled-components";
import {
  constructKey,
  setVCurrentIx,
  addLine,
  focusV,
} from "./lineStateSlice";

const Outer = styled.div`
  position: relative;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
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
  background-color: ${props => props.isCurrent ? "blue" : "cornflowerblue"};
`;

class Vertical extends PureComponent {
  constructor(props) {
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
    this.activeRefs = {};
    this.awaitOrderChange = null;
    this.awaitCurrentChange = null;
  }

  componentDidMount() {
    this.componentDidUpdate({}, {});
  }

  componentWillUnmount() {
    if (this.bandRef.current) {
      this.bandRef.current.removeEventListener("scroll", this.handleScroll);
    }
  }

  handleScroll = () => {
    const band = this.bandRef.current;
    if (!band) {
      return;
    }
    let startTime = null;
    let startScroll = band.scrollTop;
    const that = this;

    function checkScroll(time) {
      const isScrolling = band.scrollTop !== startScroll;
      if (startTime === null) {
        startTime = time;
      }
      const update = time - startTime >= 100 || isScrolling;
      if (update) {
        if (that.state.isScrolling !== isScrolling) {
          console.log(`isScrollV ${isScrolling}`);
          that.setState({ isScrolling });
        }
      } else {
        requestAnimationFrame(checkScroll);
      }
    }

    requestAnimationFrame(checkScroll);
  }

  debugString() {
    const { offset, order, currentIx, correction } = this.props;
    const { itemCount } = this.state;
    const rangeOrder = [0, order.length];
    const rangeArray = [correction + offset, correction + offset + itemCount];
    const adjIndex = correction + currentIx;
    const minIx = Math.min(rangeOrder[0], rangeArray[0], adjIndex);
    const maxIx = Math.max(rangeOrder[1], rangeArray[1], adjIndex);
    const ord = [...Array(maxIx - minIx).keys()].reduce((cur, val) => {
      const isOrd = val + minIx >= rangeOrder[0] && val + minIx < rangeOrder[1];
      return `${cur}${isOrd ? '#' : '_'}`;
    }, '');
    const arr = [...Array(maxIx - minIx).keys()].reduce((cur, val) => {
      const isArr = val + minIx >= rangeArray[0] && val + minIx < rangeArray[1];
      return `${cur}${isArr ? '#' : '_'}`;
    }, '');
    const cur = [...Array(maxIx - minIx).keys()].reduce((cur, val) => {
      const isCur = val + minIx === adjIndex;
      return `${cur}${isCur ? 'X' : '_'}`;
    }, '');
    console.group("VState");
    console.log(`ORD ${ord}`);
    console.log(`ARR ${arr}`);
    console.log(`CUR ${cur}`);
    console.groupEnd();
  }

  computeIx() {
    const { itemCount } = this.state;
    const out = [...Array(itemCount).keys()].reduce((res, ix) => {
      const realIx = this.getRealIndex(ix);
      const curRef = this.activeRefs[realIx];
      if (curRef === null) {
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
      if (res[0] === null || res[0] >= top) {
        return [top, realIx];
      }
      return res;
    }, [null, null]);
    return out[1];
  }

  componentDidUpdate(prevProps, prevState) {
    const {
      correction,
      currentIx,
      dispatch,
      focus,
      focusSmooth,
      getChildLine,
      getParentLine,
      offset,
      order,
    } = this.props;
    const {
      focusIx,
      isScrolling,
      itemCount,
      scrollInit,
    } = this.state;
    let viewUpdate = this.state.viewUpdate;

    if (prevProps.currentIx !== currentIx
        || prevProps.offset !== offset
        || prevProps.order !== order
        || prevProps.correction !== correction
        || prevState.itemCount !== itemCount) {
      this.debugString();
    }

    if (!scrollInit && this.bandRef.current) {
      this.bandRef.current.addEventListener(
        "scroll", this.handleScroll, { passive: true });
      this.setState({
        scrollInit: true,
      });
    }

    if (this.awaitOrderChange === null) {
      if (currentIx + correction >= order.length - 2) {
        // console.log(`addLine ${order.length} back ${this.awaitOrderChange}`);
        dispatch(addLine({
          lineName: getChildLine(order[order.length - 1]),
          isBack: true,
        }));
        this.awaitOrderChange = order;
      } else if (currentIx + correction <= 0) {
        // console.log(`addLine ${order.length} front ${this.awaitOrderChange}`);
        dispatch(addLine({
          lineName: getParentLine(order[0]),
          isBack: false,
        }));
        this.awaitOrderChange = order;
      }
    }
    if (this.awaitOrderChange !== null && this.awaitOrderChange !== order) {
      // console.log(
      //   `order change confirmed ${this.awaitOrderChange.length} `
      //   +`${order.length}`);
      this.awaitOrderChange = null;
    }

    console.log(
      "curup",
      "isNotScrolling", !isScrolling,
      "isNotCurrentChange", this.awaitCurrentChange === null,
      "isNotOrderChange", this.awaitOrderChange === null,
      "focus", focus === focusIx,
      "awaitOrderChange", this.awaitOrderChange,
      "awaitCurrentChange", this.awaitCurrentChange,
      "notViewUpdate", !viewUpdate);
    if (!isScrolling
        && !viewUpdate
        && this.awaitCurrentChange === null
        && this.awaitOrderChange === null
        && focus === focusIx) {
      const computedIx = this.computeIx();
      if (computedIx !== null && computedIx !== currentIx) {
        dispatch(setVCurrentIx({
          vIndex: computedIx,
          hIndex: this.getHIndex(computedIx),
          isParent: this.isParent(computedIx),
          lineName: this.lineName(computedIx),
        }));
        this.awaitCurrentChange = currentIx;
        this.setState({
          viewUpdate: true,
        });
        viewUpdate = true;
      }
    }
    if (this.awaitCurrentChange !== null
        && this.awaitCurrentChange !== currentIx) {
      // console.log(
      //   `currentIx change confirmed ${this.awaitCurrentChange} ${currentIx}`);
      this.awaitCurrentChange = null;
    }

    this.updateViews(prevProps, prevState);

    if (this.awaitCurrentChange === null
        && this.awaitOrderChange === null
        && focus !== focusIx
        && !viewUpdate) {
      this.focus(focus, focusSmooth);
    }
  }

  updateViews(prevProps, prevState) {
    const { offset, order, current } = this.props;
    const { itemCount, viewUpdate } = this.state;

    let newViewUpdate = false;
    if (prevProps.offset !== offset || prevState.itemCount !== itemCount
        || prevProps.order !== order || prevProps.current !== current) {
      Object.keys(this.activeRefs).forEach(realIx => {
        if (realIx < offset || realIx >= offset + itemCount) {
          delete this.activeRefs[realIx];
          newViewUpdate = true;
        }
      });
      [...Array(itemCount).keys()].forEach(ix => {
        const realIx = this.getRealIndex(ix);
        if (!this.activeRefs[realIx]) {
          this.activeRefs[realIx] = React.createRef();
          newViewUpdate = true;
        }
      });
    }
    if (newViewUpdate && !viewUpdate) {
      this.setState({
        viewUpdate: true,
      });
    } else if (viewUpdate) {
      const allReady = Object.values(this.activeRefs).reduce((cur, val) => {
        return cur && val.current !== null;
      }, true);
      if (allReady) {
        console.log(Object.values(this.activeRefs).map((val) => val.current));
        this.setState({
          viewUpdate: false,
        });
      } else {
        setTimeout(() => { this.requestRedraw(); }, 10);
      }
    }
  }

  focus(focusIx, smooth) {
    console.log("focus", this.activeRefs, focusIx);
    const item = this.activeRefs[focusIx];
    console.log(
      `scroll to ${focusIx} smooth: ${smooth} ` +
      `success: ${!!(item && item.current)}`);
    if (item && item.current) {
      const curItem = item.current;
      console.log("doFocus", curItem, focusIx);
      const band = this.bandRef.current;
      // if (!smooth && band !== null) {
      //   setTimeout(() => {
      //     band.scrollTop = (focusIx - this.getRealIndex(0)) * this.props.height;
      //   }, 0);
      // } else {
      curItem.scrollIntoView({
        behavior: smooth ? "smooth" : "auto",
        block: "start",
        inline: "nearest",
      });
      // }
      setTimeout(() => {
        this.setState({ focusIx });
      }, 10);
    }
  }

  getRealIndex(index) {
    return this.props.offset + index;
  }

  isParent(index) {
    return index <= this.props.currentIx;
  }

  lineName(index) {
    return this.props.order[index + this.props.correction];
  }

  getHIndex(index) {
    const key = constructKey(this.lineName(index));
    const res = this.props.currentLineIxs[key];
    if (res === undefined) {
      return 0;
    }
    return res;
  }

  handleUp = (event) => {
    const { currentIx, dispatch } = this.props;
    dispatch(focusV({ index: currentIx - 1 }));
    event.preventDefault();
  }

  handleDown = (event) => {
    const { currentIx, dispatch } = this.props;
    dispatch(focusV({ index: currentIx + 1 }));
    event.preventDefault();
  }

  requestRedraw = () => {
    const { redraw } = this.state;
    this.setState({
      redraw: !redraw,
    });
  }

  render() {
    const {
      correction,
      currentIx,
      getItem,
      height,
      order,
    } = this.props;
    const { itemCount } = this.state;
    return (
      <Outer ref={this.rootBox}>
        <Band ref={this.bandRef}>
          {
            [...Array(itemCount).keys()].map(ix => {
              const realIx = this.getRealIndex(ix);
              if (realIx + correction >= order.length
                  || realIx + correction < 0) {
                return null;
              }
              return (
                <Item
                  key={realIx}
                  id={`id${realIx}`}
                  ref={this.activeRefs[realIx]}
                  isCurrent={currentIx === realIx}>
                {
                  getItem(this.isParent(realIx), this.lineName(realIx), height)
                }
                </Item>
              );
            })
          }
        </Band>
      </Outer>
    );
  }
} // Vertical

export default connect((state) => ({
  correction: state.lineState.vCorrection,
  currentIx: state.lineState.vCurrentIx,
  currentLineIxs: state.lineState.currentLineIxs,
  focus: state.lineState.vFocus,
  focusSmooth: state.lineState.vFocusSmooth,
  offset: state.lineState.vOffset,
  order: state.lineState.vOrder,
}))(Vertical);
