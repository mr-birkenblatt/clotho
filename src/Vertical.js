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

const Pad = styled.div`
  height: ${props => props.padSize}px;
  width: 100%;
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
      needViews: true,
      awaitOrderChange: null,
      awaitCurrentChange: null,
      redraw: false,
      scrollInit: false,
      isScrolling: false,
      focusIx: 0,
    };
    this.rootBox = React.createRef();
    this.bandRef = React.createRef();
    this.activeRefs = {};
    this.activeView = {};
    this.currentVisible = new Set();
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
    return [...this.currentVisible.values()].reduce(
      (cur, val) => {
        if (cur === null) {
          return val;
        }
        return Math.min(cur, val);
      },
      null);
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
      awaitCurrentChange,
      awaitOrderChange,
      focusIx,
      isScrolling,
      itemCount,
      needViews,
      scrollInit,
    } = this.state;
    let isOrderChange = awaitOrderChange !== null;
    let isCurrentChange = awaitCurrentChange !== null;

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

    if (!isOrderChange && focus === focusIx) {
      if (currentIx + correction >= order.length - 2) {
        console.log(`addLine ${order.length} back`);
        dispatch(addLine({
          lineName: getChildLine(order[order.length - 1]),
          isBack: true,
        }));
      } else if (currentIx + correction <= 0) {
        console.log(`addLine ${order.length} front`);
        dispatch(addLine({
          lineName: getParentLine(order[0]),
          isBack: false,
        }));
      }
      this.setState({
        awaitOrderChange: order,
      });
      isOrderChange = true;
    }
    if (isOrderChange
        && awaitOrderChange !== null
        && awaitOrderChange !== order) {
      console.log(
        `order change confirmed ${awaitOrderChange.length} ${order.length}`);
      this.setState({
        awaitOrderChange: null,
      });
    }

    console.log(
      "curup",
      "isNotScrolling", !isScrolling,
      "awaitCurrentChange", !isCurrentChange,
      "awaitOrderChange", !isOrderChange,
      "focus", focus === focusIx);
    if (!isScrolling
        && !isCurrentChange
        && !isOrderChange
        && focus === focusIx) {
      const computedIx = this.computeIx();
      if (computedIx !== null && computedIx !== currentIx) {
        dispatch(setVCurrentIx({
          vIndex: computedIx,
          hIndex: this.getHIndex(computedIx),
          isParent: this.isParent(computedIx),
          lineName: this.lineName(computedIx),
        }));
        this.setState({
          updateCurrentIx: false,
          awaitCurrentChange: currentIx,
        });
        isCurrentChange = true;
      }
    }
    if (isCurrentChange
        && awaitCurrentChange !== null
        && awaitCurrentChange !== currentIx) {
      console.log(
        `currentIx change confirmed ${awaitCurrentChange} ${currentIx}`);
      this.setState({
        awaitCurrentChange: null,
      });
    }

    const needViewsNew = this.updateViews(prevProps, prevState);
    if (needViews !== needViewsNew) {
      this.setState({
        needViews: needViewsNew,
      });
    }

    if (!isCurrentChange && !isOrderChange && focus !== focusIx) {
      this.focus(focus - correction, focusSmooth);
    }
  }

  updateViews(prevProps, prevState) {
    const { offset, order } = this.props;
    const { itemCount, needViews } = this.state;
    let needViewsNew = needViews;
    if (prevProps.offset !== offset || prevState.itemCount !== itemCount
        || prevProps.order !== order) {
      Object.keys(this.activeView).forEach(realIx => {
        if (realIx < offset || realIx >= offset + itemCount) {
          if (this.activeView[realIx]) {
            this.activeView[realIx].disconnect();
          }
          delete this.activeView[realIx];
          console.log(`delete current index ${realIx} ${this.computeIx()}`);
          this.currentVisible.delete(realIx);
        }
      });
      Object.keys(this.activeRefs).forEach(realIx => {
        if (realIx < offset || realIx >= offset + itemCount) {
          delete this.activeRefs[realIx];
        }
      });
      [...Array(itemCount).keys()].forEach(ix => {
        const realIx = this.getRealIndex(ix);
        if (!this.activeRefs[realIx]) {
          this.activeRefs[realIx] = React.createRef();
          needViewsNew = true;
        }
      });
    }

    const that = this;

    function createObserver(ref, index) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            that.currentVisible.add(index);
            console.log(`set current index ${index} ${that.computeIx()}`);
          } else {
            that.currentVisible.delete(index);
            console.log(`remove current index ${index} ${that.computeIx()}`);
          }
          that.setState({
            updateCurrentIx: true,
          });
        });
      }, {
        root: that.rootBox.current,
        rootMargin: "0px",
        threshold: 0.9,
      });
      observer.observe(ref.current);
      return observer;
    }

    if (needViews) {
      [...Array(itemCount).keys()].forEach(ix => {
        const realIx = this.getRealIndex(ix);
        const curRef = this.activeRefs[realIx];
        if (curRef.current) {
          if (this.rootBox.current && !this.activeView[realIx]) {
            this.activeView[realIx] = createObserver(curRef, realIx);
          }
        }
      });
    }
    return needViewsNew;
  }

  focus(focusIx, smooth) {
    const { correction } = this.props;
    console.log(this.activeRefs, focusIx);
    const item = this.activeRefs[focusIx];
    console.log(
      `scroll to ${focusIx} smooth:${smooth} ` +
      `success:${!!(item && item.current)}`);
    if (item && item.current) {
      item.current.scrollIntoView({
        behavior: smooth ? "smooth" : "auto",
        block: "center",
        inline: "nearest",
      });
      this.setState({ focusIx: focusIx + correction });
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
      padSize,
    } = this.props;
    const { itemCount } = this.state;
    return (
      <Outer ref={this.rootBox}>
        <Band ref={this.bandRef}>
          <Pad padSize={padSize} />
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
