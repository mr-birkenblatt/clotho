import React, { PureComponent } from "react";
import { connect } from "react-redux";
import styled from "styled-components";
import {
  constructKey,
  setVCurrentIx,
  addLine,
  changeVOffset,
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
      awaitOrderChange: true,
      awaitOffsetChange: false,
      redraw: false,
      scrollInit: false,
      pendingPad: false,
      isScrolling: false,
      updateCurrentIx: false,
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

  componentDidUpdate(prevProps, prevState) {
    const { offset, order, getChildLine, currentIx, dispatch } = this.props;
    const {
      itemCount,
      needViews,
      awaitOrderChange,
      awaitOffsetChange,
      scrollInit,
      pendingPad,
      isScrolling,
      updateCurrentIx,
    } = this.state;

    if (!scrollInit && this.bandRef.current) {
      this.bandRef.current.addEventListener(
        "scroll", this.handleScroll, { passive: true });
      this.setState({
        scrollInit: true,
      });
    }

    if (!isScrolling && updateCurrentIx) {
      const computedIx = [...this.currentVisible.values()].reduce(
        (cur, val) => {
          if (cur === null) {
            return val;
          }
          return Math.min(cur, val);
        },
        null);
      if (computedIx !== currentIx) {
        dispatch(setVCurrentIx({
          vIndex: computedIx,
          hIndex: this.getHIndex(computedIx),
          isParent: this.isParent(computedIx),
          lineName: this.lineName(computedIx),
        }));
      }
      this.setState({
        updateCurrentIx: false,
      });
    }

    if (!awaitOrderChange && (order.length - offset < itemCount
        || order.length < currentIx - offset + itemCount)) {
      console.log("addLine");
      dispatch(addLine({
        lineName: getChildLine(order[order.length - 1]),
        isBack: true,
      }));
      this.setState({
        awaitOrderChange: true,
      });
    }
    if (awaitOrderChange && prevProps.order !== order) {
      console.log("order change confirmed");
      this.setState({
        awaitOrderChange: false,
      });
    }

    if (pendingPad || !awaitOffsetChange) {
      if (isScrolling) {
        if (!pendingPad) {
          this.setState({
            pendingPad: true,
          });
        }
      } else {
        if (currentIx - offset >= itemCount - 2) {
          console.log(`offset inc change ${currentIx} ${offset}`);
          dispatch(changeVOffset({ isIncrease: true }));
          this.setState({
            awaitOffsetChange: true,
            pendingPad: false,
          });
        // } else if (currentIx - offset < 1 && offset > 0) {
        //   console.log(`offset dec change ${currentIx} ${offset}`);
        //   dispatch(changeVOffset({ isIncrease: false }));
        }
      }
    }
    if (awaitOffsetChange && prevProps.offset !== offset) {
      console.log("offset change confirmed");
      this.setState({
        awaitOffsetChange: false,
      })
    }

    const needViewsNew = this.updateViews(prevProps, prevState);
    if (needViews !== needViewsNew) {
      this.setState({
        needViews: needViewsNew,
      });
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
          console.log(`delete current index ${realIx}`);
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
            console.log(`set current index ${index}`);
          } else {
            console.log(`remove current index ${index}`);
            that.currentVisible.delete(index);
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

  getRealIndex(index) {
    return this.props.offset + index;
  }

  isParent(index) {
    return index <= this.props.currentIx;
  }

  lineName(index) {
    return this.props.order[index];
  }

  getHIndex(index) {
    const key = constructKey(this.lineName(index));
    const res = this.props.currentLineIxs[key];
    if (res === undefined) {
      return 0;
    }
    return res;
  }

  requestRedraw = () => {
    const { redraw } = this.state;
    this.setState({
      redraw: !redraw,
    });
  }

  render() {
    const { padSize, height, getItem, order, currentIx } = this.props;
    const { itemCount } = this.state;
    return (
      <Outer ref={this.rootBox}>
        <Band ref={this.bandRef}>
          <Pad padSize={padSize} />
          {
            [...Array(itemCount).keys()].map(ix => {
              const realIx = this.getRealIndex(ix);
              if (realIx >= order.length) {
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
  currentIx: state.lineState.vCurrentIx,
  currentLineIxs: state.lineState.currentLineIxs,
  height: state.lineState.vSize,
  offset: state.lineState.vOffset,
  order: state.lineState.vOrder,
  padSize: state.lineState.vPadSize,
}))(Vertical);
