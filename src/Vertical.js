import React, { PureComponent } from "react";
import { connect } from "react-redux";
import styled from "styled-components";
import { constructKey, setVCurrentIx, addLine } from "./lineStateSlice";

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
  height: ${props => props.padSize};
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
      redraw: false,
      awaitOrderChange: true,
    };
    this.rootBox = React.createRef();
    this.activeRefs = {};
    this.activeView = {};
    this.currentVisible = new Set();
  }

  componentDidMount() {
    this.componentDidUpdate({}, {});
  }

  componentDidUpdate(prevProps, prevState) {
    const { offset, order, getChildLine, currentIx, dispatch } = this.props;
    const { itemCount, needViews, awaitOrderChange } = this.state;
    console.log(
      "cix", currentIx, "order", order.length,
      "offset", offset, "itemCount", itemCount);
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
          const { currentIx, dispatch } = that.props;
          if (entry.isIntersecting) {
            that.currentVisible.add(index);
            console.log(`set current index ${index}`);
          } else {
            console.log(`remove current index ${index}`);
            that.currentVisible.delete(index);
          }
          const computedIx = [...that.currentVisible.values()].reduce(
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
              hIndex: that.getHIndex(computedIx),
              isParent: that.isParent(computedIx),
              lineName: that.lineName(computedIx),
            }));
          }
        });
      }, {
        root: that.rootBox.current,
        rootMargin: "0px",
        threshold: 0.8,
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
    const key = constructKey(this.isParent(index), this.lineName(index));
    const res = this.props.currentLineIxs[key];
    if (res === undefined) {
      return 0;
    }
    return res;
  }

  render() {
    const { padSize, height, getItem, order, currentIx } = this.props;
    const { itemCount } = this.state;
    return (
      <Outer ref={this.rootBox}>
        <Band>
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
