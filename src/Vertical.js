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

const IntersectBoxTop = styled.div`
  width: 100%;
  height: 30%;
  position: absolute;
  left: 0;
  top: 0;
  pointer-events: none;
`;

const IntersectBoxBottom = styled.div`
  width: 100%;
  height: 30%;
  position: absolute;
  left: 0;
  bottom: 0;
  pointer-events: none;
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
  background-color: cornflowerblue;
`;

class Vertical extends PureComponent {
  constructor(props) {
    super(props);
    this.state = {
      itemCount: 4,
      needViews: true,
      redraw: false,
    };
    this.topBox = React.createRef();
    this.bottomBox = React.createRef();
    this.activeRefs = {};
    this.activeViewTop = {};
    this.activeViewBottom = {};
  }

  componentDidMount() {
    this.componentDidUpdate({}, {});
  }

  componentDidUpdate(prevProps, prevState) {
    const { offset, order, getChildLine, currentIx, dispatch } = this.props;
    const { itemCount, needViews } = this.state;
    console.log(order.length, offset, itemCount);
    if (prevProps.order === order && (order.length - offset < itemCount
        || order.length < currentIx - offset + itemCount)) {
      console.log("addLine");
      dispatch(addLine({
        lineName: getChildLine(order[order.length - 1]),
        isBack: true,
      }));
    }
    const needViewsNew = this.updateViews(prevProps, prevState);
    if (needViews !== needViewsNew) {
      this.setState({
        needViews: needViewsNew,
      });
    }
  }

  updateViews(prevProps, prevState) {
    const { offset } = this.props;
    const { itemCount, needViews } = this.state;
    let needViewsNew = needViews;
    if (prevProps.offset !== offset || prevState.itemCount !== itemCount) {
      Object.keys(this.activeViewTop).forEach(realIx => {
        if (realIx < offset || realIx >= offset + itemCount) {
          if (this.activeViewTop[realIx]) {
            this.activeViewTop[realIx].disconnect();
          }
          delete this.activeViewTop[realIx];
        }
      });
      Object.keys(this.activeViewBottom).forEach(realIx => {
        if (realIx < offset || realIx >= offset + itemCount) {
          if (this.activeViewBottom[realIx]) {
            this.activeViewBottom[realIx].disconnect();
          }
          delete this.activeViewBottom[realIx];
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

    function createObserver(ref, index, top) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (!entry.isIntersecting) {
            return;
          }
          if (top) {
            const { dispatch } = that.props;
            dispatch(setVCurrentIx({
              vIndex: index,
              hIndex: that.getHIndex(index),
              isParent: that.isParent(index),
              lineName: that.lineName(index),
            }));
          } else {
            console.log(`nothing so far... bottom ${index}`);
          }
        });
      }, {
        root: top ? that.topBox.current : that.bottomBox.current,
        rootMargin: "0px",
        threshold: top ? 1.0 : 0.1,
      });
      observer.observe(ref.current);
      return observer;
    }

    if (needViews) {
      [...Array(itemCount).keys()].forEach(ix => {
        const realIx = this.getRealIndex(ix);
        const curRef = this.activeRefs[realIx];
        if (curRef.current) {
          if (this.topBox.current && !this.activeViewTop[realIx]) {
            this.activeViewTop[realIx] = createObserver(curRef, realIx, true);
          }
          if (this.bottomBox.current && !this.activeViewBottom[realIx]) {
            this.activeViewBottom[realIx] = createObserver(
              curRef, realIx, false);
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
    const key =constructKey(this.isParent(index), this.lineName(index));
    return this.props.currentLineIxs[key];
  }

  render() {
    const { padSize, height, getItem, order } = this.props;
    const { itemCount } = this.state;
    return (
      <Outer>
        <IntersectBoxTop ref={this.topBox} />
        <IntersectBoxBottom res={this.bottomBox} />
        <Band>
          <Pad padSize={padSize} />
          {
            [...Array(itemCount).keys()].map(ix => {
              const realIx = this.getRealIndex(ix);
              if (realIx >= order.length) {
                return null;
              }
              return (
                <Item key={realIx}>
                  {
                    getItem(
                      this.isParent(realIx),
                      this.lineName(realIx),
                      height)
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
