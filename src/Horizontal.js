import React, { PureComponent } from "react";
import styled from "styled-components";

const Outer = styled.div`
  position: relative;
  top: 0;
  left: 0;
  width: 100%;
  height: ${props => props.itemHeight}px;
  background-color: red;
`;

const Overlay = styled.div`
  height: 100%;
  position: absolute;
  left: 0;
  top: 0;
  display: flex;
  justify-content: space-between;
  flex-direction: row;
  flex-wrap: nowrap;
  width: 100%;
  pointer-events: none;
  opacity: 0.8;
`;

const NavButton = styled.button`
  width: ${props => props.buttonSize}px;
  height: 100%;
  pointer-events: auto;
`;

const Band = styled.div`
  display: inline-block;
  height: 100%;
  width: ${props => props.itemWidth}px;
  background-color: green;
  white-space: nowrap;
  overflow-x: scroll;
  overflow-y: hidden;

  &::-webkit-scrollbar {
    display: none;
  }

  -ms-overflow-style: none;
  scrollbar-width: none;
  scroll-snap-type: x mandatory;
`;

const Item = styled.span`
  display: inline-block;
  width: ${props => props.itemWidth}px;
  height: 100%;
  scroll-snap-align: center;
  background-color: cornflowerblue;
`;

const ItemContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  width: ${props => props.itemWidth - 2 * props.itemPadding}px;
  height: calc(100% - ${props => 2 * props.itemPadding}px);
  margin: ${props => props.itemPadding}px auto;
  border-radius: ${props => props.itemRadius}px;
  padding: ${props => -props.itemPadding}px;
  background-color: pink;
`;

const Pad = styled.div`
  display: inline-block;
  width: ${props => props.padSize}px;
  height: 1px;
`;

// TODO: lock in item
// TODO: vertical
// TODO: flexible height


export default class Horizontal extends PureComponent {
  constructor(props) {
    super(props);
    this.state = {
      offset: 0,
      currentIx: 0,
      itemCount: 5,
      padSize: 0,
      needViews: true,
      redraw: false,
      locked: null,
    };
    this.activeRefs = {};
    this.activeView = {};
    this.lockedRef = React.createRef();
    this.lockedView = null;
    this.rootView = React.createRef();
    this.updateViews({});
  }

  componentDidMount() {
    this.componentDidUpdate({}, {});
  }

  componentDidUpdate(prevProps, prevState) {
    const {
      currentIx,
      itemCount,
      needViews,
      offset,
      padSize,
    } = this.state;
    const { itemWidth } = this.props;
    if (prevState.currentIx !== currentIx) {
      let newOffset = offset;
      let newPadSize = padSize;
      while (newOffset > 0 && currentIx < newOffset + itemCount * 0.5 - 1) {
        newOffset -= 1;
        newPadSize -= itemWidth;
      }
      while (currentIx > newOffset + itemCount * 0.5) {
        newOffset += 1;
        newPadSize += itemWidth;
      }
      this.setState({
        offset: newOffset,
        padSize: newPadSize,
      });
    }
    const needViewsNew = this.updateViews(prevState);
    if (needViews !== needViewsNew) {
      this.setState({
        needViews: needViewsNew,
      });
    }
  }

  updateViews(prevState) {
    const { offset, itemCount, needViews } = this.state;
    let needViewsNew = needViews;
    if (prevState.offset !== offset || prevState.itemCount !== itemCount) {
      Object.keys(this.activeView).forEach(realIx => {
        if (realIx < offset || realIx >= offset + itemCount) {
          if (this.activeView[realIx]) {
            this.activeView[realIx].disconnect();
          }
          delete this.activeView[realIx];
        }
      });
      Object.keys(this.activeRefs).forEach(realIx => {
        if (realIx < offset || realIx >= offset + itemCount) {
          delete this.activeRefs[realIx];
        }
      });
      [...Array(itemCount).keys()].forEach(ix => {
        const realIx = offset + ix;
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
          if (!entry.isIntersecting) {
            return;
          }
          that.setState({
            currentIx: index,
          });
        });
      }, {
        root: that.rootView.current,
        rootMargin: "0px",
        threshold: 1.0,
      });
      observer.observe(ref.current);
      return observer;
    }

    if (needViews) {
      [...Array(itemCount).keys()].forEach(ix => {
        const realIx = offset + ix;
        const curRef = this.activeRefs[realIx];
        if (curRef.current && this.rootView.current) {
          if (!this.activeView[realIx]) {
            this.activeView[realIx] = createObserver(curRef, realIx);
          }
        }
      });
    }
    if (!this.lockedView && this.lockedRef.current && this.rootView.current) {
      this.lockedView = createObserver(this.lockedRef, -1);
    }
    return needViewsNew;
  }

  focus(focusIx, smooth) {
    const item = focusIx < 0 ? this.lockedRef : this.activeRefs[focusIx];
    if (item && item.current) {
      item.current.scrollIntoView(
        smooth ? { behavior: "smooth", block: "center" } : {});
    }
  }

  getContent(isParent, lineName, index) {
    const { getItem } = this.props;
    const { locked } = this.state;
    if (locked && index < 0) {
      return this.getContent(locked.isParent, locked.lineName, locked.index);
    }
    return getItem(isParent, lineName, index, (hasItem, content) => {
      return `${hasItem ? content : "loading"} [${index}]`;
    }, this.requestRedraw);
  }

  adjustIndex(index) {
    const { offset, itemCount, locked } = this.state;
    const lockedIx = locked && locked.skipItem
      ? locked.index : offset + itemCount;
    return index + (lockedIx > index ? 0 : 1);
  }

  lockCurrent = (skipItem) => {
    const { currentIx } = this.state;
    if (currentIx < 0) {
      return;
    }
    const { isParent, lineName } = this.props;
    const locked = {
      isParent,
      lineName,
      index: this.adjustIndex(currentIx),
      skipItem,
    };
    this.setState({
      locked,
      currentIx: -1,
      padSize: 0,
      offset: 0,
      needViews: false,
    }, () => {
      this.focus(-1, false);
    });
  }

  requestRedraw = () => {
    const { redraw } = this.state;
    this.setState({
      redraw: !redraw,
    });
  }

  handleLeft = (event) => {
    const { currentIx } = this.state;
    this.focus(currentIx - 1, true);
    event.preventDefault();
  }

  handleRight = (event) => {
    const { currentIx } = this.state;
    this.focus(currentIx + 1, true);
    event.preventDefault();
  }

  render() {
    const {
      itemWidth,
      itemHeight,
      itemRadius,
      buttonSize,
      itemPadding,
      isParent,
      lineName,
    } = this.props;
    const { locked, offset, itemCount, padSize } = this.state;
    const offShift = offset < 0 ? -offset : 0;
    return (
      <Outer itemHeight={itemHeight} ref={this.rootView}>
        <Overlay>
          <NavButton buttonSize={buttonSize} onClick={this.handleLeft}>
            &lt;
          </NavButton>
          <NavButton buttonSize={buttonSize} onClick={this.handleRight}>
            &gt;
          </NavButton>
        </Overlay>
        <Band itemWidth={itemWidth}>
          { locked ? (
            <Item
                itemWidth={itemWidth}>
              <ItemContent
                itemPadding={itemPadding}
                itemWidth={itemWidth}
                itemRadius={itemRadius}
                ref={this.lockedRef}>
              {this.getContent(isParent, lineName, -1)}
              </ItemContent>
            </Item>
          ) : null }
          <Pad padSize={padSize} />
          {
            [...Array(itemCount - offShift).keys()].map(ix => {
              const realIx = offset + ix + offShift;
              return (
                <Item
                    key={realIx}
                    itemWidth={itemWidth}>
                  <ItemContent
                    itemPadding={itemPadding}
                    itemWidth={itemWidth}
                    itemRadius={itemRadius}
                    ref={this.activeRefs[realIx]}>
                  {
                    this.getContent(
                      isParent, lineName, this.adjustIndex(realIx))
                  }
                  </ItemContent>
                </Item>
              );
            })
          }
        </Band>
      </Outer>
    );
  }
} // Horizontal
