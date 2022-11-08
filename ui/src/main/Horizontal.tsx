import React, { PureComponent } from 'react';
import ReactMarkdown from 'react-markdown';
import { connect, ConnectedProps } from 'react-redux';
import styled from 'styled-components';
import {
  AdjustedLineIndex,
  FullKey,
  LineKey,
  ReadyCB,
  toFullKey,
} from '../misc/CommentGraph';
import { range } from '../misc/util';
import { RootState } from '../store';
import {
  constructKey,
  focusAt,
  LineIndex,
  LOCK_INDEX,
  setHCurrentIx,
} from './LineStateSlice';
import { VPosType } from './Vertical';

const Outer = styled.div`
  position: relative;
  top: 0;
  left: 0;
  width: 100%;
  height: ${(props: { itemHeight: number }) => props.itemHeight}px;
  // background-color: red;
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
  width: ${(props: { buttonSize: number }) => props.buttonSize}px;
  height: 100%;
  pointer-events: auto;
`;

const Band = styled.div<BandProps>`
  display: inline-block;
  height: 100%;
  width: ${(props) => props.itemWidth}px;
  opacity: ${(props) => (props.vPosType === VPosType.BelowFocus ? 0.5 : 1.0)};
  background-color: ${(props) =>
    props.vPosType === VPosType.BelowFocus ? 'green' : 'none'};
  filter: blur(${(props) => (props.isViewUpdate ? '1px' : '0')});
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
  width: ${(props: { itemWidth: number }) => props.itemWidth}px;
  height: 100%;
  scroll-snap-align: center;
  // background-color: cornflowerblue;
`;

const ItemContent = styled.div<ItemContentProps>`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  width: ${(props) => props.itemWidth - 2 * props.itemPadding}px;
  height: calc(100% - ${(props) => 2 * props.itemPadding}px);
  margin: ${(props) => props.itemPadding}px auto;
  border-radius: ${(props) => props.itemRadius}px;
  padding: ${(props) => -props.itemPadding}px;
  // background-color: pink;
  white-space: normal;
`;

const Pad = styled.div`
  display: inline-block;
  width: ${(props: { padSize: number }) => props.padSize}px;
  height: 1px;
`;

export type ItemCB = (
  fullLinkKey: FullKey,
  readyCb: ReadyCB,
) => string | undefined;

type BandProps = {
  itemWidth: number;
  isViewUpdate: boolean;
  vPosType: VPosType;
};

type ItemContentProps = {
  itemWidth: number;
  itemRadius: number;
  itemPadding: number;
};

interface HorizontalProps extends ConnectHorizontal {
  itemWidth: number;
  itemHeight: number;
  itemRadius: number;
  buttonSize: number;
  itemPadding: number;
  lineKey: LineKey;
  isViewUpdate: boolean;
  vPosType: VPosType;
  getItem: ItemCB;
}

type EmptyHorizontalProps = {
  currentLineIxs: undefined;
  currentLineFocus: undefined;
};

type HOffset = number & { _hOffset: void };

type HorizontalState = {
  offset: HOffset;
  itemCount: number;
  padSize: number;
  needViews: boolean;
  redraw: boolean;
  scrollInit: boolean;
  pendingPad: boolean;
  isScrolling: boolean;
};

type EmptyHorizontalState = {
  offset: undefined;
  itemCount: undefined;
};

class Horizontal extends PureComponent<HorizontalProps, HorizontalState> {
  activeRefs: Map<LineIndex, React.RefObject<HTMLDivElement>>;
  activeView: Map<LineIndex, IntersectionObserver>;
  lockedRef: React.RefObject<HTMLDivElement>;
  lockedView: IntersectionObserver | undefined;
  rootBox: React.RefObject<HTMLDivElement>;
  bandRef: React.RefObject<HTMLDivElement>;

  constructor(props: HorizontalProps) {
    super(props);
    this.state = {
      offset: 0 as HOffset,
      itemCount: 5,
      padSize: 0,
      needViews: true,
      redraw: false,
      scrollInit: false,
      pendingPad: false,
      isScrolling: false,
    };
    this.activeRefs = new Map();
    this.activeView = new Map();
    this.lockedRef = React.createRef();
    this.lockedView = undefined;
    this.rootBox = React.createRef();
    this.bandRef = React.createRef();
    this.updateViews({ offset: undefined, itemCount: undefined });
  }

  componentDidMount(): void {
    this.componentDidUpdate(
      { currentLineIxs: undefined, currentLineFocus: undefined },
      { offset: undefined, itemCount: undefined },
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
    const startScroll = band.scrollLeft;
    const that = this;

    function checkScroll(time: number) {
      const isScrolling = band.scrollLeft !== startScroll;
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

  componentDidUpdate(
    prevProps: HorizontalProps | EmptyHorizontalProps,
    prevState: HorizontalState | EmptyHorizontalState,
  ): void {
    const { lineKey, currentLineIxs, currentLineFocus, itemWidth } =
      this.props;
    const key = constructKey(lineKey);
    const currentIx = currentLineIxs[key];
    const lineFocus = currentLineFocus[key];
    const prevCurrentIx: LineIndex | undefined =
      prevProps.currentLineIxs && prevProps.currentLineIxs[key];
    const prevLineFocus: LineIndex | undefined =
      prevProps.currentLineFocus && prevProps.currentLineFocus[key];
    const {
      itemCount,
      needViews,
      offset,
      padSize,
      scrollInit,
      pendingPad,
      isScrolling,
    } = this.state;

    if (!scrollInit && this.bandRef.current) {
      this.bandRef.current.addEventListener('scroll', this.handleScroll, {
        passive: true,
      });
      this.setState({
        scrollInit: true,
      });
    }

    if (pendingPad || prevCurrentIx !== currentIx) {
      if (isScrolling) {
        if (!pendingPad) {
          this.setState({
            pendingPad: true,
          });
        }
      } else {
        let newOffset = offset;
        let newPadSize = padSize;
        while (newOffset > 0 && currentIx < newOffset + itemCount * 0.5 - 1) {
          newOffset = (newOffset - 1) as HOffset;
          newPadSize -= itemWidth;
        }
        while (currentIx > newOffset + itemCount * 0.5) {
          newOffset = (newOffset + 1) as HOffset;
          newPadSize += itemWidth;
        }
        this.setState({
          offset: newOffset,
          padSize: newPadSize,
          pendingPad: false,
        });
      }
    }

    const needViewsNew = this.updateViews(prevState);
    if (needViews !== needViewsNew) {
      this.setState({
        needViews: needViewsNew,
      });
    }
    if (prevLineFocus !== lineFocus) {
      const isInstant = lineFocus < 0 && currentIx < 0;
      if (isInstant) {
        this.setState({
          padSize: 0,
          offset: 0 as HOffset,
          needViews: false,
        });
      }
      this.focus(lineFocus, !isInstant);
    }
  }

  updateViews(prevState: HorizontalState | EmptyHorizontalState): boolean {
    const { offset, itemCount, needViews } = this.state;
    let needViewsNew = needViews;
    if (prevState.offset !== offset || prevState.itemCount !== itemCount) {
      Array.from(this.activeView.keys()).forEach((realIx) => {
        if (realIx < offset || realIx >= offset + itemCount) {
          const obs = this.activeView.get(realIx);
          if (obs) {
            obs.disconnect();
          }
          this.activeView.delete(realIx);
        }
      });
      Array.from(this.activeRefs.keys()).forEach((realIx) => {
        if (realIx < offset || realIx >= offset + itemCount) {
          this.activeRefs.delete(realIx);
        }
      });
      range(itemCount).forEach((ix) => {
        const realIx = (offset + ix) as LineIndex;
        if (!this.activeRefs.has(realIx)) {
          this.activeRefs.set(realIx, React.createRef());
          needViewsNew = true;
        }
      });
    }

    const createObserver = (
      index: LineIndex,
      current: HTMLDivElement,
      currentRoot: HTMLDivElement,
    ): IntersectionObserver => {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) {
              return;
            }
            const { lineKey, dispatch } = this.props;
            dispatch(setHCurrentIx({ lineKey, index }));
          });
        },
        {
          root: currentRoot,
          rootMargin: '0px',
          threshold: 1.0,
        },
      );
      observer.observe(current);
      return observer;
    };

    if (needViews) {
      range(itemCount).forEach((ix) => {
        const realIx = (offset + ix) as LineIndex;
        const curRef = this.activeRefs.get(realIx);
        if (curRef && curRef.current && this.rootBox.current) {
          if (!this.activeView.has(realIx)) {
            this.activeView.set(
              realIx,
              createObserver(realIx, curRef.current, this.rootBox.current),
            );
          }
        }
      });
    }
    if (!this.lockedView && this.lockedRef.current && this.rootBox.current) {
      this.lockedView = createObserver(
        LOCK_INDEX,
        this.lockedRef.current,
        this.rootBox.current,
      );
    }
    return needViewsNew;
  }

  focus(focusIx: LineIndex, smooth: boolean): void {
    const item = focusIx < 0 ? this.lockedRef : this.activeRefs.get(focusIx);
    if (item && item.current) {
      item.current.scrollIntoView({
        behavior: smooth ? 'smooth' : 'auto',
        block: 'nearest',
        inline: 'center',
      });
    }
  }

  getContent(lineKey: LineKey, index: LineIndex): string | JSX.Element {
    const { getItem, locks } = this.props;
    const locked = locks[constructKey(lineKey)];

    const getContent = (
      lkey: LineKey,
      adjIndex: AdjustedLineIndex,
    ): string | JSX.Element => {
      const fullLinkKey = toFullKey(lkey, adjIndex);
      const msg = getItem(fullLinkKey, this.requestRedraw);
      if (msg !== undefined) {
        return <ReactMarkdown skipHtml={true}>{msg}</ReactMarkdown>;
      }
      return `loading [${index}]`;
    };

    if (locked && index < 0) {
      return getContent(locked.lineKey, locked.index);
    }
    return getContent(lineKey, this.adjustIndex(index));
  }

  adjustIndex(index: LineIndex): AdjustedLineIndex {
    const { lineKey, locks } = this.props;
    const { offset, itemCount } = this.state;
    const locked = locks[constructKey(lineKey)];
    const lockedIx =
      locked && locked.skipItem
        ? locked.index
        : ((offset + itemCount) as AdjustedLineIndex);
    return (index + (lockedIx > index ? 0 : 1)) as AdjustedLineIndex;
  }

  requestRedraw = (): void => {
    console.groupCollapsed('H');
    console.log('request redraw H');
    console.trace();
    console.groupEnd();
    const { redraw } = this.state;
    this.setState({
      redraw: !redraw,
    });
  };

  handleLeft = (event: React.MouseEvent<HTMLButtonElement>): void => {
    const { lineKey, currentLineIxs, dispatch } = this.props;
    const currentIx = currentLineIxs[constructKey(lineKey)];
    dispatch(focusAt({ lineKey, index: (currentIx - 1) as LineIndex }));
    event.preventDefault();
  };

  handleRight = (event: React.MouseEvent<HTMLButtonElement>): void => {
    const { lineKey, currentLineIxs, dispatch } = this.props;
    const currentIx = currentLineIxs[constructKey(lineKey)];
    dispatch(focusAt({ lineKey, index: (currentIx + 1) as LineIndex }));
    event.preventDefault();
  };

  render() {
    const {
      itemWidth,
      itemHeight,
      itemRadius,
      buttonSize,
      itemPadding,
      isViewUpdate,
      vPosType,
      lineKey,
      locks,
    } = this.props;
    const { offset, itemCount, padSize } = this.state;
    const locked = locks[constructKey(lineKey)];
    const offShift = (offset < 0 ? -offset : 0) as HOffset;
    return (
      <Outer
        itemHeight={itemHeight}
        ref={this.rootBox}>
        <Overlay>
          <NavButton
            buttonSize={buttonSize}
            onClick={this.handleLeft}>
            &lt;
          </NavButton>
          <NavButton
            buttonSize={buttonSize}
            onClick={this.handleRight}>
            &gt;
          </NavButton>
        </Overlay>
        <Band
          itemWidth={itemWidth}
          isViewUpdate={isViewUpdate}
          vPosType={vPosType}
          ref={this.bandRef}>
          {locked ? (
            <Item itemWidth={itemWidth}>
              <ItemContent
                itemPadding={itemPadding}
                itemWidth={itemWidth}
                itemRadius={itemRadius}
                ref={this.lockedRef}>
                {this.getContent(lineKey, LOCK_INDEX)}
              </ItemContent>
            </Item>
          ) : null}
          <Pad padSize={padSize} />
          {range(itemCount - offShift).map((ix) => {
            const realIx = (offset + ix + offShift) as LineIndex;
            return (
              <Item
                key={realIx}
                itemWidth={itemWidth}>
                <ItemContent
                  itemPadding={itemPadding}
                  itemWidth={itemWidth}
                  itemRadius={itemRadius}
                  ref={this.activeRefs.get(realIx)}>
                  {this.getContent(lineKey, realIx)}
                </ItemContent>
              </Item>
            );
          })}
        </Band>
      </Outer>
    );
  }
} // Horizontal

const connector = connect((state: RootState) => ({
  currentLineIxs: state.lineState.currentLineIxs,
  currentLineFocus: state.lineState.currentLineFocus,
  locks: state.lineState.locks,
}));

export default connector(Horizontal);

type ConnectHorizontal = ConnectedProps<typeof connector>;
