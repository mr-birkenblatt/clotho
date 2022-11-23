import { connect, ConnectedProps } from 'react-redux';
import React, { PureComponent, ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import styled from 'styled-components';
import { RootState } from '../store';
import {
  Cell,
  GraphView,
  progressView,
  scrollBottomHorizontal,
  scrollTopHorizontal,
  scrollVertical,
} from '../misc/GraphView';
import CommentGraph from '../misc/CommentGraph';
import { safeStringify, toReadableNumber } from '../misc/util';
import { setView } from './ViewStateSlice';
import { NormalComponents } from 'react-markdown/lib/complex-types';
import { SpecialComponents } from 'react-markdown/lib/ast-to-react';

const Outer = styled.div`
  position: relative;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
`;

const Temp = styled.div<NoScrollProp>`
  display: ${(props) => (props.noScroll ? 'block' : 'none')};
  position: relative;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  overflow: hidden;
`;

const VBand = styled.div<NoScrollProp>`
  margin: 0;
  padding: 0;
  display: block;
  width: 100%;
  height: 100%;
  vertical-align: top;
  background-color: #282c34;
  overflow-x: hidden;
  overflow-y: ${(props) => (props.noScroll ? 'hidden' : 'scroll')};

  &::-webkit-scrollbar {
    display: none;
  }

  -ms-overflow-style: none;
  scrollbar-width: none;
  scroll-snap-type: y mandatory;
`;

const HBand = styled.div<NoScrollProp>`
  margin: 0;
  padding: 0;
  display: inline-block;
  width: 100%;
  height: var(--main-size);
  white-space: nowrap;
  vertical-align: top;
  overflow-x: ${(props) => (props.noScroll ? 'hidden' : 'scroll')};
  overflow-y: hidden;

  &::-webkit-scrollbar {
    display: none;
  }

  -ms-overflow-style: none;
  scrollbar-width: none;
  scroll-snap-type: x mandatory;
  scroll-snap-align: start;
`;

const Item = styled.div`
  display: inline-block;
  white-space: nowrap;
  vertical-align: top;
  margin: 0;
  padding: 0;
  width: var(--main-size);
  height: var(--main-size);
  scroll-snap-align: center;
`;

const ItemContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  width: calc(var(--main-size) * 0.9);
  height: calc(var(--main-size) * 0.9);
  margin: 0;
  border-radius: calc(var(--main-size) * 0.05);
  padding: calc(var(--main-size) * 0.05);
  white-space: normal;
  overflow: hidden;

  background-color: #393d45;
`;

const ItemMid = styled.div`
  display: flex;
  width: 100%;
  height: 0;
  position: relative;
  top: calc(var(--main-size) * 0.06);
  left: 0;
  align-items: center;
  justify-content: center;
  text-align: center;
  opacity: 0.8;
  flex-direction: column;
`;

const ItemMidVotes = styled.div`
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
`;

const ItemMidName = styled.div`
  font-size: 0.6em;
  align-items: center;
  justify-content: center;
`;

const ItemMidContent = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: row;
  height: calc(var(--main-size) * 0.05);
  border-radius: calc(var(--main-size) * 0.0125);
  padding: calc(var(--main-size) * 0.0125);
`;

const Link = styled.a`
  color: silver;

  &:visited {
    color: silver;
  }
  &:active {
    color: silver;
  }
  &:hover {
    color: #ddd;
  }
`;

const VOTE_SYMBOL = new Map();
VOTE_SYMBOL.set('up', '👍');
VOTE_SYMBOL.set('down', '👎');
VOTE_SYMBOL.set('honor', '⭐');

const MD_COMPONENTS: Partial<
  Omit<NormalComponents, keyof SpecialComponents> & SpecialComponents
> = {
  a: ({ node: _, ...props }) => <Link {...props} />,
};

type NoScrollProp = {
  noScroll: boolean;
};

enum ResetView {
  StopScroll,
  ResetBottom,
  ResetTop,
  Done,
}

type Refs = {
  top: React.RefObject<HTMLDivElement>;
  topLeft: React.RefObject<HTMLDivElement>;
  centerTop: React.RefObject<HTMLDivElement>;
  topRight: React.RefObject<HTMLDivElement>;
  bottomLeft: React.RefObject<HTMLDivElement>;
  centerBottom: React.RefObject<HTMLDivElement>;
  bottomRight: React.RefObject<HTMLDivElement>;
  bottom: React.RefObject<HTMLDivElement>;
};

type Obs = {
  top: IntersectionObserver | undefined;
  topLeft: IntersectionObserver | undefined;
  topRight: IntersectionObserver | undefined;
  bottomLeft: IntersectionObserver | undefined;
  bottomRight: IntersectionObserver | undefined;
  bottom: IntersectionObserver | undefined;
};

type ObsKey = keyof Obs;

type NavigationCB = (
  view: Readonly<GraphView>,
  upRight: boolean,
) => Readonly<GraphView> | undefined;

const navigationCBs: { [Property in ObsKey]: [NavigationCB, boolean] } = {
  top: [scrollVertical, true],
  topLeft: [scrollTopHorizontal, false],
  topRight: [scrollTopHorizontal, true],
  bottomLeft: [scrollBottomHorizontal, false],
  bottomRight: [scrollBottomHorizontal, true],
  bottom: [scrollVertical, false],
};

interface ViewProps extends ConnectView {
  graph: CommentGraph;
}

type EmptyViewProps = {
  view: undefined;
};

type ViewState = {
  resetView: ResetView;
  redraw: boolean;
  tempContent: [Readonly<Cell> | undefined, Readonly<Cell> | undefined];
  pending: [NavigationCB, boolean] | undefined;
};

class View extends PureComponent<ViewProps, ViewState> {
  rootRef: React.RefObject<HTMLDivElement>;
  curRefs: Refs;
  curObs: Obs;

  constructor(props: Readonly<ViewProps>) {
    super(props);
    this.state = {
      resetView: ResetView.ResetBottom,
      redraw: false,
      tempContent: [undefined, undefined],
      pending: undefined,
    };
    this.rootRef = React.createRef();
    this.curRefs = {
      top: React.createRef(),
      topLeft: React.createRef(),
      centerTop: React.createRef(),
      topRight: React.createRef(),
      bottomLeft: React.createRef(),
      centerBottom: React.createRef(),
      bottomRight: React.createRef(),
      bottom: React.createRef(),
    };
    this.curObs = {
      top: undefined,
      topLeft: undefined,
      topRight: undefined,
      bottomLeft: undefined,
      bottomRight: undefined,
      bottom: undefined,
    };
  }

  componentDidMount(): void {
    this.componentDidUpdate({ view: undefined }, undefined);
  }

  componentDidUpdate(
    prevProps: Readonly<ViewProps> | EmptyViewProps,
    _prevState: Readonly<ViewState> | undefined,
  ): void {
    const { graph, view, dispatch } = this.props;
    const { resetView, redraw, pending } = this.state;
    if (view !== prevProps.view || pending !== undefined) {
      const finalView = progressView(graph, view, (newView) => {
        dispatch(setView({ view: newView }));
      });
      if (finalView !== undefined && pending !== undefined) {
        const [navigator, upRight] = pending;
        const nextView = navigator(view, upRight);
        if (nextView !== undefined) {
          dispatch(setView({ view: nextView }));
        }
        this.setState({ pending: undefined });
      }
    }
    if (resetView !== ResetView.Done) {
      if (resetView === ResetView.StopScroll) {
        if (pending === undefined) {
          setTimeout(() => {
            this.setState({ resetView: ResetView.ResetBottom });
          }, 100);
        }
      } else if (
        resetView === ResetView.ResetBottom &&
        this.curRefs.centerBottom.current !== null
      ) {
        this.curRefs.centerBottom.current.scrollIntoView({
          behavior: 'auto',
          block: 'end',
          inline: 'nearest',
        });
        this.setState({ resetView: ResetView.ResetTop });
      } else if (this.curRefs.centerTop.current !== null) {
        this.curRefs.centerTop.current.scrollIntoView({
          behavior: 'auto',
          block: 'start',
          inline: 'nearest',
        });
        this.setState({
          resetView: ResetView.Done,
          tempContent: [undefined, undefined],
        });
      }
    }

    const ensureObserver = (key: ObsKey): boolean => {
      if (this.curObs[key] !== undefined) {
        return true;
      }
      const current = this.curRefs[key].current;
      if (current === null) {
        return false;
      }
      const root = this.rootRef.current;
      if (root === null) {
        return false;
      }
      const observer = new IntersectionObserver(
        (entries) => {
          const { pending, resetView } = this.state;
          let done = pending !== undefined || resetView !== ResetView.Done;
          entries.forEach((entry) => {
            if (!entry.isIntersecting) {
              return;
            }
            if (done) {
              return;
            }
            this.navigate(key, ...navigationCBs[key]);
            done = true;
          });
        },
        {
          root,
          rootMargin: '0px',
          threshold: 0.9,
        },
      );
      observer.observe(current);
      this.curObs[key] = observer;
      return true;
    };
    const obsReady = Object.keys(this.curObs).map((key) => {
      return ensureObserver(key as ObsKey);
    });
    if (!obsReady.every((res) => res)) {
      console.warn('delayed intersection observers!');
      this.setState({ redraw: !redraw });
    }
  }

  private getTempConfig(
    key: ObsKey,
  ): [Readonly<Cell> | undefined, Readonly<Cell> | undefined] {
    const { view } = this.props;
    if (key === 'top') {
      return [view.top, view.centerTop];
    }
    if (key == 'topLeft') {
      return [view.topLeft, view.centerBottom];
    }
    if (key == 'topRight') {
      return [view.topRight, view.centerBottom];
    }
    if (key == 'bottomLeft') {
      return [view.centerTop, view.bottomLeft];
    }
    if (key == 'bottomRight') {
      return [view.centerTop, view.bottomRight];
    }
    if (key == 'bottom') {
      return [view.centerBottom, view.bottom];
    }
    return [view.centerTop, view.centerBottom];
  }

  private navigate(
    key: ObsKey,
    navigator: NavigationCB,
    upRight: boolean,
  ): void {
    const { dispatch, view } = this.props;
    const newView = navigator(view, upRight);
    if (newView !== undefined) {
      dispatch(setView({ view: newView }));
    } else {
      this.setState({ pending: [navigator, upRight] });
    }
    this.setState({
      resetView: ResetView.StopScroll,
      tempContent: this.getTempConfig(key),
      // newView !== undefined
      //   ? [newView.centerTop, newView.centerBottom]
      //   : this.getTempConfig(key),
    });
  }

  render(): ReactNode {
    const { view } = this.props;
    const { resetView, tempContent } = this.state;
    const noScroll = resetView === ResetView.StopScroll;

    const getTopLink = (cell: Cell | undefined): JSX.Element | null => {
      if (cell === undefined) {
        return null;
      }
      if (cell.topLink === undefined) {
        return null;
      }
      const link = cell.topLink;
      if (link.invalid) {
        return null;
      }
      return (
        <ItemMid>
          <ItemMidVotes>
            {Object.keys(link.votes).map((voteName) => (
              <ItemMidContent key={voteName}>
                {toReadableNumber(link.votes[voteName])}{' '}
                {VOTE_SYMBOL.get(voteName) ?? `[${voteName}]`}
              </ItemMidContent>
            ))}
          </ItemMidVotes>
          <ItemMidName>{link.user}</ItemMidName>
        </ItemMid>
      );
    };

    const getContent = (cell: Cell | undefined): JSX.Element | string => {
      if (cell === undefined) {
        return '[loading]';
      }
      if (cell.content === undefined) {
        return `[loading...${safeStringify(cell.fullKey)}]`;
      }
      return (
        <ReactMarkdown
          skipHtml={true}
          components={MD_COMPONENTS}>
          {cell.content}
        </ReactMarkdown>
      );
    };

    return (
      <Outer ref={this.rootRef}>
        <Temp noScroll={noScroll}>
          <Item>
            {getTopLink(tempContent[0])}
            <ItemContent>{getContent(tempContent[0])}</ItemContent>
          </Item>
          <Item>
            {getTopLink(tempContent[1])}
            <ItemContent>{getContent(tempContent[1])}</ItemContent>
          </Item>
        </Temp>
        <VBand noScroll={noScroll}>
          <HBand noScroll={noScroll}>
            <Item ref={this.curRefs.top}>
              <ItemContent>{getContent(view.top)}</ItemContent>
            </Item>
          </HBand>
          <HBand noScroll={noScroll}>
            <Item ref={this.curRefs.topLeft}>
              <ItemContent>{getContent(view.topLeft)}</ItemContent>
            </Item>
            <Item ref={this.curRefs.centerTop}>
              {getTopLink(view.centerTop)}
              <ItemContent>{getContent(view.centerTop)}</ItemContent>
            </Item>
            <Item ref={this.curRefs.topRight}>
              <ItemContent>{getContent(view.topRight)}</ItemContent>
            </Item>
          </HBand>
          <HBand noScroll={noScroll}>
            <Item ref={this.curRefs.bottomLeft}>
              {getTopLink(view.bottomLeft)}
              <ItemContent>{getContent(view.bottomLeft)}</ItemContent>
            </Item>
            <Item ref={this.curRefs.centerBottom}>
              {getTopLink(view.centerBottom)}
              <ItemContent>{getContent(view.centerBottom)}</ItemContent>
            </Item>
            <Item ref={this.curRefs.bottomRight}>
              {getTopLink(view.bottomRight)}
              <ItemContent>{getContent(view.bottomRight)}</ItemContent>
            </Item>
          </HBand>
          <HBand noScroll={noScroll}>
            <Item ref={this.curRefs.bottom}>
              {getTopLink(view.bottom)}
              <ItemContent>{getContent(view.bottom)}</ItemContent>
            </Item>
          </HBand>
        </VBand>
      </Outer>
    );
  }
} // View

const connector = connect((state: RootState) => ({
  view: state.viewState.currentView,
}));

export default connector(View);

type ConnectView = ConnectedProps<typeof connector>;
