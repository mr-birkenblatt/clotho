import { connect, ConnectedProps } from 'react-redux';
import React, { PureComponent, ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import styled from 'styled-components';
import { RootState } from '../store';
import {
  Cell,
  Direction,
  horizontal,
  NavigationCB,
  progressView,
  scrollBottomHorizontal,
  scrollTopHorizontal,
  scrollVertical,
  vertical,
} from '../misc/GraphView';
import CommentGraph from '../misc/CommentGraph';
import { errHnd, safeStringify, toReadableNumber } from '../misc/util';
import { setView } from './ViewStateSlice';
import { NormalComponents } from 'react-markdown/lib/complex-types';
import { SpecialComponents } from 'react-markdown/lib/ast-to-react';
import { RichVote, VoteType, VOTE_TYPES } from '../misc/keys';

const Outer = styled.div`
  position: relative;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
`;

const NavButton = styled.button`
  appearance: none;
  display: inline-block;
  text-align: center;
  vertical-align: middle;
  pointer-events: auto;
  cursor: pointer;
  border: 0;
  opacity: 0.8;
  color: #5b5f67;
  border-radius: calc(var(--main-size) * 0.1);
  width: calc(var(--main-size) * 0.15);
  height: calc(var(--main-size) * 0.15);
  background-color: #393d45;

  &:hover {
    background-color: #4a4e56;
  }
  &:active {
    background-color: #5b5f67;
  }
`;

const VNavButton = styled(NavButton)<OverlayProps>`
  position: fixed;
  ${(props) => (props.isTop ? 'top' : 'bottom')}: 0;
  right: 0;
`;

const HOverlay = styled.div<OverlayProps>`
  height: calc(var(--main-size) * 0.15);
  position: absolute;
  left: 0;
  ${(props) => (props.isTop ? 'top' : 'bottom')}: 0;
  display: flex;
  justify-content: space-between;
  flex-direction: row;
  flex-wrap: nowrap;
  width: var(--main-size);
  pointer-events: none;
`;

const HNavButton = styled(NavButton)``;

const Temp = styled.div<NoScrollProps>`
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

const VBand = styled.div<NoScrollProps>`
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

const HBand = styled.div<NoScrollProps>`
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
  border-radius: calc(var(--main-size) * 0.1);
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

const ButtonDiv = styled.div<ButtonDivProps>`
  display: inline-block;
  text-align: center;
  vertical-align: middle;
  pointer-events: auto;
  cursor: pointer;
  border: 0;
  ${(props) => (props.isChecked ? 'background-color: #6c7078;' : '')}

  &:hover {
    background-color: #4a4e56;
  }
  &:active {
    background-color: #5b5f67;
  }
`;

const ItemMidName = styled(ButtonDiv)`
  font-size: 0.6em;
  align-items: center;
  justify-content: center;
  border-radius: calc(var(--main-size) * 0.025);
  margin-top: calc(var(--main-size) * -0.0125);
  padding: calc(var(--main-size) * 0.0125);
`;

const ItemMidContent = styled(ButtonDiv)`
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: row;
  height: calc(var(--main-size) * 0.05);
  border-radius: calc(var(--main-size) * 0.025);
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

type OverlayProps = {
  isTop: boolean;
};

type NoScrollProps = {
  noScroll: boolean;
};

type ButtonDivProps = {
  isChecked: boolean;
};

enum ResetView {
  StopScroll = 'StopScroll',
  ResetBottom = 'ResetBottom',
  ResetTop = 'ResetTop',
  Done = 'Done',
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

const navigationCBs: { [Property in ObsKey]: [NavigationCB, Direction] } = {
  top: [vertical(scrollVertical), Direction.UpRight],
  topLeft: [horizontal(scrollTopHorizontal), Direction.DownLeft],
  topRight: [horizontal(scrollTopHorizontal), Direction.UpRight],
  bottomLeft: [horizontal(scrollBottomHorizontal), Direction.DownLeft],
  bottomRight: [horizontal(scrollBottomHorizontal), Direction.UpRight],
  bottom: [vertical(scrollVertical), Direction.DownLeft],
};

const scrollBlocks: { [Property in ObsKey]: ScrollLogicalPosition } = {
  top: 'start',
  topLeft: 'start',
  topRight: 'start',
  bottomLeft: 'end',
  bottomRight: 'end',
  bottom: 'end',
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
  pending: [NavigationCB, Direction] | undefined;
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
    const { graph, view, changes, dispatch } = this.props;
    const { resetView, redraw, pending } = this.state;
    if (view !== prevProps.view || pending !== undefined) {
      progressView(graph, view).then(
        ({ view: newView, change }) => {
          if (change) {
            dispatch(setView({ view: newView, changes, progress: true }));
            return;
          }
          if (pending !== undefined) {
            const [navigator, direction] = pending;
            const nextView = navigator(newView, direction);
            if (nextView !== undefined) {
              dispatch(setView({ view: nextView, changes, progress: false }));
            }
            this.setState({ pending: undefined });
          }
        },
        (e) => {
          errHnd(e);
        },
      );
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

  private isNoScroll(): boolean {
    const { resetView } = this.state;
    return resetView === ResetView.StopScroll;
  }

  private navigate(
    key: ObsKey,
    navigator: NavigationCB,
    direction: Direction,
  ): void {
    const { dispatch, view, changes } = this.props;
    const newView = navigator(view, direction);
    if (newView !== undefined) {
      dispatch(setView({ view: newView, changes, progress: false }));
    } else {
      this.setState({ pending: [navigator, direction] });
    }
    this.setState({
      resetView: ResetView.StopScroll,
      tempContent: this.getTempConfig(key),
    });
  }

  private handleButtons(
    event: React.MouseEvent<HTMLElement>,
    key: ObsKey,
  ): void {
    const current = this.curRefs[key].current;
    if (current !== null && !this.isNoScroll()) {
      current.scrollIntoView({
        behavior: 'smooth',
        block: scrollBlocks[key],
        inline: 'nearest',
      });
    }
    event.preventDefault();
  }

  handleUp = (event: React.MouseEvent<HTMLElement>): void => {
    this.handleButtons(event, 'top');
  };

  handleDown = (event: React.MouseEvent<HTMLElement>): void => {
    this.handleButtons(event, 'bottom');
  };

  handleTopRight = (event: React.MouseEvent<HTMLElement>): void => {
    this.handleButtons(event, 'topRight');
  };

  handleTopLeft = (event: React.MouseEvent<HTMLElement>): void => {
    this.handleButtons(event, 'topLeft');
  };

  handleBottomRight = (event: React.MouseEvent<HTMLElement>): void => {
    this.handleButtons(event, 'bottomRight');
  };

  handleBottomLeft = (event: React.MouseEvent<HTMLElement>): void => {
    this.handleButtons(event, 'bottomLeft');
  };

  render(): ReactNode {
    const { view } = this.props;
    const { tempContent } = this.state;
    const noScroll = this.isNoScroll();

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
            {VOTE_TYPES.map((voteType: VoteType): RichVote => {
              const res = link.votes[voteType];
              if (res === undefined) {
                return { voteType, count: 0, userVoted: false };
              }
              const { count, userVoted } = res;
              return { voteType, count, userVoted };
            }).map(({ voteType, count, userVoted }) => (
              <ItemMidContent
                key={voteType}
                isChecked={userVoted}>
                {toReadableNumber(count)}{' '}
                {VOTE_SYMBOL.get(voteType) ?? `[${voteType}]`}
              </ItemMidContent>
            ))}
          </ItemMidVotes>
          <ItemMidName isChecked={false}>{link.user}</ItemMidName>
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
        <HOverlay isTop={true}>
          <HNavButton onClick={this.handleTopLeft}>◀</HNavButton>
          <HNavButton onClick={this.handleTopRight}>▶</HNavButton>
        </HOverlay>
        <HOverlay isTop={false}>
          <HNavButton onClick={this.handleBottomLeft}>◀</HNavButton>
          <HNavButton onClick={this.handleBottomRight}>▶</HNavButton>
        </HOverlay>
        <VNavButton
          isTop={true}
          onClick={this.handleUp}>
          ▲
        </VNavButton>
        <VNavButton
          isTop={false}
          onClick={this.handleDown}>
          ▼
        </VNavButton>
      </Outer>
    );
  }
} // View

const connector = connect((state: RootState) => ({
  view: state.viewState.currentView,
  changes: state.viewState.currentChanges,
}));

export default connector(View);

type ConnectView = ConnectedProps<typeof connector>;
