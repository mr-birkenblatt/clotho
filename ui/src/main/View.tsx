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
import { safeStringify } from '../misc/util';
import { setView } from './ViewStateSlice';

const Outer = styled.div`
  position: relative;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
`;

const VBand = styled.div<NoScrollProp>`
  margin: 0;
  padding: 0;
  display: block;
  width: 100%;
  height: 100%;
  background-color: green;
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
  opacity: ${(props) => (props.noScroll ? 0.5 : 1.0)};
  filter: blur(${(props) => (props.noScroll ? '1px' : '0')});
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

  background-color: lime;
`;

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
    const { resetView, redraw } = this.state;
    if (view !== prevProps.view) {
      progressView(graph, view, (newView) => {
        dispatch(setView({ view: newView }));
      });
    }
    if (resetView !== ResetView.Done) {
      if (resetView === ResetView.StopScroll) {
        setTimeout(() => {
          this.setState({ resetView: ResetView.ResetBottom });
        }, 100);
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
        this.setState({ resetView: ResetView.Done });
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
          entries.forEach((entry) => {
            if (!entry.isIntersecting) {
              return;
            }
            console.log('navigate', key);
            this.navigate(...navigationCBs[key]);
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

  navigate(navigator: NavigationCB, upRight: boolean): void {
    const { dispatch, view } = this.props;
    const newView = navigator(view, upRight);
    if (newView !== undefined) {
      console.log('new view');
      dispatch(setView({ view: newView }));
    }
    this.setState({ resetView: ResetView.StopScroll });
  }

  render(): ReactNode {
    const { view } = this.props;
    const { resetView } = this.state;
    const noScroll = resetView === ResetView.StopScroll;

    const getContent = (cell: Cell | undefined) => {
      if (cell === undefined) {
        return '[loading]';
      }
      if (cell.content === undefined) {
        return `[loading...${safeStringify(cell.fullKey)}]`;
      }
      return <ReactMarkdown skipHtml={true}>{cell.content}</ReactMarkdown>;
    };

    return (
      <Outer ref={this.rootRef}>
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
              <ItemContent>{getContent(view.centerTop)}</ItemContent>
            </Item>
            <Item ref={this.curRefs.topRight}>
              <ItemContent>{getContent(view.topRight)}</ItemContent>
            </Item>
          </HBand>
          <HBand noScroll={noScroll}>
            <Item ref={this.curRefs.bottomLeft}>
              <ItemContent>{getContent(view.bottomLeft)}</ItemContent>
            </Item>
            <Item ref={this.curRefs.centerBottom}>
              <ItemContent>{getContent(view.centerBottom)}</ItemContent>
            </Item>
            <Item ref={this.curRefs.bottomRight}>
              <ItemContent>{getContent(view.bottomRight)}</ItemContent>
            </Item>
          </HBand>
          <HBand noScroll={noScroll}>
            <Item ref={this.curRefs.bottom}>
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
