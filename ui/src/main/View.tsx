import { connect, ConnectedProps } from 'react-redux';
import React, { PureComponent, ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import styled from 'styled-components';
import { RootState } from '../store';
import { Cell, progressView } from '../misc/GraphView';
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

const VBand = styled.div`
  margin: 0;
  padding: 0;
  display: block;
  width: 100%;
  height: 100%;
  background-color: green;
  overflow-x: hidden;
  overflow-y: scroll;

  &::-webkit-scrollbar {
    display: none;
  }

  -ms-overflow-style: none;
  scrollbar-width: none;
  scroll-snap-type: y mandatory;
`;

const HBand = styled.div`
  margin: 0;
  padding: 0;
  display: inline-block;
  width: 100%;
  height: var(--main-size);
  white-space: nowrap;
  overflow-x: scroll;
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

  background-color: green;
  &:hover {
    background-color: blue;
  }
`;

enum ResetView {
  ResetBottom,
  ResetTop,
  Done,
}

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
  top: React.RefObject<HTMLDivElement>;
  topLeft: React.RefObject<HTMLDivElement>;
  centerTop: React.RefObject<HTMLDivElement>;
  topRight: React.RefObject<HTMLDivElement>;
  bottomLeft: React.RefObject<HTMLDivElement>;
  centerBottom: React.RefObject<HTMLDivElement>;
  bottomRight: React.RefObject<HTMLDivElement>;
  bottom: React.RefObject<HTMLDivElement>;

  constructor(props: Readonly<ViewProps>) {
    super(props);
    this.state = {
      resetView: ResetView.ResetBottom,
      redraw: false,
    };
    this.top = React.createRef();
    this.topLeft = React.createRef();
    this.centerTop = React.createRef();
    this.topRight = React.createRef();
    this.bottomLeft = React.createRef();
    this.centerBottom = React.createRef();
    this.bottomRight = React.createRef();
    this.bottom = React.createRef();
  }

  componentDidMount(): void {
    this.componentDidUpdate({ view: undefined }, undefined);
  }

  componentDidUpdate(
    prevProps: Readonly<ViewProps> | EmptyViewProps,
    _prevState: Readonly<ViewState> | undefined,
  ): void {
    const { graph, view, dispatch } = this.props;
    const { resetView } = this.state;
    if (view !== prevProps.view) {
      progressView(graph, view, (newView) => {
        dispatch(setView({ view: newView }));
      });
    }
    if (resetView !== ResetView.Done) {
      if (
        resetView === ResetView.ResetBottom &&
        this.centerBottom.current !== null
      ) {
        this.centerBottom.current.scrollIntoView({
          behavior: 'auto',
          block: 'end',
          inline: 'nearest',
        });
        this.setState({ resetView: ResetView.ResetTop });
      } else if (this.centerTop.current !== null) {
        this.centerTop.current.scrollIntoView({
          behavior: 'auto',
          block: 'start',
          inline: 'nearest',
        });
        this.setState({ resetView: ResetView.Done });
      }
    }
  }

  render(): ReactNode {
    const { view } = this.props;

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
      <Outer>
        <VBand>
          <HBand>
            <Item ref={this.top}>
              <ItemContent>{getContent(view.top)}</ItemContent>
            </Item>
          </HBand>
          <HBand>
            <Item ref={this.topLeft}>
              <ItemContent>{getContent(view.topLeft)}</ItemContent>
            </Item>
            <Item ref={this.centerTop}>
              <ItemContent>{getContent(view.centerTop)}</ItemContent>
            </Item>
            <Item ref={this.topRight}>
              <ItemContent>{getContent(view.topRight)}</ItemContent>
            </Item>
          </HBand>
          <HBand>
            <Item ref={this.bottomLeft}>
              <ItemContent>{getContent(view.bottomLeft)}</ItemContent>
            </Item>
            <Item ref={this.centerBottom}>
              <ItemContent>{getContent(view.centerBottom)}</ItemContent>
            </Item>
            <Item ref={this.bottomRight}>
              <ItemContent>{getContent(view.bottomRight)}</ItemContent>
            </Item>
          </HBand>
          <HBand>
            <Item ref={this.bottom}>
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
