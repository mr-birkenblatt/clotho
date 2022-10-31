import React, { PureComponent } from 'react';
// import RequireLogin from './RequireLogin.js';
import styled from 'styled-components';
import Horizontal, { ItemCB } from './main/Horizontal';
import Vertical, {
  ChildLineCB,
  LinkCB,
  ParentLineCB,
  RenderLinkCB,
  VItemCB,
} from './main/Vertical';
import CommentGraph, { toFullKey } from './misc/CommentGraph';

const Main = styled.div`
  text-align: center;
  margin: 0 auto;
  height: 100vh;
  width: 100vw;
  display: flex;
  flex-direction: column;
  background-color: #282c34;
  color: white;
`;

// const MainHeader = styled.header`
//   background-color: #282c34;
//   min-height: 100vh;
//   display: flex;
//   flex-direction: column;
//   align-items: center;
//   justify-content: center;
//   font-size: calc(10px + 2vmin);
//   color: white;
// `;

const MainColumn = styled.div`
  max-width: 500px;
  margin: 0 auto;
  width: 100%;
  height: 100vh;
  background-color: pink;
  border: green 1px solid;
  display: flex;
  flex-direction: column;
  flex-grow: 0;
  overflow: hidden;
`;

const ItemMidContent = styled.div<ItemMidContentProps>`
  display: flex;
  height: ${(props) => props.buttonSize}px;
  background-color: green;
  padding: ${(props) => props.radius}px;
  border-radius: ${(props) => props.radius}px;
  align-items: center;
  justify-content: center;
  flex-direction: column;
`;

type ItemMidContentProps = {
  buttonSize: number;
  radius: number;
};

type AppProps = Record<string, never>;
type AppState = Record<string, never>;

export default class App extends PureComponent<AppProps, AppState> {
  graph: CommentGraph;

  constructor(props: AppProps) {
    super(props);
    this.state = {};
    this.graph = new CommentGraph();
  }

  getItem: ItemCB = (fullLinkKey, readyCb) => {
    return this.graph.getMessage(fullLinkKey, () => readyCb());
  };

  getVItem: VItemCB = (lineKey, height) => {
    if (lineKey === undefined) {
      return null;
    }
    return (
      <Horizontal
        itemWidth={450}
        itemHeight={height}
        itemRadius={10}
        itemPadding={50}
        buttonSize={50}
        lineKey={lineKey}
        getItem={this.getItem}
      />
    );
  };

  getChildLine: ChildLineCB = (lineKey, index, childIndex, callback) => {
    this.graph.getChild(toFullKey(lineKey, index), childIndex, callback);
  };

  getParentLine: ParentLineCB = (lineKey, index, parentIndex, callback) => {
    this.graph.getParent(toFullKey(lineKey, index), parentIndex, callback);
  };

  getLink: LinkCB = (fullLinkKey, parentIndex, readyCb) => {
    return this.graph.getTopLink(fullLinkKey, parentIndex, () => readyCb());
  };

  renderLink: RenderLinkCB = (link, buttonSize, radius) => {
    if (!link.valid) {
      return null;
    }
    return (
      <div>
        <div>
          {link.user}: {link.first}
        </div>
        <div>
          {Object.keys(link.votes).map((voteName) => (
            <ItemMidContent
              buttonSize={buttonSize}
              radius={radius}
              key={voteName}>
              <span>
                {voteName}: {link.votes[voteName]}
              </span>
            </ItemMidContent>
          ))}
        </div>
      </div>
    );
  };

  render() {
    return (
      <Main>
        {/* <MainHeader>
          <RequireLogin />
        </MainHeader> */}
        <MainColumn>
          <Vertical
            getItem={this.getVItem}
            getChildLine={this.getChildLine}
            getParentLine={this.getParentLine}
            getLink={this.getLink}
            renderLink={this.renderLink}
            height={450}
            radius={10}
            buttonSize={50}
          />
        </MainColumn>
      </Main>
    );
  }
} // App
