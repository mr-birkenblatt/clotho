import React, { PureComponent } from 'react';
// import RequireLogin from './RequireLogin.js';
import styled from 'styled-components';
import ContentLoader from './misc/ContentLoader';
import Horizontal from './main/Horizontal';
import Vertical from './main/Vertical';

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

export default class App extends PureComponent {
  loader: ContentLoader;

  constructor(props) {
    super(props);
    this.state = {};
    this.loader = new ContentLoader();
  }

  getItem = (isParent, name, index, contentCb, readyCb) => {
    return this.loader.getItem(isParent, name, index, contentCb, readyCb);
  };

  getVItem = (isParent, lineName, height) => {
    return (
      <Horizontal
        itemWidth={450}
        itemHeight={height}
        itemRadius={10}
        itemPadding={50}
        buttonSize={50}
        isParent={isParent}
        lineName={lineName}
        getItem={this.getItem}
      />
    );
  };

  getChildLine = (lineName, cb) => {
    this.loader.getChild(lineName, cb);
  };

  getParentLine = (lineName, cb) => {
    this.loader.getParent(lineName, cb);
  };

  getLinkItems = (parentLineName, childLineName, parentIndex, childIndex) => {
    return this.loader.getLinkInfo(
      parentLineName,
      childLineName,
      parentIndex,
      childIndex,
    );
  };

  renderLinkItem = (link) => {
    return (
      <span>
        {link.key}: {link.count}
      </span>
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
            getLinkItems={this.getLinkItems}
            renderLinkItem={this.renderLinkItem}
            height={450}
            radius={10}
            buttonSize={50}
          />
        </MainColumn>
      </Main>
    );
  }
} // App
