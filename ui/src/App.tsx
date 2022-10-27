import React, { PureComponent } from 'react';
// import RequireLogin from './RequireLogin.js';
import styled from 'styled-components';
import ContentLoader, { Link } from './misc/ContentLoader';
import Horizontal from './main/Horizontal';
import Vertical from './main/Vertical';
import { ContentCB, ReadyCB } from './misc/GenericLoader';

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

export default class App extends PureComponent<{}, {}> {
  loader: ContentLoader;

  constructor(props: {}) {
    super(props);
    this.state = {};
    this.loader = new ContentLoader();
  }

  getItem = (
    isParent: boolean,
    name: string,
    index: number,
    contentCb: ContentCB<Link, string | JSX.Element>,
    readyCb: ReadyCB,
  ): string | JSX.Element => {
    return this.loader.getItem(isParent, name, index, contentCb, readyCb);
  };

  getVItem = (
    isParent: boolean,
    lineName: string,
    height: number,
  ): JSX.Element => {
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

  getChildLine = (
    lineName: string,
    index: number,
    callback: (child: string) => void,
  ): void => {
    this.loader.getChildLine(lineName, index, callback);
  };

  getParentLine = (
    lineName: string,
    index: number,
    callback: (parent: string) => void,
  ): void => {
    this.loader.getParentLine(lineName, index, callback);
  };

  getLink = (
    parentLineName: string,
    childLineName: string,
    parentIndex: number,
    childIndex: number,
    readyCb: ReadyCB,
  ): Link | undefined => {
    return this.loader.getLink(
      parentLineName,
      childLineName,
      parentIndex,
      childIndex,
      readyCb,
    );
  };

  renderLink = (
    link: Link,
    buttonSize: number,
    radius: number,
  ): JSX.Element => {
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
