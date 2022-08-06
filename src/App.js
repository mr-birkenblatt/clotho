import React, { PureComponent } from 'react';
// import RequireLogin from './RequireLogin.js';
import styled from 'styled-components';
import ContentLoader from './contentLoader.js';
import Horizontal from './Horizontal.js';
import Vertical from './Vertical.js';

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
`


export default class App extends PureComponent {
  constructor(props) {
    super(props);
    this.state = {};
    this.parentLines = new ContentLoader(5, (name, offset, limit, cb) => {
      console.log(`loading parent ${name} ${offset} ${limit}`);
      setTimeout(() => {
        const res = {};
        [...Array(limit).keys()].forEach(ix => {
          res[ix + offset] = `**name**: ${name} _ix_: ${ix + offset}`;
        });
        cb(res);
      }, 500);
    });
    this.childLines = new ContentLoader(5, (name, offset, limit, cb) => {
      console.log(`loading child ${name} ${offset} ${limit}`);
      setTimeout(() => {
        const res = {};
        [...Array(limit).keys()].forEach(ix => {
          res[ix + offset] = `**name**: ${name} _ix_: ${ix + offset}`;
        });
        cb(res);
      }, 1000);
    });
  }

  getItem = (isParent, name, index, contentCb, readyCb) => {
    return (isParent ? this.parentLines : this.childLines).get(
      name, index, contentCb, readyCb);
  }

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
        getItem={this.getItem} />
    );
  }

  getChildLine = (lineName) => {
    return `L${+lineName.slice(1) + 1}`;
  }

  render() {
    return (
      <Main>
        {/* <MainHeader>
          <RequireLogin />
        </MainHeader> */}
        <MainColumn>
          <Vertical getItem={this.getVItem} getChildLine={this.getChildLine} />
        </MainColumn>
      </Main>
    );
  }
} // App
