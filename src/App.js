import React, { PureComponent } from 'react';
// import RequireLogin from './RequireLogin.js';
import styled from 'styled-components';
import ContentLoader from './contentLoader.js';
import Horizontal from './Horizontal.js';

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
  background-color: pink;
  border: green 1px solid;
  display: flex;
  flex-direction: column;
  flex-grow: 1;
`


export default class App extends PureComponent {
  constructor(props) {
    super(props);
    this.state = {
      parentName: "abc",
      childName: "def",
    };
    this.parentLines = new ContentLoader(5, (name, offset, limit, cb) => {
      console.log(`loading ${name} ${offset} ${limit}`);
      setTimeout(() => {
        const res = {};
        [...Array(limit).keys()].forEach(ix => {
          res[ix + offset] = `name: ${name} ix: ${ix + offset}`;
        });
        cb(res);
      }, 500);
    });
    this.childLines = new ContentLoader(5, (name, offset, limit, cb) => {
      console.log(`loading ${name} ${offset} ${limit}`);
      setTimeout(() => {
        const res = {};
        [...Array(limit).keys()].forEach(ix => {
          res[ix + offset] = `name: ${name} ix: ${ix + offset}`;
        });
        cb(res);
      }, 500);
    });
  }

  getItem = (isParent, name, index, contentCb, readyCb) => {
    return (isParent ? this.parentLines : this.childLines).get(
      name, index, contentCb, readyCb);
  }

  render() {
    return (
      <Main>
        {/* <MainHeader>
          <RequireLogin />
        </MainHeader> */}
        <MainColumn>
          <Horizontal
            itemWidth={450}
            itemHeight={450}
            itemRadius={10}
            itemPadding={50}
            buttonSize={50}
            isParent={true}
            lineName={this.state.parentName}
            getItem={this.getItem} />
          <Horizontal
            itemWidth={450}
            itemHeight={450}
            itemRadius={10}
            itemPadding={50}
            buttonSize={50}
            isParent={false}
            lineName={this.state.childName}
            getItem={this.getItem} />
        </MainColumn>
      </Main>
    );
  }
} // App
