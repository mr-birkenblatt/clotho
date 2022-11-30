import React, { PureComponent } from 'react';
import styled from 'styled-components';
import View from './main/View';
import CommentGraph from './graph/CommentGraph';
import { advancedGraph } from './graph/TestGraph';

const DEBUG = false;

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
  margin: 0 auto;
  width: var(--main-size);
  height: 100vh;
  border: 0;
  display: flex;
  flex-direction: column;
  flex-grow: 0;
  overflow: hidden;
`;

type AppProps = Record<string, never>;
type AppState = Record<string, never>;

export default class App extends PureComponent<AppProps, AppState> {
  graph: CommentGraph;

  constructor(props: AppProps) {
    super(props);
    this.state = {};
    this.graph = new CommentGraph(
      DEBUG ? advancedGraph().getApiProvider() : undefined,
    );
  }

  render() {
    return (
      <Main>
        {/* <MainHeader>
          <RequireLogin />
        </MainHeader> */}
        <MainColumn>
          <View graph={this.graph} />
        </MainColumn>
      </Main>
    );
  }
} // App
