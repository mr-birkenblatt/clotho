import React, { PureComponent } from 'react';
import styled from 'styled-components';
import View from './main/View';
import CommentGraph from './misc/CommentGraph';
import { advancedGraph } from './misc/TestGraph';

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

// const ItemMidContent = styled.div<ItemMidContentProps>`
//   display: flex;
//   height: ${(props) => props.buttonSize}px;
//   background-color: green;
//   padding: ${(props) => props.radius}px;
//   border-radius: ${(props) => props.radius}px;
//   align-items: center;
//   justify-content: center;
//   flex-direction: column;
// `;

// type ItemMidContentProps = {
//   buttonSize: number;
//   radius: number;
// };

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

  // getHash: HashCB = (fullKey, callback) => {
  //   this.graph.getHash(fullKey, callback);
  // };

  // getItem: ItemCB = (fullLinkKey, readyCb) => {
  //   return this.graph.getMessage(fullLinkKey, () => readyCb());
  // };

  // getVItem: VItemCB = (lineKey, height, isViewUpdate, vPosType) => {
  //   if (lineKey === undefined) {
  //     return null;
  //   }
  //   return (
  //     <Horizontal
  //       itemWidth={450}
  //       itemHeight={height}
  //       itemRadius={10}
  //       itemPadding={50}
  //       buttonSize={50}
  //       isViewUpdate={isViewUpdate}
  //       vPosType={vPosType}
  //       lineKey={lineKey}
  //       getItem={this.getItem}
  //     />
  //   );
  // };

  // getChildLine: ChildLineCB = (fullKey, callback) => {
  //   this.graph.getChild(fullKey, callback);
  // };

  // getParentLine: ParentLineCB = (fullKey, callback) => {
  //   this.graph.getParent(fullKey, callback);
  // };

  // getLink: LinkCB = (fullLinkKey, parentIndex, readyCb) => {
  //   return this.graph.getTopLink(fullLinkKey, parentIndex, () => readyCb());
  // };

  // getTopLink: TopLinkCB = (fullLinkKey, parentIndex, callback) => {
  //   const res = this.graph.getTopLink(fullLinkKey, parentIndex, callback);
  //   if (res !== undefined) {
  //     callback(res);
  //   }
  // };

  // getSingleLink: SingleLinkCB = (parent, child, callback) => {
  //   this.graph.getSingleLink(parent, child, callback);
  // };

  // renderLink: RenderLinkCB = (link, buttonSize, radius) => {
  //   if (link.invalid) {
  //     return null;
  //   }
  //   return (
  //     <div>
  //       <div>
  //         {link.user}: {link.first}
  //       </div>
  //       <div>
  //         {Object.keys(link.votes).map((voteName) => (
  //           <ItemMidContent
  //             buttonSize={buttonSize}
  //             radius={radius}
  //             key={voteName}>
  //             <span>
  //               {voteName}: {link.votes[voteName]}
  //             </span>
  //           </ItemMidContent>
  //         ))}
  //       </div>
  //     </div>
  //   );
  // };

  render() {
    return (
      <Main>
        {/* <MainHeader>
          <RequireLogin />
        </MainHeader> */}
        <MainColumn>
          <View graph={this.graph} />
          {/* <Vertical
            getItem={this.getVItem}
            getHash={this.getHash}
            getChildLine={this.getChildLine}
            getParentLine={this.getParentLine}
            getLink={this.getLink}
            getTopLink={this.getTopLink}
            getSingleLink={this.getSingleLink}
            renderLink={this.renderLink}
            height={450}
            radius={10}
            buttonSize={50}
          /> */}
        </MainColumn>
      </Main>
    );
  }
} // App
