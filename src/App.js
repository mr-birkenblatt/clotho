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
`;


const URL_PREFIX = `${window.location.origin}/api`;


function getChild(name, cb) {
  console.log("child", name);
  fetch(`${URL_PREFIX}/children`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      "parent": name,
      "offset": 0,
      "limit": 1,
      "scorer": "best",
    })
  }).then((resp) => resp.json()).then((obj) => {
    const { links } = obj;
    cb(links[0].child);
  }).catch((e) => {
    console.error(e);
  });
}


function getParent(name, cb) {
  console.log("parent", name);
  fetch(`${URL_PREFIX}/parents`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      "child": name,
      "offset": 0,
      "limit": 1,
      "scorer": "best",
    })
  }).then((resp) => resp.json()).then((obj) => {
    const { links } = obj;
    cb(links[0].parent);
  }).catch((e) => {
    console.error(e);
  });
}


function getChildren(name, offset, limit, cb) {
  fetch(`${URL_PREFIX}/children`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      "parent": name,
      "offset": offset,
      "limit": limit,
      "scorer": "best",
    })
  }).then((resp) => resp.json()).then((obj) => {
    const { links, next } = obj;
    const res = {};
    links.forEach((link, ix) => {
      res[ix + offset] = `**name**: ${link.child} _ix_: ${ix + offset}`;
    });
    if (links.length < limit) {
      if (next > 0 && links.length > 0) {
        getChildren(name, next, limit - links.length, (rec) => {
          cb({...res, ...rec});
        });
        return;
      } else {
        [...Array(limit - links.length).keys()].forEach((ix) => {
          const pos = offset + links.length + ix;
          res[pos] = `no data ${pos}`;
        });
      }
    }
    cb(res);
  }).catch((e) => {
    console.error(e);
  });
}


function getParents(name, offset, limit, cb) {
  fetch(`${URL_PREFIX}/parents`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      "child": name,
      "offset": offset,
      "limit": limit,
      "scorer": "best",
    })
  }).then((resp) => resp.json()).then((obj) => {
    const { links, next } = obj;
    const res = {};
    links.forEach((link, ix) => {
      res[offset + ix] = `**name**: ${link.parent} _ix_: ${ix + offset}`;
    });
    if (links.length < limit) {
      if (next > 0 && links.length > 0) {
        getParents(name, next, limit - links.length, (rec) => {
          cb({...res, ...rec});
        });
        return;
      } else {
        [...Array(limit - links.length).keys()].forEach((ix) => {
          const pos = offset + links.length + ix;
          res[pos] = `no data ${pos}`;
        });
      }
    }
    cb(res);
  }).catch((e) => {
    console.error(e);
  });
}


export default class App extends PureComponent {
  constructor(props) {
    super(props);
    this.state = {};
    this.parentLines = new ContentLoader(5, getParents);
    this.childLines = new ContentLoader(5, getChildren);
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

  getChildLine = (lineName, cb) => {
    getChild(lineName, cb);
  }

  getParentLine = (lineName, cb) => {
    getParent(lineName, cb);
  }

  getLinkItems = (parentLineName, childLineName, parentIndex, childIndex) => {
    return [
      [parentLineName, parentIndex],
      [childLineName, childIndex],
    ];
  }

  renderLinkItem = (val) => {
    return (<span>[L({val[0]}) H({val[1]})]</span>);
  }

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
            buttonSize={50} />
        </MainColumn>
      </Main>
    );
  }
} // App
