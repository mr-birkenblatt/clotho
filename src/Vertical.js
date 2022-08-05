import React, { PureComponent } from "react";
import styled from "styled-components";

const Outer = styled.div`
  position: relative;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
`;

const IntersectBoxTop = styled.div`
  width: 100%;
  height: 30%;
  position: absolute;
  left: 0;
  top: 0;
  pointer-events: none;
`;

const IntersectBoxBottom = styled.div`
  width: 100%;
  height: 30%;
  position: absolute;
  left: 0;
  bottom: 0;
  pointer-events: none;
`;

const Band = styled.div`
  width: 100%;
  height: 100%;
  background-color: green;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: scroll;

  &::-webkit-scrollbar {
    display: none;
  }

  -ms-overflow-style: none;
  scrollbar-width: none;
  scroll-snap-type: y mandatory;
`;

const Pad = styled.div`
  height: ${props => props.padSize};
  width: 100%;
`;

const Item = styled.div`
  width: 100%;
  height: auto;
  scroll-snap-align: start;
  background-color: cornflowerblue;
`;

export default class Vertical extends PureComponent {
  constructor(props) {
    super(props);
    this.state = {
      offset: 0,
      itemCount: 4,
      padSize: 0,
    };
    this.topBox = React.createRef();
    this.bottomBox = React.createRef();
  }

  render() {
    const { getItem } = this.props;
    const { padSize, itemCount, offset } = this.state;
    return (
      <Outer>
        <IntersectBoxTop ref={this.topBox} />
        <IntersectBoxBottom res={this.bottomBox} />
        <Band>
          <Pad padSize={padSize} />
          {
            [...Array(itemCount).keys()].map(ix => {
              const realIx = offset + ix;
              return (
                <Item key={realIx}>
                  {getItem(realIx)}
                </Item>
              );
            })
          }
        </Band>
      </Outer>
    );
  }
} // Vertical
