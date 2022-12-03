import styled from 'styled-components';

const NavButton = styled.button`
  appearance: none;
  display: inline-block;
  text-align: center;
  vertical-align: middle;
  pointer-events: auto;
  cursor: pointer;
  border: 0;
  opacity: 0.8;
  color: var(--button-text-dim);
  border-radius: var(--button-radius);
  width: var(--button-size);
  height: var(--button-size);
  background-color: var(--button-background);

  &:hover {
    background-color: var(--button-hover);
  }
  &:active {
    background-color: var(--button-active);
  }
`;

export const VNavButton = styled(NavButton)<OverlayProps>`
  display: ${(props) => (props.isVisible ? 'inline-block' : 'none')};
  position: fixed;
  ${(props) => (props.isTop ? 'top' : 'bottom')}: 0;
  right: 0;
`;

export const HOverlay = styled.div<OverlayProps>`
  height: var(--button-size);
  position: absolute;
  left: 0;
  ${(props) => (props.isTop ? 'top' : 'bottom')}: 0;
  display: ${(props) => (props.isVisible ? 'flex' : 'none')};
  justify-content: space-between;
  flex-direction: row;
  flex-wrap: nowrap;
  width: var(--main-size);
  pointer-events: none;
`;

export const HNavButton = styled(NavButton)``;

export const WMOverlay = styled.div<WMProps>`
  width: var(--button-size);
  height: var(--main-size);
  position: absolute;
  right: 0;
  top: 0;
  display: ${(props) => (props.isVisible ? 'flex' : 'none')};
  justify-content: end;
  flex-direction: column;
  flex-wrap: nowrap;
  pointer-events: none;
`;

export const WriteMessageButton = styled(NavButton)`
  font-size: 1.5em;
`;

type OverlayProps = {
  isTop: boolean;
  isVisible: boolean;
};

type WMProps = {
  isVisible: boolean;
};
