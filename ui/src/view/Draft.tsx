import { connect, ConnectedProps } from 'react-redux';
import React, { PureComponent, ReactNode } from 'react';
import styled from 'styled-components';
import { RootState } from '../store';
import { Cell } from '../graph/GraphView';
import Item, { ItemContent, ItemDiv } from './Item';
import { HNavButton, HOverlay, WMOverlay } from './buttons';

const DraftInput = styled.textarea`
  appearance: none;
  margin: auto 0;
  width: var(--md-size-w);
  height: var(--md-size-h);
  overflow: hidden;
  resize: none;
`;

const DraftSubmit = styled.input`
  appearance: none;
  border-radius: var(--item-radius);
  padding: var(--item-padding);
  cursor: pointer;
  color: var(--main-text);
  background-color: var(--button-background);

  &:hover {
    background-color: var(--button-hover);
  }
  &:active {
    background-color: var(--button-active);
  }
`;

export type DraftMode = {
  parent: Readonly<Cell>;
  child: Readonly<Cell> | undefined;
};

interface DraftProps extends ConnectDraft {
  draft: Readonly<DraftMode> | undefined;
  onClose: () => void;
}

type DraftState = {
  draftValue: string;
  awaitFocus: boolean;
};

class Draft extends PureComponent<DraftProps, DraftState> {
  private readonly inputRef: React.RefObject<HTMLTextAreaElement>;

  constructor(props: DraftProps) {
    super(props);
    this.state = {
      draftValue: '',
      awaitFocus: false,
    };
    this.inputRef = React.createRef();
  }

  componentDidMount(): void {
    this.componentDidUpdate();
  }

  componentDidUpdate(): void {
    const { awaitFocus } = this.state;
    if (awaitFocus && this.inputRef.current) {
      this.inputRef.current.focus();
      this.setState({ awaitFocus: false });
    }
  }

  handleChange = (event: React.FormEvent<HTMLTextAreaElement>): void => {
    this.setState({ draftValue: event.currentTarget.value });
  };

  handleSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
    const { draft } = this.props;
    const { draftValue } = this.state;
    if (draft !== undefined) {
      console.log(draft.parent.mhash, draftValue);
    }
    event.preventDefault();
  };

  handleClose = (event: React.FormEvent<HTMLElement>): void => {
    const { onClose } = this.props;
    onClose();
    event.preventDefault();
  };

  render(): ReactNode {
    const { draft } = this.props;
    if (draft === undefined) {
      return null;
    }
    const { draftValue } = this.state;
    const parent = draft.parent;
    return (
      <form onSubmit={this.handleSubmit}>
        <WMOverlay isVisible={true}>
          <DraftSubmit
            type="submit"
            value="✉"
          />
        </WMOverlay>
        <Item
          cell={parent}
          isLocked={false}
        />
        <ItemDiv>
          <ItemContent isLocked={false}>
            <DraftInput
              value={draftValue}
              ref={this.inputRef}
              onChange={this.handleChange}
            />
          </ItemContent>
        </ItemDiv>
        <HOverlay
          isTop={true}
          isVisible={true}>
          <HNavButton onClick={this.handleClose}>✖</HNavButton>
        </HOverlay>
      </form>
    );
  }
} // Draft

const connector = connect((_state: RootState) => ({}));

export default connector(Draft);

type ConnectDraft = ConnectedProps<typeof connector>;
