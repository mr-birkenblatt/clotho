import { connect, ConnectedProps } from 'react-redux';
import React, { PureComponent, ReactNode } from 'react';
import styled from 'styled-components';
import UserActions from './UserActions';
import { RootState } from '../store';
import { setUser } from './UserStateSlice';
import { errHnd } from '../misc/util';
import { Username } from '../api/types';

const Menu = styled.div`
  display: inline-block;
  position: fixed;
  top: 0;
  left: 0;
  text-align: center;
  vertical-align: middle;
  border: 0;
  color: var(--button-text);
  border-radius: var(--button-radius);
  height: var(--button-size);
  background-color: var(--button-background-lit);
`;

const MenuButton = styled.button<LoggedInProps>`
  appearance: none;
  text-align: center;
  vertical-align: middle;
  pointer-events: auto;
  cursor: pointer;
  border: 0;
  opacity: 0.8;
  color: var(--button-text);
  border-radius: var(--button-radius);
  ${(props) => (props.isLoggedIn ? '' : 'width: var(--button-size);')}
  height: var(--button-size);
  background-color: var(--button-background);
  border-style: groove;

  &:hover {
    background-color: var(--button-hover);
  }
  &:active {
    background-color: var(--button-active);
    border-style: inset;
  }
`;

const LogoutButton = styled.button`
  appearance: none;
  padding: 0 1em;
  text-align: center;
  vertical-align: middle;
  pointer-events: auto;
  cursor: pointer;
  border: 0;
  opacity: 0.8;
  color: var(--button-text);
  border-radius: var(--button-radius);
  height: var(--button-size);
  background-color: var(--button-background-lit);

  &:hover {
    color: var(--button-text-hover-lit);
  }
  &:active {
    color: var(--button-text-active-lit);
  }
`;

const Modal = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  pointer-events: none;
`;

const ModalOutside = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  opacity: 0.8;
  background-color: var(--main-background);
  pointer-events: auto;
`;

const ModalMain = styled.div`
  position: relative;
  color: var(--main-text);
  background-color: var(--item-background);
  border-radius: var(--item-radius-sm);
  padding: 0 var(--item-padding);
  pointer-events: auto;
  box-shadow: 0 0 16px var(--item-radius-sm) var(--item-shadow);
`;

const ModalInputText = styled.input`
  appearance: none;
  border-radius: var(--item-radius);
  padding: var(--item-padding);
  border-right-style: none;
  border-top-right-radius: 0;
  border-bottom-right-radius: 0;
`;

const ModalSubmit = styled.input`
  appearance: none;
  border-radius: var(--item-radius);
  border-left-style: none;
  border-top-left-radius: 0;
  border-bottom-left-radius: 0;
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

const ModalPad = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  width: 100%;
  height: var(--item-radius);
  text-align: center;
  vertical-align: middle;
  padding: var(--item-padding) 0;
`;

const ModalExit = styled.button`
  position: absolute;
  top: 0;
  right: 0;
  user-select: none;
  appearance: none;
  text-align: center;
  vertical-align: middle;
  pointer-events: auto;
  cursor: pointer;
  border: 0;
  border-radius: var(--item-radius);
  padding: 0;
  margin: var(--item-padding);
  cursor: pointer;
  background-color: unset;
  color: var(--button-text);

  &:hover {
    color: var(--button-text-hover);
  }
  &:active {
    color: var(--button-text-active);
  }
`;

const Header = styled.span`
  font-size: 1.5em;
`;

const Footer = styled.span`
  font-size: 0.75em;
`;

type LoggedInProps = {
  isLoggedIn: boolean;
};

enum ModalState {
  None = 'None',
  ShowLogin = 'ShowLogin',
  ShowLogout = 'ShowLogout',
}

interface UserMenuProps extends ConnectUser {
  userActions: UserActions;
}

type UserMenuState = {
  loginValue: string;
  modal: ModalState;
};

class UserMenu extends PureComponent<UserMenuProps, UserMenuState> {
  private readonly inputTextRef: React.RefObject<HTMLInputElement>;

  constructor(props: UserMenuProps) {
    super(props);
    this.state = {
      loginValue: '',
      modal: ModalState.None,
    };
    this.inputTextRef = React.createRef();
  }

  private isLoggedIn(): boolean {
    const { user } = this.props;
    return user !== undefined;
  }

  private closeModal(): void {
    this.setState({
      loginValue: '',
      modal: ModalState.None,
    });
  }

  handleMenuClick = (event: React.MouseEvent<HTMLElement>): void => {
    const isLoggedIn = this.isLoggedIn();
    const { modal } = this.state;
    if (modal === ModalState.None) {
      this.setState(
        {
          loginValue: '',
          modal: isLoggedIn ? ModalState.ShowLogout : ModalState.ShowLogin,
        },
        () => {
          if (!isLoggedIn && this.inputTextRef.current) {
            this.inputTextRef.current.focus();
          }
        },
      );
    } else {
      this.closeModal();
    }
    event.preventDefault();
  };

  handleModalOutside = (event: React.MouseEvent<HTMLElement>): void => {
    this.closeModal();
    event.preventDefault();
  };

  handleLogoutClick = (event: React.MouseEvent<HTMLElement>): void => {
    const { userActions, user } = this.props;
    if (user !== undefined) {
      userActions.logout(user.token).then(
        (success) => {
          if (!success) {
            console.warn('logout was not successful!');
          }
          const { dispatch } = this.props;
          dispatch(setUser({ user: undefined }));
          this.closeModal();
        },
        (e) => {
          errHnd(e);
        },
      );
    }
    event.preventDefault();
  };

  handleChange = (event: React.FormEvent<HTMLInputElement>): void => {
    this.setState({ loginValue: event.currentTarget.value });
  };

  handleSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
    const { userActions } = this.props;
    const { loginValue } = this.state;
    userActions.login(loginValue as Username).then(
      (user) => {
        const { dispatch } = this.props;
        dispatch(setUser({ user }));
        this.closeModal();
      },
      (e) => {
        errHnd(e);
      },
    );
    event.preventDefault();
  };

  render(): ReactNode {
    const { user } = this.props;
    const isLoggedIn = user !== undefined;
    const { modal, loginValue } = this.state;
    const username = isLoggedIn ? user.name : undefined;
    const menuText = isLoggedIn ? username : 'Sign In';
    return (
      <Menu>
        <MenuButton
          isLoggedIn={isLoggedIn}
          onClick={this.handleMenuClick}>
          {menuText}
        </MenuButton>
        {modal === ModalState.ShowLogout ? (
          <LogoutButton onClick={this.handleLogoutClick}>
            ⇥ Logout
          </LogoutButton>
        ) : null}
        {modal === ModalState.ShowLogin ? (
          <ModalOutside onClick={this.handleModalOutside} />
        ) : null}
        {modal === ModalState.ShowLogin ? (
          <Modal>
            <ModalMain>
              <ModalPad>
                <Header>Welcome back!</Header>
                <ModalExit onClick={this.handleModalOutside}>✖</ModalExit>
              </ModalPad>
              <form onSubmit={this.handleSubmit}>
                <label>
                  Name:&nbsp;
                  <ModalInputText
                    type="text"
                    value={loginValue}
                    ref={this.inputTextRef}
                    onChange={this.handleChange}
                  />
                </label>
                <ModalSubmit
                  type="submit"
                  value="Login"
                />
              </form>
              <ModalPad>
                <Footer>Sign in to be able to vote and write messages.</Footer>
              </ModalPad>
            </ModalMain>
          </Modal>
        ) : null}
      </Menu>
    );
  }
} // UserMenu

const connector = connect((state: RootState) => ({
  user: state.userState.currentUser,
}));

export default connector(UserMenu);

type ConnectUser = ConnectedProps<typeof connector>;
