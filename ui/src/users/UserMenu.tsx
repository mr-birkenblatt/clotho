import { connect, ConnectedProps } from 'react-redux';
import React, { PureComponent, ReactNode } from 'react';
import styled from 'styled-components';
import UserActions from './UserActions';
import { RootState } from '../store';

const MenuButton = styled.button`
  appearance: none;
  position: fixed;
  top: 0;
  left: 0;
  text-align: center;
  vertical-align: middle;
  pointer-events: auto;
  cursor: pointer;
  border: 0;
  opacity: 0.8;
  color: #5b5f67;
  border-radius: var(--button-radius);
  width: var(--button-size);
  height: var(--button-size);
  background-color: #393d45;

  &:hover {
    background-color: #4a4e56;
  }
  &:active {
    background-color: #5b5f67;
  }
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

enum ModalState {
  None = 'None',
  MenuOpen = 'MenuOpen',
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
  constructor(props: UserMenuProps) {
    super(props);
    this.state = {
      loginValue: '',
      modal: ModalState.None,
    };
  }

  render(): ReactNode {
    const { user } = this.props;
    const username = user !== undefined ? user.name : undefined;
    const menuText = username !== undefined ? username : 'Login';
    return <MenuButton>{menuText}</MenuButton>;
  }

  // logout = (
  //  event: React.MouseEvent<HTMLButtonElement, MouseEvent>): void => {
  //   localStorage.removeItem('user');
  //   localStorage.removeItem('token');
  //   this.setState({
  //     user: null,
  //     token: null,
  //   });
  //   event.preventDefault();
  // };

  // handleChange = (event: React.FormEvent<HTMLInputElement>): void => {
  //   this.setState({ value: event.currentTarget.value });
  // };

  // handleSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
  //   const { value } = this.state;
  //   this.setState({
  //     user: value,
  //   });
  //   localStorage.setItem('user', value);
  //   event.preventDefault();
  // };

  // render() {
  //   const { value, user } = this.state;
  //   return (
  //     <div>
  //       {!user ? (
  //         <form onSubmit={this.handleSubmit}>
  //           <label>
  //             Name:&nbsp;
  //             <input
  //               type="text"
  //               value={value}
  //               onChange={this.handleChange}
  //             />
  //           </label>
  //           <input
  //             type="submit"
  //             value="Submit"
  //           />
  //         </form>
  //       ) : (
  //         <p>
  //           Hello {user}
  //           <button onClick={this.logout}>Logout</button>
  //         </p>
  //       )}
  //     </div>
  //   );
  // }
} // UserMenu

const connector = connect((state: RootState) => ({
  user: state.userState.currentUser,
}));

export default connector(UserMenu);

type ConnectUser = ConnectedProps<typeof connector>;
