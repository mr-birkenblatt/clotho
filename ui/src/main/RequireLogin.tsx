import React, { PureComponent } from 'react';

type RequireLoginProps = {
  value: string;
  user: string;
};

type RequireLoginState = {
  value: string;
  user: string | null;
  token: string | null;
};

export default class RequireLogin extends PureComponent<
  RequireLoginProps,
  RequireLoginState
> {
  constructor(props: RequireLoginProps) {
    super(props);
    this.state = {
      value: '',
      user: localStorage.getItem('user'),
      token: localStorage.getItem('token'),
    };
  }

  logout = (event: React.MouseEvent<HTMLButtonElement, MouseEvent>): void => {
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    this.setState({
      user: null,
      token: null,
    });
    event.preventDefault();
  };

  handleChange = (event: React.FormEvent<HTMLInputElement>): void => {
    this.setState({ value: event.currentTarget.value });
  };

  handleSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
    const { value } = this.state;
    this.setState({
      user: value,
    });
    localStorage.setItem('user', value);
    event.preventDefault();
  };

  render() {
    const { value, user } = this.state;
    return (
      <div>
        {!user ? (
          <form onSubmit={this.handleSubmit}>
            <label>
              Name:&nbsp;
              <input
                type="text"
                value={value}
                onChange={this.handleChange}
              />
            </label>
            <input
              type="submit"
              value="Submit"
            />
          </form>
        ) : (
          <p>
            Hello {user}
            <button onClick={this.logout}>Logout</button>
          </p>
        )}
      </div>
    );
  }
} // RequireLogin
