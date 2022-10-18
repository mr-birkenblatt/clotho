import React, { PureComponent } from "react";

export default class RequireLogin extends PureComponent {
  constructor(props) {
    super(props);
    this.state = {
      value: "",
      user: localStorage.getItem("user"),
      token: localStorage.getItem("token"),
    };
  }

  logout = (event) => {
    localStorage.removeItem("user");
    localStorage.removeItem("token");
    this.setState({
      user: null,
      token: null,
    });
    event.preventDefault();
  }

  handleChange = (event) => {
    this.setState({value: event.target.value});
  }

  handleSubmit = (event) => {
    const { value } = this.state;
    this.setState({
      user: value,
    })
    localStorage.setItem("user", value);
    event.preventDefault();
  }

  render() {
    const {value, user} = this.state;
    return (
      <div>
        { !user ? (
          <form onSubmit={this.handleSubmit}>
            <label>
              Name:&nbsp;
              <input
                type="text"
                value={value}
                onChange={this.handleChange} />
            </label>
            <input type="submit" value="Submit" />
          </form>
        ) : (
          <p>
            Hello {user}
            <button onClick={this.logout}>Logout</button>
          </p>
        ) }
      </div>
    );
  }
} // RequireLogin