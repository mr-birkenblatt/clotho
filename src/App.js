import React from 'react';
import RequireLogin from './RequireLogin.js';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <RequireLogin />
      </header>
    </div>
  );
}

export default App;
