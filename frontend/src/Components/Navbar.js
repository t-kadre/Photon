import React from 'react';
import "./Navbar.css";
import logout from "../Assets/logout.png";
import logo from "../Assets/logo.png";
import gotohome from "../Assets/gotohome.png";
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="logo-name">
      <Link to="/"><div className="logo"><img src={logo}></img></div></Link>
      </div>
      <div className="nav-btns">
        <div className="go-to-home">
        <Link to="/"><div className="log-out"><img src={gotohome}></img></div></Link>
        </div>
        <div className="log-out" onClick={() => {
            localStorage.removeItem('token');
            window.location.href = '/';
          }}>
          <img src={logout} alt="logout" />
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
