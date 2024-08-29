import React from 'react';
import "./Navbar.css";
import logout from "../Assets/logout.png";
import logo from "../Assets/logo.png";
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="logo-name">
        <div className="logo"><Link to = "/Home"><img src={logo}></img></Link></div>
      </div>
      <div className="log-out">
        <img src={logout} alt="logout" onClick={() => {
          localStorage.removeItem('token');
          window.location.href = '/';
        }}/>
      </div>
    </nav>
  );
};

export default Navbar;
