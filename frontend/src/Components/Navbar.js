import React from 'react';

const Navbar = () => {
  return (
    <nav style={styles.navbar}>
      <h1 style={styles.title}>AI Album Sorter</h1>
      <ul style={styles.navLinks}>
        <li><a href="/">All Albums</a></li>
        <li><a href="#addAlbum">Add Album</a></li>
        <li><a href="#login">LOGIN</a></li>
      </ul>
    </nav>
  );
};

const styles = {
  navbar: {
    backgroundColor: '#333',
    padding: '10px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    color: '#fff',
    margin: '0',
  },
  navLinks: {
    listStyleType: 'none',
    display: 'flex',
    gap: '10px',
  },
};

export default Navbar;