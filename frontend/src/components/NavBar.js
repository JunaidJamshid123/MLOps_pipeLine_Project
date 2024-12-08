import React from 'react';
import { Link, useNavigate } from 'react-router-dom';

import './NavBar.css'; // Import the CSS file

const NavBar = () => {
  const navigate = useNavigate();
  const token = localStorage.getItem('token');

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/login');
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-content">
          <div className="navbar-left">
            
            <Link to="/" className="navbar-logo">Weather Predictor</Link>
          </div>

          <div className="navbar-right">
            {token ? (
              <>
                <Link 
                  to="/dashboard" 
                  className="navbar-link"
                >
                  Dashboard
                </Link>
                <button
                  onClick={handleLogout}
                  className="btn-logout"
                >
                  Logout
                </button>
              </>
            ) : (
              <>
                <Link 
                  to="/login" 
                  className="navbar-link"
                >
                  Login
                </Link>
                <Link 
                  to="/register" 
                  className="btn-register"
                >
                  Register
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
