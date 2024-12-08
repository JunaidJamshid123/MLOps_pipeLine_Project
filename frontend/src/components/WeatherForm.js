import React, { useState } from 'react';
import './WeatherForm.css'; // Import the custom CSS

const WeatherForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    humidity: '',
    wind_speed: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      humidity: parseFloat(formData.humidity),
      wind_speed: parseFloat(formData.wind_speed)
    });
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="weather-form-container">
      <div className="weather-form-box">
        <h2 className="weather-form-title">Weather Prediction</h2>
        <form onSubmit={handleSubmit} className="weather-form">
          <div className="input-group">
            <label htmlFor="humidity" className="input-label">Humidity (%)</label>
            <input
              type="number"
              id="humidity"
              name="humidity"
              min="0"
              max="100"
              value={formData.humidity}
              onChange={handleChange}
              className="input-field"
              placeholder="Enter humidity percentage"
              required
            />
          </div>
          <div className="input-group">
            <label htmlFor="wind_speed" className="input-label">Wind Speed (m/s)</label>
            <input
              type="number"
              id="wind_speed"
              name="wind_speed"
              min="0"
              step="0.01"
              value={formData.wind_speed}
              onChange={handleChange}
              className="input-field"
              placeholder="Enter wind speed"
              required
            />
          </div>
          <button
            type="submit"
            className="submit-button"
          >
            Get Temperature Prediction
          </button>
        </form>
      </div>
    </div>
  );
};

export default WeatherForm;
