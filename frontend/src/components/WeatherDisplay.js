import React from 'react';
import './WeatherDisplay.css'; // Import the custom CSS

const WeatherDisplay = ({ weatherData }) => {
  if (!weatherData) return null;

  return (
    <div className="weather-display-container">
      <div className="weather-display-box">
        <h2 className="weather-display-title">Weather Prediction Results</h2>
        <div className="weather-info">
          <div className="weather-info-item">
            <span className="info-label">Temperature</span>
            <span className="info-value">{weatherData.temperature}°C</span>
          </div>
          <div className="weather-info-item">
            <span className="info-label">Humidity</span>
            <span className="info-value">{weatherData.humidity}%</span>
          </div>
          <div className="weather-info-item">
            <span className="info-label">Wind Speed</span>
            <span className="info-value">{weatherData.wind_speed} m/s</span>
          </div>
          <div className="weather-info-item">
            <span className="info-label">Condition</span>
            <span className="info-value">{weatherData.weather_condition}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WeatherDisplay;
