import pytest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture(scope="session")
def sample_dates():
    base_date = datetime(2024, 1, 1)
    return [base_date + timedelta(hours=i) for i in range(5)]

@pytest.fixture(scope="session")
def raw_weather_data(sample_dates):
    return pd.DataFrame({
        'temperature': np.random.uniform(15, 25, 5),
        'humidity': np.random.uniform(60, 80, 5),
        'wind_speed': np.random.uniform(4, 8, 5),
        'weather_condition': np.random.choice(['Clear', 'Cloudy', 'Rain'], 5),
        'date_time': [d.strftime('%Y-%m-%d %H:%M:%S') for d in sample_dates]
    })

def test_data_format(raw_weather_data):
    assert 'temperature' in raw_weather_data.columns
    assert 'humidity' in raw_weather_data.columns
    assert 'wind_speed' in raw_weather_data.columns
    assert 'weather_condition' in raw_weather_data.columns
    assert 'date_time' in raw_weather_data.columns

def test_temperature_range(raw_weather_data):
    assert raw_weather_data['temperature'].between(15, 25).all()

def test_humidity_range(raw_weather_data):
    assert raw_weather_data['humidity'].between(60, 80).all()

def test_wind_speed_range(raw_weather_data):
    assert raw_weather_data['wind_speed'].between(4, 8).all()

def test_weather_conditions(raw_weather_data):
    valid_conditions = ['Clear', 'Cloudy', 'Rain']
    assert raw_weather_data['weather_condition'].isin(valid_conditions).all()

def test_datetime_format(raw_weather_data):
    assert pd.to_datetime(raw_weather_data['date_time'], errors='coerce').notna().all()

def test_api_key_exists(api_key):
    assert api_key
    assert len(api_key) > 0

@pytest.fixture(scope="session")
def api_key():
    key = os.getenv('WEATHER_API_KEY')
    if not key:
        pytest.skip("WEATHER_API_KEY environment variable not set")
    return key