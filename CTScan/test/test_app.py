import sys
import os
import pytest
import random
import json
import numpy as np
from io import BytesIO
from flask import url_for

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app, plasma_list, serum_list

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_base_route(client):
    """Test the base route."""
    rv = client.get('/')
    assert rv.status_code == 200

def test_about_route(client):
    """Test the about route."""
    rv = client.get('/about')
    assert rv.status_code == 200
    assert b'About' in rv.data

def test_metaboliteanalysis_route(client):
    """Test the metabolite analysis route."""
    rv = client.get('/metaboliteanalysis')
    assert rv.status_code == 200

def test_upload_metabolite_data(client):
    """Test the upload metabolite data route."""
    data = {
        f'plasma_{i}': random.uniform(0.1, 1.0) for i in range(len(plasma_list))
    }
    data.update({
        f'serum_{i}': random.uniform(0.1, 1.0) for i in range(len(serum_list))
    })
    rv = client.post('/upload_metabolite_data', data=data)
    assert rv.status_code == 200

def test_fill_sample_data(client):
    """Test the fill sample data route."""
    rv = client.post('/fill_sample_data')
    assert rv.status_code == 200
    # Check for a known string or HTML element from the rendered template
    # Assuming there are elements or text that mention plasma and serum sample data
