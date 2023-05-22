from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)

def test_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_healthcheck():
    """Test the healthcheck endpoint"""
    response = client.get("/api/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"healthcheck": "Everything OK!"}

def test_get_open_graph_image():
    """Test the get_open_graph_image endpoint"""
    response = client.post("/api/v1/og/?url=https://www.bbc.co.uk/news/uk-england-dorset-65668603")
    assert response.status_code == 200
    assert response.json() == {"image": "https://ichef.bbci.co.uk/news/1024/branded_news/DD18/production/_129800665_mediaitem129800664.jpg"}

def test_get_top_vader_from_db():
    """Test the endpoint that retrives top stored news articles"""
    response = client.get("/api/v1/vader/summary/pos/top")
    assert response.status_code == 200

def test_get_vader_scored_bbc_news_feed():
    """Test the al purpose bbc news endpoint"""
    category = "england"  # Provide a category for testing

    response = client.get(f"/api/v1/vader/live/{category}")
    assert response.status_code == 200
