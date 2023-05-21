from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_vader_scores_appended_to_given_bbc_news_feed():
    """Test the all purpose bbc news endpoint"""
    category = "england"  # Provide a category for testing

    response = client.get(f"/api/v1/vader/live/{category}")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)

    for item in data:
        assert "title" in item
        assert "summary" in item
        assert "vaderTitle" in item
        assert "vaderSummary" in item
        assert "id" in item
        assert "imageUrl" in item
        assert "published" in item

    # Assertion to check if the 'vaderTitle' field is a dictionary:
    for item in data:
        assert isinstance(item["vaderTitle"], dict)

    # Assertion to check if the 'imageUrl' field is a URL:
    for item in data:
        assert item["imageUrl"].startswith("http")
