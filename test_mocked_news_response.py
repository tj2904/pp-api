from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app=app)

def test_vader_scores_appended_to_given_bbc_news_feed():
    """Test the all purpose bbc news endpoint using mocked RSS feed"""
    category = "england"  # Provide a category for testing

    # Define the mocked API response
    mocked_response = """
        <rss version="2.0">
        <channel>
        <item>
            <title><![CDATA[Manchester Arena attack: Young survivors lack support, study finds]]></title>
            <description><![CDATA[Some young Manchester Arena attack survivors have not received professional support, research finds.]]></description>
            <link>https://www.bbc.co.uk/news/uk-england-manchester-65644397?at_medium=RSS&amp;at_campaign=KARANGA</link>
            <guid isPermaLink="false">https://www.bbc.co.uk/news/uk-england-manchester-65644397</guid>
            <pubDate>Mon, 22 May 2023 06:04:43 GMT</pubDate>
        </item>
        <item>
            <title><![CDATA[Laura Nuttall: Bucket list brain cancer fundraiser dies]]></title>
            <description><![CDATA[The 23-year-old was given 12 months to live five years ago and went on to complete a list of ambitions.]]></description>
            <link>https://www.bbc.co.uk/news/uk-england-lancashire-65460230?at_medium=RSS&amp;at_campaign=KARANGA</link>
            <guid isPermaLink="false">https://www.bbc.co.uk/news/uk-england-lancashire-65460230</guid>
            <pubDate>Mon, 22 May 2023 09:34:58 GMT</pubDate>
        </item>
        </channel>
        </rss>
    """

    # Mock the external API response
    with patch("app.feedparser.parse") as mock_parse:
        mock_parse.return_value = mocked_response

        response = client.get(f"/api/v1/vader/live/{category}")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2  # Assuming the mocked response has 2 items

        for item in data:
            assert "title" in item
            assert "summary" in item
            assert "vaderTitle" in item
            assert "vaderSummary" in item
            assert "id" in item
            assert "imageUrl" in item
            assert "published" in item
