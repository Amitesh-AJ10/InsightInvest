"""
News Scrapping Module for InsightInvest
========================================
Fetches news results from Google News RSS and returns structured data.
"""

import re
import requests
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any
from urllib.parse import quote_plus


# -------- Constants --------

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

TAG_SOURCE_NS = "{http://news.google.com}source"
TAG_MAP = {
    "item": "item",
    "title": "title",
    "link": "link",
    "pubDate": "pubDate",
    "description": "description",
    "source": "source",
}

TAG_CLEANER = re.compile(r"<[^>]+>")  # Remove HTML tags


# -------- Helper Functions --------

def _text(elem: Optional[ET.Element]) -> Optional[str]:
    """Extract and clean text from XML element."""
    return elem.text.strip() if (elem is not None and elem.text) else None


def _resolve_redirect(link: str) -> Optional[str]:
    """Resolve Google News redirect to original publisher URL."""
    try:
        # Try HEAD request first (faster)
        response = requests.head(
            link,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
            timeout=5  # Reduced timeout
        )
        if response.url != link:
            return response.url

        # Fallback to GET if HEAD doesn't redirect
        response = requests.get(
            link,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
            timeout=10 # Reduced timeout
        )
        return response.url
    except Exception:
        return None


# -------- Main Functions --------

def generate_google_news_rss_url(
    query: str,
    hl: str = "en-IN",
    gl: str = "IN",
    ceid: str = "IN:en"
) -> str:
    """
    Generate Google News RSS URL for given search query.
    """
    base_url = "https://news.google.com/rss/search"
    params = f"q={quote_plus(query)}&hl={quote_plus(hl)}&gl={quote_plus(gl)}&ceid={quote_plus(ceid)}"
    return f"{base_url}?{params}"


def fetch_news_items(
    rss_url: str,
    max_results: int = 10,
    follow_redirects: bool = False # OPTIMIZATION: Default to False for speed
) -> List[Dict[str, Any]]:
    """
    Fetch and parse news items from Google News RSS feed.
    """
    try:
        # Fetch RSS feed
        response = requests.get(
            rss_url,
            headers={"User-Agent": USER_AGENT},
            timeout=10
        )
        response.raise_for_status()

        # Parse XML
        root = ET.fromstring(response.content)
        channel = root.find("channel")

        if channel is None:
            return []

        items = []
        # Find all items first
        all_items = channel.findall(TAG_MAP["item"])

        for item_elem in all_items[:max_results]:
            # Extract basic fields
            title = _text(item_elem.find(TAG_MAP["title"]))
            link = _text(item_elem.find(TAG_MAP["link"]))
            published = _text(item_elem.find(TAG_MAP["pubDate"]))

            # Extract and clean description
            desc_raw = _text(item_elem.find(TAG_MAP["description"]))
            snippet = TAG_CLEANER.sub("", desc_raw) if desc_raw else None

            # Extract source (try namespaced first, then plain)
            src_elem = item_elem.find(TAG_SOURCE_NS) or item_elem.find(TAG_MAP["source"])
            source = _text(src_elem)

            # Resolve original link if requested
            original_link = None
            if follow_redirects and link:
                original_link = _resolve_redirect(link)
            else:
                original_link = link # Optimization: Just use the google link if we don't resolve

            # Build news item
            news_item = {
                "title": title or "(no title)",
                "link": link or "",
                "original_link": original_link,
                "source": source,
                "published": published,
                "snippet": snippet,
            }
            items.append(news_item)

        return items

    except requests.RequestException as e:
        print(f"Error fetching news: {e}")
        return []
    except ET.ParseError as e:
        print(f"Error parsing RSS: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in fetch_news_items: {e}")
        return []


def get_stock_news(
    symbol: str,
    max_results: int = 10,
    include_terms: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Convenient function to get news for a specific stock symbol.
    """
    # Build search query
    search_terms = [symbol]
    if include_terms:
        search_terms.extend(include_terms)

    query = " ".join(search_terms)

    # Generate URL and fetch news
    rss_url = generate_google_news_rss_url(query)
    return fetch_news_items(rss_url, max_results=max_results)


def get_news_titles(
    symbol: str,
    max_results: int = 10
) -> List[str]:
    """
    Get just the titles of news articles for a symbol (useful for sentiment analysis).
    """
    news_items = get_stock_news(symbol, max_results=max_results)
    return [item["title"] for item in news_items if item.get("title")]


# -------- Validation Functions --------

def is_valid_symbol(symbol: str) -> bool:
    """Check if symbol looks like a valid stock ticker."""
    if not symbol or len(symbol) > 10:
        return False
    return re.match(r"^[A-Z0-9:.]+$", symbol.upper()) is not None


if __name__ == "__main__":
    # Test the functions
    test_symbol = "AAPL"
    print(f"Testing news scraping for {test_symbol}...")
    url = generate_google_news_rss_url(test_symbol)
    print(f"Generated URL: {url}")
    news = get_stock_news(test_symbol, max_results=5)
    print(f"Found {len(news)} news items")
    if news:
        print(f"First item: {news[0]['title']}")