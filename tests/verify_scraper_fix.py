import sys
import os

# MOVE THIS TO THE TOP
# Add parent directory and venv site-packages to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../venv_gpt/Lib/site-packages')))

import logging
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.web_scraper import WebScraper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_incremental_scraping():
    """Test that callback is triggered and max_pages is respected"""
    # Initialize scraper
    scraper = WebScraper(headless=True, follow_links=True, max_depth=1)
    
    scraped_count = 0
    def my_callback(doc):
        nonlocal scraped_count
        scraped_count += 1
        print(f"Callback triggered for: {doc.metadata.get('source')} (Total: {scraped_count})")

    # Use a real URL but limit to 2 pages to be fast
    test_url = "https://quotes.toscrape.com/"
    print(f"Starting test scrape of {test_url} with max_pages=3...")
    
    docs = scraper.scrape_urls([test_url], max_pages=3, callback=my_callback)
    
    print(f"\nFinal Statistics:")
    print(f"Total documents returned: {len(docs)}")
    print(f"Callback trigger count: {scraped_count}")
    
    # Assertions
    assert len(docs) <= 3, f"Expected at most 3 docs, got {len(docs)}"
    assert scraped_count == len(docs), f"Callback count ({scraped_count}) should match doc count ({len(docs)})"
    assert len(docs) > 0, "Expected at least one document scraped"
    
    print("\n[SUCCESS] Incremental scraping and max_pages test passed!")

if __name__ == "__main__":
    try:
        test_incremental_scraping()
    except Exception as e:
        print(f"\n[FAILURE] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
