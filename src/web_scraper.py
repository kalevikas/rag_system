"""
Web Scraper with Selenium for Knowledge Base Articles
Supports dynamic content loading and robust error handling
"""
import os
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse
import hashlib

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class WebScraper:
    """Scrape web pages using Selenium for dynamic content"""
    
    def __init__(self, headless: bool = True, page_load_timeout: int = 30, 
                 follow_links: bool = True, max_depth: int = 1, path_pattern: Optional[str] = None):
        """
        Initialize web scraper
        
        Args:
            headless: Run browser in headless mode
            page_load_timeout: Maximum time to wait for page load (seconds)
            follow_links: Whether to follow and scrape internal KB links
            max_depth: Maximum depth for link following (0 = no following, 1 = one level deep)
            path_pattern: Optional path pattern to filter links (e.g. '/kb/')
        """
        self.headless = headless
        self.page_load_timeout = page_load_timeout
        self.follow_links = follow_links
        self.max_depth = max_depth
        self.path_pattern = path_pattern
        self.driver = None
        logger.info(f"WebScraper initialized (headless={headless}, follow_links={follow_links}, max_depth={max_depth}, path_pattern={path_pattern})")
    
    def _init_driver(self):
        """Initialize Selenium WebDriver with Chrome"""
        # Check if existing driver is still alive
        if self.driver is not None:
            try:
                # Test if driver is still responsive
                _ = self.driver.current_window_handle
                return self.driver
            except:
                # Driver crashed, clean it up
                logger.warning("Existing WebDriver is unresponsive, restarting...")
                try:
                    self.driver.quit()
                except:
                    pass
                self.driver = None
        
        # Create new driver
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-plugins')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.page_load_timeout)
            logger.info("[OK] Selenium WebDriver initialized")
        except Exception as e:
            logger.error(f"Error initializing WebDriver: {str(e)}")
            logger.error("Make sure Chrome and ChromeDriver are installed and not blocked by antivirus")
            self.driver = None
            raise
        
        return self.driver
    
    def close(self):
        """Close Selenium WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.info("[OK] WebDriver closed")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {str(e)}")
    
    def _is_internal_link(self, base_url: str, href: str) -> bool:
        """Check if a link is an internal link (including pagination)"""
        if not href:
            return False
        
        try:
            parsed_base = urlparse(base_url)
            
            # Convert relative URLs to absolute
            if not href.startswith('http'):
                href = urljoin(base_url, href)
            
            parsed = urlparse(href)
            
            # Check if same domain
            is_same_domain = parsed.netloc == parsed_base.netloc
            
            # Check if it matches the path pattern (if provided)
            is_pattern_match = True
            if self.path_pattern:
                is_pattern_match = self.path_pattern in parsed.path
            
            # Exclude certain paths (like login, sign-up, etc.)
            excluded_paths = ['/login', '/signup', '/sign-in', '/register', '/search', '/community']
            is_excluded = any(ex in parsed.path.lower() for ex in excluded_paths)
            
            # Allow pagination (URLs with ?page=)
            has_pagination = 'page=' in parsed.query
            
            return is_same_domain and is_pattern_match and not is_excluded
        except:
            return False
    
    def _discover_links(self, url: str, soup: BeautifulSoup, driver) -> List[str]:
        """Discover all internal KB links on a page including pagination"""
        links = set()
        
        try:
            # Find all anchor tags
            for anchor in soup.find_all('a', href=True):
                href = anchor.get('href', '').strip()
                
                if self._is_internal_link(url, href):
                    # Convert to absolute URL
                    full_url = urljoin(url, href)
                    # Remove URL fragments but keep query params (for pagination)
                    full_url = full_url.split('#')[0]
                    links.add(full_url)
            
            # Also check for links using Selenium (for dynamically loaded content)
            try:
                selenium_anchors = driver.find_elements(By.TAG_NAME, 'a')
                for anchor in selenium_anchors:
                    try:
                        href = anchor.get_attribute('href')
                        if href and self._is_internal_link(url, href):
                            full_url = href.split('#')[0]
                            links.add(full_url)
                    except:
                        continue
            except:
                pass
            
            # Check for pagination - look for "Next" button
            pagination_found = False
            for anchor in soup.find_all('a', href=True):
                text = anchor.get_text(strip=True).lower()
                if text in ['next', 'next »', '»', 'next page']:
                    href = anchor.get('href', '').strip()
                    if href:
                        next_url = urljoin(url, href)
                        links.add(next_url)
                        pagination_found = True
                        logger.info(f"Found pagination link: {next_url}")
            
            logger.info(f"Discovered {len(links)} internal KB links on {url} (pagination: {pagination_found})")
        except Exception as e:
            logger.error(f"Error discovering links: {str(e)}")
        
        return sorted(list(links))
    
    def _extract_structured_content(self, element, soup_element) -> str:
        """
        Extract content while preserving structure (tables, lists, links, etc.) in Markdown format
        Performs comprehensive HTML parsing to capture all page content
        
        Args:
            element: Selenium WebElement or None
            soup_element: BeautifulSoup element
            
        Returns:
            Markdown-formatted content string
        """
        if not soup_element:
            return ""
        
        # Preserve links as Markdown links [text](url)
        for link in soup_element.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            if href and text:
                # Make relative URLs absolute
                from urllib.parse import urljoin
                # We'll need the base URL - for now, keep as-is
                link.replace_with(f"[{text}]({href})")

        
        # Note images with alt text
        for img in soup_element.find_all('img'):
            alt = img.get('alt', 'image')
            src = img.get('src', '')
            if alt or src:
                img.replace_with(f"\n[Image: {alt}]\n")
        
        # Convert HTML tables to Markdown tables
        for table in soup_element.find_all('table'):
            markdown_table = self._html_table_to_markdown(table)
            if markdown_table:
                # Replace table HTML with markdown
                table.replace_with(f"\n\n{markdown_table}\n\n")
        
        # Convert ordered lists to numbered markdown (preserve nested structure)
        for ol in soup_element.find_all('ol'):
            items = ol.find_all('li', recursive=False)
            markdown_list = "\n".join([f"{i+1}. {item.get_text(strip=True)}" for i, item in enumerate(items)])
            ol.replace_with(f"\n\n{markdown_list}\n\n")
        
        # Convert unordered lists to bullet markdown
        for ul in soup_element.find_all('ul'):
            items = ul.find_all('li', recursive=False)
            markdown_list = "\n".join([f"- {item.get_text(strip=True)}" for item in items])
            ul.replace_with(f"\n\n{markdown_list}\n\n")
        
        # Bold important headers
        for h2 in soup_element.find_all('h2'):
            h2.replace_with(f"\n\n**{h2.get_text(strip=True)}**\n\n")
        
        for h3 in soup_element.find_all('h3'):
            h3.replace_with(f"\n\n**{h3.get_text(strip=True)}**\n\n")
        
        for h4 in soup_element.find_all('h4'):
            h4.replace_with(f"\n\n**{h4.get_text(strip=True)}**\n\n")
        
        # Code blocks
        for code in soup_element.find_all('code'):
            code_text = code.get_text(strip=True)
            if code_text:
                code.replace_with(f"`{code_text}`")
        
        for pre in soup_element.find_all('pre'):
            pre_text = pre.get_text(strip=True)
            if pre_text:
                pre.replace_with(f"\n```\n{pre_text}\n```\n")
        
        # Get final text with structure preserved
        content = soup_element.get_text(separator='\n', strip=True)
        return content
    
    def _html_table_to_markdown(self, table) -> str:
        """Convert HTML table to Markdown table format"""
        try:
            rows = table.find_all('tr')
            if not rows:
                return ""
            
            markdown_rows = []
            
            # Process header row
            header_row = rows[0]
            headers = header_row.find_all(['th', 'td'])
            if headers:
                header_text = " | ".join([h.get_text(strip=True) for h in headers])
                markdown_rows.append(f"| {header_text} |")
                markdown_rows.append("|" + " --- |" * len(headers))
            
            # Process data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_text = " | ".join([c.get_text(strip=True) for c in cells])
                    markdown_rows.append(f"| {row_text} |")
            
            return "\n".join(markdown_rows)
        except Exception as e:
            logger.debug(f"Error converting table to markdown: {e}")
            return ""
    
    def save_extracted_content(self, url: str, parsed_content: str, html_content: str = None) -> str:
        """
        Save extracted content to file for verification
        
        Args:
            url: Source URL
            parsed_content: Parsed text content
            html_content: Original HTML (optional)
            
        Returns:
            Path to saved file
        """
        try:
            # Create directory if it doesn't exist
            save_dir = Path("data/extracted_html")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Save parsed content
            parsed_file = save_dir / f"parsed_{timestamp}_{url_hash}.txt"
            with open(parsed_file, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n")
                f.write(f"Extracted: {datetime.utcnow().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                f.write(parsed_content)
            
            logger.info(f"Saved parsed content to: {parsed_file}")
            
            # Optionally save raw HTML
            if html_content:
                html_file = save_dir / f"html_{timestamp}_{url_hash}.html"
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content[:50000])  # Save first 50K to avoid huge files
                logger.debug(f"Saved HTML to: {html_file}")
            
            return str(parsed_file)
            
        except Exception as e:
            logger.error(f"Error saving extracted content: {e}")
            return ""
    
    def _parse_html_comprehensive(self, html_content: str, base_url: str = "") -> str:
        """
        Comprehensive HTML parsing to extract and organize all page content
        Preserves structure and extracts all relevant elements
        
        Args:
            html_content: Raw HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            Structured text content with all parsed elements
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer']):
                element.decompose()
            
            output = []
            
            # Extract page title/heading
            title = soup.find(['h1', 'title'])
            if title:
                output.append(f"# {title.get_text(strip=True)}\n")
            
            # Extract metadata description
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                output.append(f"Description: {meta_desc.get('content')}\n")
            
            # Extract main content
            main_content = soup.find(['main', 'article', 'div[role="main"]'])
            if not main_content:
                main_content = soup.find('body')
            
            if main_content:
                # Extract all headers with hierarchy
                for header in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    level = int(header.name[1])
                    header_text = header.get_text(strip=True)
                    output.append(f"{'#' * level} {header_text}\n")
                
                # Extract all paragraphs
                for para in main_content.find_all('p'):
                    text = para.get_text(strip=True)
                    if text:
                        output.append(f"{text}\n")
                
                # Extract all lists
                for ul in main_content.find_all(['ul', 'ol']):
                    list_type = "ordered" if ul.name == 'ol' else "unordered"
                    output.append(f"\n[{list_type} list]\n")
                    for i, li in enumerate(ul.find_all('li', recursive=False), 1):
                        prefix = f"{i}." if ul.name == 'ol' else "-"
                        output.append(f"{prefix} {li.get_text(strip=True)}\n")
                    output.append("\n")
                
                # Extract all tables
                for table in main_content.find_all('table'):
                    output.append("\n[Table]\n")
                    markdown_table = self._html_table_to_markdown(table)
                    if markdown_table:
                        output.append(f"{markdown_table}\n")
                    output.append("\n")
                
                # Extract all code blocks
                for code_block in main_content.find_all('pre'):
                    code_text = code_block.get_text(strip=True)
                    if code_text:
                        output.append(f"\n```\n{code_text}\n```\n\n")
                
                # Extract all images with alt text
                for img in main_content.find_all('img'):
                    alt_text = img.get('alt', 'Image')
                    src = img.get('src', '')
                    if src or alt_text:
                        output.append(f"[Image: {alt_text}]\n")
                
                # Extract all links
                links_section = []
                for link in main_content.find_all('a', href=True):
                    href = link.get('href', '')
                    text = link.get_text(strip=True)
                    if href and text:
                        # Resolve relative URLs
                        if base_url and not href.startswith('http'):
                            href = urljoin(base_url, href)
                        links_section.append(f"- [{text}]({href})")
                
                if links_section:
                    output.append("\n[Links found on page]\n")
                    output.extend(links_section)
                    output.append("\n")
                
                # Extract blockquotes
                for blockquote in main_content.find_all('blockquote'):
                    quote_text = blockquote.get_text(strip=True)
                    if quote_text:
                        output.append(f"> {quote_text}\n\n")
            
            content = "".join(output)
            
            # Final cleanup
            import re
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Remove excessive blank lines
            content = re.sub(r'[ \t]+', ' ', content)  # Remove excessive spaces
            
            return content
            
        except Exception as e:
            logger.error(f"Error in comprehensive HTML parsing: {e}")
            return ""
    
    def _html_table_to_markdown(self, table) -> str:
        """Convert HTML table to Markdown table format"""
        try:
            rows = table.find_all('tr')
            if not rows:
                return ""
            
            markdown_rows = []
            
            # Process header row
            header_row = rows[0]
            headers = header_row.find_all(['th', 'td'])
            if headers:
                header_text = " | ".join([h.get_text(strip=True) for h in headers])
                markdown_rows.append(f"| {header_text} |")
                markdown_rows.append("|" + " --- |" * len(headers))
            
            # Process data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_text = " | ".join([c.get_text(strip=True) for c in cells])
                    markdown_rows.append(f"| {row_text} |")
            
            return "\n".join(markdown_rows)
        except Exception as e:
            logger.debug(f"Error converting table to markdown: {e}")
            return ""
    
    def _scroll_and_load_all_content(self, driver) -> str:
        """
        Scroll through the entire page to load all dynamically loaded content
        Returns the complete page source after scrolling
        
        Args:
            driver: Selenium WebDriver instance
            
        Returns:
            Complete page source after scrolling to end
        """
        try:
            logger.info("Scrolling page to load all dynamic content...")
            
            # Get initial page height
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_count = 0
            max_scrolls = 50  # Prevent infinite scrolling
            scroll_pause_time = 0.5
            
            while scroll_count < max_scrolls:
                # Scroll down
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time)
                
                # Calculate new height
                new_height = driver.execute_script("return document.body.scrollHeight")
                
                # If page height hasn't changed, we've reached the bottom
                if new_height == last_height:
                    logger.info(f"Reached end of page after {scroll_count} scrolls")
                    break
                
                last_height = new_height
                scroll_count += 1
            
            # Scroll back to top to ensure all content is processed
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(0.3)
            
            logger.info(f"Page scrolling complete: {scroll_count} scrolls, final height: {last_height}px")
            
            # Return the complete page source after all scrolling
            return driver.page_source
            
        except Exception as e:
            logger.warning(f"Error during page scrolling: {e}")
            # Return current page source even if scrolling failed
            return driver.page_source
    
    def scrape_url(self, url: str, discover_links: bool = False) -> tuple[Optional[Document], List[str]]:
        """
        Scrape a single URL and return as LangChain Document
        
        Args:
            url: URL to scrape
            discover_links: Whether to discover internal links on this page
            
        Returns:
            Tuple of (Document object with content and metadata or None if failed, list of discovered links)
        """
        discovered_links = []
        try:
            driver = self._init_driver()
            
            logger.info(f"Loading page: {url}")
            
            # Set a very short timeout to fail fast if the browser is stuck
            driver.set_page_load_timeout(15)  # Absolute max 15 seconds for page load
            
            timeout_occurred = False
            try:
                driver.get(url)
            except TimeoutException:
                logger.warning(f"Page load timeout for {url} - trying to get partial page source")
                timeout_occurred = True
                # Get whatever was loaded so far instead of giving up
                try:
                    page_source = driver.page_source
                    if page_source and len(page_source) > 100:
                        logger.info(f"Retrieved partial page source ({len(page_source)} bytes) despite timeout")
                    else:
                        logger.warning("Partial page source too small, using fallback scraper")
                        self.close()
                        return None, []
                except Exception as e:
                    logger.error(f"Could not get partial page source: {e}")
                    self.close()
                    return None, []
            except WebDriverException as e:
                logger.error(f"WebDriver error loading {url}: {str(e)}")
                self.close()  # Force close crashed driver
                return None, []
            
            # Short wait for body element only (skip if timeout)
            if not timeout_occurred:
                try:
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                except TimeoutException:
                    logger.warning("Body element not found, using partial page")
                
                # Only scroll if initial page load succeeded
                time.sleep(0.3)
                page_source = self._scroll_and_load_all_content(driver)
            else:
                # If timeout happened, don't try to scroll, just use what we have
                logger.info("Skipping scroll step due to timeout")
                try:
                    time.sleep(0.5)  # Wait for any DOM updates
                    page_source = driver.page_source
                except:
                    pass  # Use the page_source we already got
            
            # Alternative: extract text directly from DOM using JavaScript
            # Only attempt if initial page load succeeded (not timed out)
            if not timeout_occurred:
                try:
                    logger.info("Extracting rendered text from DOM using JavaScript...")

                    dom_text = driver.execute_script("""
                        // Remove script and style elements
                        var scripts = document.querySelectorAll('script, style, nav, footer');
                        scripts.forEach(el => el.remove());
                        
                        // Get all text content
                        return document.body.innerText || document.body.textContent;
                    """)
                    
                    if dom_text and len(dom_text.strip()) > len(page_source) / 10:  # If DOM text is substantial
                        logger.info(f"DOM extraction successful: {len(dom_text)} characters")
                        # Use DOM text as page source for parsing
                        page_source = dom_text
                except Exception as e:
                    logger.debug(f"DOM text extraction failed: {e}, using page_source")
            else:
                logger.info("Skipping DOM extraction due to timeout")
            
            # Parse the scrolled page source with comprehensive HTML parsing
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract title
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else "Untitled"
            
            # Try to extract title from Selenium if not found
            if title_text == "Untitled":
                try:
                    title_element = driver.find_element(By.TAG_NAME, "h1")
                    title_text = title_element.text.strip() or "Untitled"
                except:
                    try:
                        title_text = driver.title or "Untitled"
                    except:
                        title_text = "Untitled"
            
            # Use comprehensive HTML parsing to extract all page content
            logger.info("Performing comprehensive HTML parsing...")
            content = self._parse_html_comprehensive(page_source, url)
            
            # If comprehensive parsing returned minimal content, try fallback methods
            if not content or len(content) < 100:
                logger.info("Comprehensive parsing returned minimal content, trying fallback methods...")
                
                # Extract main content (try multiple selectors)
                main_content = (
                    soup.find('article') or 
                    soup.find('main') or 
                    soup.find('div', class_='content') or
                    soup.find('div', id='content') or
                    soup.find('div', class_='article-content')
                )
                
                if main_content:
                    # Remove unwanted elements
                    for element in main_content(["script", "style", "nav", "header", "footer", "aside"]):
                        element.decompose()
                    
                    # Get structured content (preserves tables, lists, etc. in Markdown)
                    content = self._extract_structured_content(None, main_content)
                else:
                    # Try multiple content extraction strategies
                    logger.info("Main content container not found, trying alternative extraction methods...")
                    
                    # Try to extract from common content containers
                    content_candidates = [
                        soup.find('div', class_=lambda x: x and 'post' in x.lower()),
                    soup.find('div', class_=lambda x: x and 'article' in x.lower()),
                    soup.find('div', class_=lambda x: x and 'content' in x.lower()),
                    soup.find('section'),
                    soup.find('div', id=lambda x: x and 'content' in x.lower()),
                ]
                
                content = ""
                for candidate in content_candidates:
                    if candidate:
                        try:
                            # Remove unwanted elements
                            for element in candidate(["script", "style", "nav", "header", "footer", "aside"]):
                                element.decompose()
                            content = self._extract_structured_content(None, candidate)
                            if len(content) > 100:  # Only use if substantial content found
                                logger.info(f"Found content ({len(content)} chars) in container")
                                break
                        except:
                            continue
                
                # Last resort: get all text content from body
                if not content or len(content) < 100:
                    logger.info("Using full page text extraction (low quality)")
                    content = soup.get_text(separator='\n', strip=True)
            
            # Clean up excessive whitespace
            import re
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content)
            
            # Extract metadata
            metadata = {
                'source': url,
                'title': title_text,
                'scraped_at': datetime.utcnow().isoformat(),
                'content_length': len(content),
                'char_count': len(content),
                'word_count': len(content.split()),
                'domain': urlparse(url).netloc,
                'doc_type': 'web_page',
                'scroll_extraction': True
            }
            
            # Try to extract category or tags from meta tags
            category = soup.find('meta', {'name': 'category'})
            if category:
                metadata['category'] = category.get('content', '')
            
            # Try to extract description
            description = soup.find('meta', {'name': 'description'})
            if description:
                metadata['description'] = description.get('content', '')
            
            # Add HTML parsing method info to metadata
            metadata['extraction_method'] = 'comprehensive_html_parsing'
            metadata['html_parsed'] = True
            
            # Discover internal KB links if requested
            if discover_links:
                discovered_links = self._discover_links(url, soup, driver)
            
            logger.info(f"[OK] Successfully scraped with comprehensive HTML parsing: {title_text} ({len(content)} characters, {metadata['word_count']} words)")
            
            return Document(
                page_content=content,
                metadata=metadata
            ), discovered_links
            
        except WebDriverException as e:
            logger.error(f"WebDriver error scraping {url}: {str(e)}")
            return None, []
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None, []
    
    def scrape_urls(self, urls: List[str], max_pages: int = 50, callback: Optional[callable] = None) -> List[Document]:
        """
        Scrape multiple URLs with optional recursive link following
        
        Args:
            urls: List of seed URLs to scrape
            max_pages: Maximum number of pages to scrape in total
            callback: Optional function called after each successful scrape: callback(doc)
            
        Returns:
            List of Document objects (includes linked pages if follow_links=True)
        """
        documents = []
        visited_urls = set()
        
        # Queue: (url, depth)
        url_queue = [(url, 0) for url in urls]
        
        try:
            while url_queue and len(visited_urls) < max_pages:
                url, depth = url_queue.pop(0)
                
                # Skip if already visited
                if url in visited_urls:
                    continue
                
                visited_urls.add(url)
                
                logger.info(f"Scraping [{depth}] ({len(visited_urls)}/{max_pages}): {url}")
                
                # Decide whether to discover links based on depth and total page limit
                should_discover = self.follow_links and depth < self.max_depth and len(visited_urls) < max_pages
                
                # Scrape the URL
                try:
                    doc, discovered_links = self.scrape_url(url, discover_links=should_discover)
                except Exception as e:
                    logger.error(f"Exception scraping {url}: {e}")
                    doc = None
                    discovered_links = []
                
                if doc:
                    documents.append(doc)
                    logger.info(f"Scraped: {doc.metadata.get('title', 'Untitled')} ({len(doc.page_content)} chars)")
                    
                    # Trigger callback if provided
                    if callback:
                        try:
                            callback(doc)
                        except Exception as cb_err:
                            logger.error(f"Error in scraper callback: {cb_err}")
                else:
                    logger.warning(f"Failed to scrape: {url}")
                
                # Add discovered links to queue if we should follow them
                if should_discover and discovered_links:
                    new_links = [link for link in discovered_links if link not in visited_urls]
                    # Respect max_pages in queue addition
                    remaining_slots = max_pages - len(visited_urls)
                    if remaining_slots > 0:
                        url_queue.extend([(link, depth + 1) for link in new_links[:remaining_slots]])
                
                # Small delay between requests
                time.sleep(0.5)
            
            if len(visited_urls) >= max_pages:
                logger.info(f"Reached maximum page limit: {max_pages}")
                
            logger.info(f"Successfully scraped {len(documents)} documents from {len(visited_urls)} URLs")
            
        except Exception as e:
            logger.error(f"Error in scrape_urls: {e}")
            return documents
        finally:
            self.close()
            
        return documents
    
    def save_discovered_links(self, all_urls: List[str], seed_urls: List[str], output_file: str = "data/discovered_links.txt"):
        """Save all discovered links to a file for verification"""
        try:
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            discovered = [url for url in all_urls if url not in seed_urls]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("DISCOVERED LINKS REPORT\n")
                f.write("="*80 + "\n\n")
                f.write(f"Total URLs Scraped: {len(all_urls)}\n")
                f.write(f"Seed URLs: {len(seed_urls)}\n")
                f.write(f"Discovered Links: {len(discovered)}\n\n")
                
                f.write("="*80 + "\n")
                f.write(f"SEED URLs ({len(seed_urls)})\n")
                f.write("="*80 + "\n")
                for i, url in enumerate(seed_urls, 1):
                    f.write(f"{i}. {url}\n")
                
                if discovered:
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"DISCOVERED LINKS ({len(discovered)})\n")
                    f.write("="*80 + "\n")
                    for i, url in enumerate(discovered, 1):
                        f.write(f"{i}. {url}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("ALL URLs (Combined)\n")
                f.write("="*80 + "\n")
                for i, url in enumerate(all_urls, 1):
                    f.write(f"{i}. {url}\n")
            
            logger.info(f"Saved discovered links report to: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error saving discovered links: {str(e)}")
            return None
    
    def scrape_from_file(self, filepath: str) -> List[Document]:
        """
        Read URLs from a file and scrape them
        
        Args:
            filepath: Path to file containing URLs (one per line)
            
        Returns:
            List of Document objects
        """
        try:
            with open(filepath, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            logger.info(f"Loaded {len(urls)} URLs from {filepath}")
            return self.scrape_urls(urls)
            
        except Exception as e:
            logger.error(f"Error reading URLs from file: {str(e)}")
            return []
