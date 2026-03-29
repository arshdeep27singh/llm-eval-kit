"""
Web Scraper — uses Playwright to extract Q&A pairs from websites.

=== WHY PLAYWRIGHT FOR SCRAPING? ===

Regular HTTP scraping (requests, httpx, BeautifulSoup) only gets the
raw HTML the server sends. But modern websites render content with
JavaScript — the HTML you GET is often just a skeleton that JS fills in.

Playwright launches a REAL browser, executes ALL the JavaScript, and
gives you the FINAL rendered DOM. This means it can scrape:
    - Single Page Apps (React, Vue, etc.)
    - Dynamically loaded content (infinite scroll, AJAX)
    - Content behind "Click to expand" buttons
    - Pages that require cookies/sessions

=== HOW THIS SCRAPER WORKS ===

1. Launch a headless browser
2. Navigate to the target URL
3. Wait for the page to fully render
4. Extract Q&A pairs using CSS selectors (configurable)
5. Output as a YAML eval suite file

=== NEW PLAYWRIGHT CONCEPTS COVERED HERE ===

- Browser launch & context management (manual, not via pytest)
- playwright.sync_api vs async_api
- Request interception (blocking images/fonts for speed)
- page.wait_for_load_state("networkidle") for dynamic content
- page.query_selector_all() for DOM querying
- page.evaluate() for running JS in the browser
- Multiple page handling
- Error recovery and timeouts
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from playwright.sync_api import sync_playwright, Page, Route


@dataclass
class ScrapedQA:
    """One scraped question-answer pair."""
    question: str
    answer: str
    source_url: str
    tags: list[str] = field(default_factory=list)


@dataclass
class ScrapeConfig:
    """Configuration for the scraper.

    Tells the scraper WHERE to find Q&A content on the page.
    Different websites structure their content differently, so
    these CSS selectors need to be customized per site.
    """
    # CSS selector for the container of each Q&A pair
    qa_container_selector: str = ".faq-item"

    # CSS selectors for question and answer WITHIN each container
    question_selector: str = ".question, h3, dt, summary"
    answer_selector: str = ".answer, .content, dd, p"

    # Tags to add to all scraped test cases
    tags: list[str] = field(default_factory=list)

    # Whether to block images/fonts/CSS for faster scraping
    block_resources: bool = True

    # Max time to wait for page load (milliseconds)
    timeout_ms: int = 30000


def scrape_qa_pairs(
    url: str,
    config: ScrapeConfig | None = None,
    *,
    playwright_instance=None,
) -> list[ScrapedQA]:
    """Scrape Q&A pairs from a URL using Playwright.

    === PLAYWRIGHT CONCEPT: sync_playwright() context manager ===

    Playwright has two APIs:
        - sync_api  — blocking, like normal Python (what we use here)
        - async_api — for asyncio, uses await/async

    sync_playwright() is a context manager that:
        1. Starts the Playwright server process
        2. Gives you a `playwright` object to launch browsers
        3. Cleans everything up when you exit the `with` block

    This is different from pytest-playwright, which manages the
    lifecycle for you via fixtures. Here we manage it manually.

    The optional `playwright_instance` parameter allows callers (like tests)
    to pass an already-running Playwright instance, avoiding conflicts
    with pytest-playwright's asyncio event loop.
    """
    if config is None:
        config = ScrapeConfig()

    # ── CONCEPT: Manual browser lifecycle ──
    # In tests, pytest-playwright manages browser/context/page for you.
    # When using Playwright as a LIBRARY (not in tests), you manage it yourself:
    #
    #   playwright → browser → context → page
    #
    # - playwright: The connection to the Playwright server
    # - browser:    A browser instance (Chromium, Firefox, WebKit)
    # - context:    An isolated session (like incognito) with its own cookies
    # - page:       A tab within a context
    #
    # Why separate context from browser?
    # You might want multiple isolated sessions in one browser
    # (e.g., test user A and user B simultaneously).

    if playwright_instance is not None:
        return _scrape_with_playwright(playwright_instance, url, config)

    with sync_playwright() as p:
        return _scrape_with_playwright(p, url, config)


def _scrape_with_playwright(p, url: str, config: ScrapeConfig) -> list[ScrapedQA]:
    """Internal function that performs the actual scraping given a Playwright instance."""
    results: list[ScrapedQA] = []

    # Launch Chromium in headless mode (no visible window)
    # headless=True is the default — the browser runs invisibly.
    # Set headless=False during development to SEE what's happening!
    browser = p.chromium.launch(headless=True)

    # Create a context with a realistic viewport and user agent.
    # Some websites serve different content based on these.
    context = browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    )

    # Set default timeout for all operations in this context
    context.set_default_timeout(config.timeout_ms)

    page = context.new_page()

    # ── CONCEPT: Request Interception ──
    # Playwright can intercept network requests and:
    # - Block them (save bandwidth, go faster)
    # - Modify them (change headers, body)
    # - Respond with fake data (mock APIs)
    #
    # For scraping, we block images/fonts/stylesheets because:
    # 1. We only need the TEXT, not how it looks
    # 2. Blocking media makes scraping 2-5x faster
    # 3. Saves bandwidth (polite scraping!)
    if config.block_resources:
        def block_unnecessary_resources(route: Route):
            """Intercept and abort requests for non-text resources.

            route.abort() kills the request.
            route.continue_() lets it through normally.

            This callback runs for EVERY network request the page makes.
            """
            resource_type = route.request.resource_type
            if resource_type in ("image", "font", "stylesheet", "media"):
                route.abort()
            else:
                route.continue_()

        # page.route() registers our interceptor.
        # "**/*" means "match ALL URLs" (glob pattern).
        page.route("**/*", block_unnecessary_resources)

    results: list[ScrapedQA] = []
    try:
        # ── CONCEPT: Navigation with wait_until ──
        # "networkidle" waits until there are no network requests
        # for 500ms. This is the MOST THOROUGH wait strategy:
        #   - "commit"            — response headers received
        #   - "domcontentloaded"  — HTML parsed, DOM ready
        #   - "load"             — all resources loaded
        #   - "networkidle"      — no requests for 500ms
        #
        # For scraping JS-heavy sites, networkidle is best because
        # it waits for AJAX calls to complete.
        page.goto(url, wait_until="networkidle")

        # Try scraping with the configured selectors
        results = _extract_qa_from_page(page, url, config)

        # If nothing found with configured selectors, try auto-detection
        if not results:
            results = _auto_detect_qa(page, url, config)

    except Exception as e:
        print(f"  Error scraping {url}: {e}")

    finally:
        # Always clean up — close context and browser.
        # This is important to prevent zombie browser processes!
        context.close()
        browser.close()

    return results


def _extract_qa_from_page(
    page: Page,
    url: str,
    config: ScrapeConfig,
) -> list[ScrapedQA]:
    """Extract Q&A pairs using configured CSS selectors.

    === PLAYWRIGHT CONCEPT: page.query_selector_all() ===

    Returns a list of ElementHandle objects matching the CSS selector.
    Each ElementHandle represents a DOM element you can interact with.

    Similar to document.querySelectorAll() in JavaScript.

    NOTE: query_selector_all() is a lower-level API. For TESTS,
    prefer locator() which has auto-waiting. For scraping (where the
    page is already loaded), query_selector_all() is fine and simpler.
    """
    results: list[ScrapedQA] = []

    containers = page.query_selector_all(config.qa_container_selector)
    if not containers:
        return results

    for container in containers:
        # Find question and answer within this container
        q_el = container.query_selector(config.question_selector)
        a_el = container.query_selector(config.answer_selector)

        if q_el and a_el:
            question = (q_el.inner_text() or "").strip()
            answer = (a_el.inner_text() or "").strip()

            if question and answer:
                results.append(ScrapedQA(
                    question=question,
                    answer=answer,
                    source_url=url,
                    tags=config.tags.copy(),
                ))

    return results


def _auto_detect_qa(
    page: Page,
    url: str,
    config: ScrapeConfig,
) -> list[ScrapedQA]:
    """Try to auto-detect Q&A content using common patterns.

    === PLAYWRIGHT CONCEPT: page.evaluate() ===

    page.evaluate() runs JavaScript CODE inside the browser and returns
    the result to Python. The JavaScript runs in the actual page context,
    so it can access the DOM, window, document, etc.

    This is POWERFUL because:
    1. Complex DOM traversal is easier in JS than with query_selector chains
    2. You can access computed styles, event handlers, etc.
    3. You can run any JS library already loaded on the page

    The return value is automatically serialized (JSON-compatible types).
    """
    results: list[ScrapedQA] = []

    # Strategy 1: Look for <details><summary> patterns (HTML5 FAQ pattern)
    details_qa = page.evaluate("""
        () => {
            const pairs = [];
            document.querySelectorAll('details').forEach(detail => {
                const summary = detail.querySelector('summary');
                // Get answer text excluding the summary
                const answerParts = [];
                detail.childNodes.forEach(node => {
                    if (node !== summary && node.textContent.trim()) {
                        answerParts.push(node.textContent.trim());
                    }
                });
                const answer = answerParts.join(' ').trim();
                if (summary && answer) {
                    pairs.push({
                        question: summary.innerText.trim(),
                        answer: answer
                    });
                }
            });
            return pairs;
        }
    """)
    for qa in details_qa:
        if qa["question"] and qa["answer"]:
            results.append(ScrapedQA(
                question=qa["question"],
                answer=qa["answer"],
                source_url=url,
                tags=config.tags.copy(),
            ))

    if results:
        return results

    # Strategy 2: Look for <dt>/<dd> pairs (definition lists)
    dt_dd_qa = page.evaluate("""
        () => {
            const pairs = [];
            const dts = document.querySelectorAll('dt');
            dts.forEach(dt => {
                const dd = dt.nextElementSibling;
                if (dd && dd.tagName === 'DD') {
                    pairs.push({
                        question: dt.innerText.trim(),
                        answer: dd.innerText.trim()
                    });
                }
            });
            return pairs;
        }
    """)
    for qa in dt_dd_qa:
        if qa["question"] and qa["answer"]:
            results.append(ScrapedQA(
                question=qa["question"],
                answer=qa["answer"],
                source_url=url,
                tags=config.tags.copy(),
            ))

    if results:
        return results

    # Strategy 3: Look for heading + next-sibling-paragraph patterns
    heading_qa = page.evaluate("""
        () => {
            const pairs = [];
            const headings = document.querySelectorAll('h2, h3, h4');
            headings.forEach(heading => {
                const text = heading.innerText.trim();
                // Only consider headings that look like questions
                if (text.includes('?') || text.toLowerCase().startsWith('what') ||
                    text.toLowerCase().startsWith('how') || text.toLowerCase().startsWith('why') ||
                    text.toLowerCase().startsWith('when') || text.toLowerCase().startsWith('where') ||
                    text.toLowerCase().startsWith('who') || text.toLowerCase().startsWith('can') ||
                    text.toLowerCase().startsWith('is') || text.toLowerCase().startsWith('do')) {

                    // Get the next paragraph(s) as the answer
                    let answer = '';
                    let sibling = heading.nextElementSibling;
                    while (sibling && !['H1','H2','H3','H4','H5','H6'].includes(sibling.tagName)) {
                        if (sibling.tagName === 'P' || sibling.tagName === 'DIV' ||
                            sibling.tagName === 'UL' || sibling.tagName === 'OL') {
                            answer += sibling.innerText.trim() + ' ';
                        }
                        sibling = sibling.nextElementSibling;
                    }
                    answer = answer.trim();
                    if (answer) {
                        pairs.push({ question: text, answer: answer });
                    }
                }
            });
            return pairs;
        }
    """)
    for qa in heading_qa:
        if qa["question"] and qa["answer"]:
            results.append(ScrapedQA(
                question=qa["question"],
                answer=qa["answer"],
                source_url=url,
                tags=config.tags.copy(),
            ))

    return results


def save_scraped_yaml(
    qa_pairs: list[ScrapedQA],
    output_path: str | Path,
    model_provider: str = "ollama",
    model_name: str = "llama3",
    evaluator: str = "contains",
) -> None:
    """Save scraped Q&A pairs as a YAML eval suite file.

    Uses the 'contains' evaluator by default because scraped answers
    are often verbose — we just want to check if the key info is present.
    """
    data = {
        "model": {
            "provider": model_provider,
            "name": model_name,
        },
        "evaluator": evaluator,
        "test_cases": [
            {
                "prompt": qa.question,
                "expected": qa.answer[:200],  # Truncate long answers
                "tags": qa.tags + [_domain_from_url(qa.source_url)],
            }
            for qa in qa_pairs
        ],
    }

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _domain_from_url(url: str) -> str:
    """Extract domain name from URL for use as a tag."""
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else "web"
