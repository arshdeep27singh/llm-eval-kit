"""
Playwright tests for the web scraper.

=== HOW DO WE TEST A SCRAPER WITHOUT HITTING REAL WEBSITES? ===

We create LOCAL HTML files that mimic different FAQ page structures,
then scrape those files using the file:// protocol. This approach:
    1. Makes tests fast (no network)
    2. Makes tests reliable (no flaky external sites)
    3. Gives us full control over the HTML structures we test against

=== NEW PLAYWRIGHT CONCEPTS IN THIS FILE ===

- page.set_content()           — Load HTML directly without a file
- page.route() for mocking     — Intercept requests and return fake responses
- page.wait_for_load_state()   — Wait for specific page states
- page.url                     — Get the current page URL
- page.content()               — Get the full HTML of the page
- expect(page).to_have_url()   — Assert the current URL
- browser.new_context()        — Create isolated browser contexts
- context.new_page()           — Create new tabs
"""

from pathlib import Path

import pytest
import yaml

from llm_eval_kit.scraper import ScrapeConfig, ScrapedQA, scrape_qa_pairs, save_scraped_yaml


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES — Create different HTML page structures for testing
# ══════════════════════════════════════════════════════════════════════════════

# NOTE: We use the `playwright` fixture from pytest-playwright.
# When running inside pytest-playwright's event loop, we can't call
# sync_playwright() again — so we pass the existing instance to our scraper
# via the `playwright_instance` parameter.
# This is a real-world pattern: make Playwright code testable by
# allowing dependency injection of the Playwright instance.


@pytest.fixture
def scrape(playwright):
    """Wrapper fixture that injects the pytest-playwright instance into scrape_qa_pairs.

    === PLAYWRIGHT CONCEPT: Dependency injection for testability ===

    pytest-playwright manages its own asyncio event loop. If our scraper
    calls sync_playwright() inside that loop, it crashes. The fix:
    accept an external Playwright instance and pass it through.

    This fixture returns a callable with the same API as scrape_qa_pairs,
    but with playwright_instance pre-filled.
    """
    def _scrape(url, config=None):
        return scrape_qa_pairs(url, config, playwright_instance=playwright)
    return _scrape

@pytest.fixture
def faq_html_details(tmp_path: Path) -> Path:
    """Create an HTML page with <details><summary> FAQ pattern.

    This is a common HTML5 pattern for FAQ pages — the browser
    natively supports expand/collapse without JavaScript!
    """
    html = """<!DOCTYPE html>
<html>
<head><title>FAQ - Details Pattern</title></head>
<body>
    <h1>Frequently Asked Questions</h1>

    <details>
        <summary>What is Python?</summary>
        <p>Python is a high-level, interpreted programming language known for its simple syntax.</p>
    </details>

    <details>
        <summary>What is JavaScript?</summary>
        <p>JavaScript is a programming language that runs in web browsers and on servers via Node.js.</p>
    </details>

    <details>
        <summary>What is Playwright?</summary>
        <p>Playwright is a browser automation library for end-to-end testing and web scraping.</p>
    </details>
</body>
</html>"""
    path = tmp_path / "faq_details.html"
    path.write_text(html)
    return path


@pytest.fixture
def faq_html_dl(tmp_path: Path) -> Path:
    """Create an HTML page with <dl><dt><dd> FAQ pattern.

    Definition lists (<dl>) are a semantic HTML pattern where:
    - <dt> = Definition Term (the question)
    - <dd> = Definition Description (the answer)
    """
    html = """<!DOCTYPE html>
<html>
<head><title>FAQ - Definition List</title></head>
<body>
    <h1>Help Center</h1>

    <dl>
        <dt>How do I reset my password?</dt>
        <dd>Go to Settings, click Security, then click Reset Password.</dd>

        <dt>How do I contact support?</dt>
        <dd>Email us at support@example.com or use the chat widget.</dd>
    </dl>
</body>
</html>"""
    path = tmp_path / "faq_dl.html"
    path.write_text(html)
    return path


@pytest.fixture
def faq_html_heading(tmp_path: Path) -> Path:
    """Create an HTML page with heading + paragraph FAQ pattern.

    Many FAQ pages just use headings (h2/h3) for questions
    followed by paragraphs for answers.
    """
    html = """<!DOCTYPE html>
<html>
<head><title>FAQ - Heading Pattern</title></head>
<body>
    <h1>Knowledge Base</h1>

    <h3>What is machine learning?</h3>
    <p>Machine learning is a subset of AI where systems learn from data.</p>

    <h3>How does neural network work?</h3>
    <p>Neural networks process data through layers of interconnected nodes.</p>
    <p>Each layer transforms the data to extract higher-level features.</p>

    <h3>Why is data important?</h3>
    <p>Data is the fuel that powers machine learning models.</p>
</body>
</html>"""
    path = tmp_path / "faq_heading.html"
    path.write_text(html)
    return path


@pytest.fixture
def faq_html_custom_class(tmp_path: Path) -> Path:
    """Create an HTML page with custom CSS classes for Q&A pairs.

    This represents a typical custom-styled FAQ page where the
    developer chose their own class names.
    """
    html = """<!DOCTYPE html>
<html>
<head><title>FAQ - Custom Classes</title></head>
<body>
    <h1>FAQ</h1>

    <div class="faq-item">
        <h3 class="question">What colors do you offer?</h3>
        <div class="answer">We offer red, blue, green, and black.</div>
    </div>

    <div class="faq-item">
        <h3 class="question">What is the return policy?</h3>
        <div class="answer">You can return items within 30 days.</div>
    </div>

    <div class="faq-item">
        <h3 class="question">Do you ship internationally?</h3>
        <div class="answer">Yes, we ship to over 50 countries worldwide.</div>
    </div>
</body>
</html>"""
    path = tmp_path / "faq_custom.html"
    path.write_text(html)
    return path


@pytest.fixture
def empty_html(tmp_path: Path) -> Path:
    """Create an HTML page with no Q&A content."""
    html = """<!DOCTYPE html>
<html>
<head><title>Empty Page</title></head>
<body>
    <h1>Nothing here</h1>
    <p>This page has no FAQ content.</p>
</body>
</html>"""
    path = tmp_path / "empty.html"
    path.write_text(html)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 1: Scraping with Custom CSS Selectors
# ══════════════════════════════════════════════════════════════════════════════

class TestCustomSelectorScraping:
    """Test scraping pages that use custom CSS class names.

    This tests the PRIMARY scraping path — when the user provides
    CSS selectors that match their page's structure.
    """

    def test_scrape_faq_items(self, faq_html_custom_class: Path, scrape):
        """Test scraping with the default .faq-item selector."""
        results = scrape(
            f"file://{faq_html_custom_class}",
            ScrapeConfig(
                qa_container_selector=".faq-item",
                question_selector=".question",
                answer_selector=".answer",
            ),
        )

        assert len(results) == 3
        assert results[0].question == "What colors do you offer?"
        assert "red, blue, green" in results[0].answer
        assert results[2].question == "Do you ship internationally?"

    def test_scrape_preserves_source_url(self, faq_html_custom_class: Path, scrape):
        """Each scraped result should record which URL it came from."""
        url = f"file://{faq_html_custom_class}"
        results = scrape(
            url,
            ScrapeConfig(
                qa_container_selector=".faq-item",
                question_selector=".question",
                answer_selector=".answer",
            ),
        )
        assert all(r.source_url == url for r in results)

    def test_scrape_with_tags(self, faq_html_custom_class: Path, scrape):
        """Tags from config should be attached to every result."""
        results = scrape(
            f"file://{faq_html_custom_class}",
            ScrapeConfig(
                qa_container_selector=".faq-item",
                question_selector=".question",
                answer_selector=".answer",
                tags=["shopping", "faq"],
            ),
        )
        for r in results:
            assert "shopping" in r.tags
            assert "faq" in r.tags


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 2: Auto-Detection Strategies
# ══════════════════════════════════════════════════════════════════════════════

class TestAutoDetection:
    """Test the auto-detection fallback strategies.

    When the configured CSS selectors don't find anything,
    the scraper tries common HTML patterns automatically.

    === WHY AUTO-DETECT? ===
    Users shouldn't need to inspect a page's HTML to use the scraper.
    We try several common patterns and return whatever works.
    """

    def test_auto_detect_details_pattern(self, faq_html_details: Path, scrape):
        """Auto-detect Q&A in <details><summary> elements."""
        results = scrape(
            f"file://{faq_html_details}",
            ScrapeConfig(
                qa_container_selector=".nonexistent",  # Force auto-detect
            ),
        )

        assert len(results) == 3
        assert results[0].question == "What is Python?"
        assert "high-level" in results[0].answer

    def test_auto_detect_dt_dd_pattern(self, faq_html_dl: Path, scrape):
        """Auto-detect Q&A in <dt>/<dd> definition list pairs."""
        results = scrape(
            f"file://{faq_html_dl}",
            ScrapeConfig(
                qa_container_selector=".nonexistent",
            ),
        )

        assert len(results) == 2
        assert "reset my password" in results[0].question.lower()
        assert "Settings" in results[0].answer

    def test_auto_detect_heading_pattern(self, faq_html_heading: Path, scrape):
        """Auto-detect Q&A in heading + paragraph patterns."""
        results = scrape(
            f"file://{faq_html_heading}",
            ScrapeConfig(
                qa_container_selector=".nonexistent",
            ),
        )

        assert len(results) >= 2  # Should find at least 2 question headings
        questions = [r.question for r in results]
        assert any("machine learning" in q.lower() for q in questions)

    def test_empty_page_returns_empty(self, empty_html: Path, scrape):
        """Pages with no recognizable Q&A should return empty list."""
        results = scrape(
            f"file://{empty_html}",
            ScrapeConfig(
                qa_container_selector=".nonexistent",
            ),
        )
        assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 3: Resource Blocking
# ══════════════════════════════════════════════════════════════════════════════

class TestResourceBlocking:
    """Test that resource blocking works correctly.

    === PLAYWRIGHT CONCEPT: Verifying intercepted requests ===

    We can't directly observe blocked requests from outside the browser,
    but we can verify the scraper still WORKS with blocking enabled
    (the default) and also without it.
    """

    def test_scraping_works_with_blocking(self, faq_html_custom_class: Path, scrape):
        """Scraping should work with resource blocking enabled (default)."""
        results = scrape(
            f"file://{faq_html_custom_class}",
            ScrapeConfig(
                qa_container_selector=".faq-item",
                question_selector=".question",
                answer_selector=".answer",
                block_resources=True,  # Default
            ),
        )
        assert len(results) == 3

    def test_scraping_works_without_blocking(self, faq_html_custom_class: Path, scrape):
        """Scraping should also work without resource blocking."""
        results = scrape(
            f"file://{faq_html_custom_class}",
            ScrapeConfig(
                qa_container_selector=".faq-item",
                question_selector=".question",
                answer_selector=".answer",
                block_resources=False,
            ),
        )
        assert len(results) == 3


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 4: YAML Output
# ══════════════════════════════════════════════════════════════════════════════

class TestYAMLOutput:
    """Test that scraped results are saved correctly as YAML eval suites."""

    def test_save_yaml_structure(self, tmp_path: Path):
        """The output YAML should have the correct structure for llm-eval-kit."""
        qa_pairs = [
            ScrapedQA(question="What is Python?", answer="A programming language", source_url="http://example.com", tags=["faq"]),
            ScrapedQA(question="What is JS?", answer="A web language", source_url="http://example.com", tags=["faq"]),
        ]

        output = tmp_path / "test_suite.yaml"
        save_scraped_yaml(qa_pairs, output)

        # Load and verify the YAML structure
        with open(output) as f:
            data = yaml.safe_load(f)

        assert "model" in data
        assert "evaluator" in data
        assert "test_cases" in data
        assert data["evaluator"] == "contains"  # Default
        assert len(data["test_cases"]) == 2

    def test_yaml_test_cases_have_required_fields(self, tmp_path: Path):
        """Each test case should have prompt, expected, and tags."""
        qa_pairs = [
            ScrapedQA(question="Q1?", answer="A1", source_url="http://example.com"),
        ]
        output = tmp_path / "test_suite.yaml"
        save_scraped_yaml(qa_pairs, output)

        with open(output) as f:
            data = yaml.safe_load(f)

        tc = data["test_cases"][0]
        assert tc["prompt"] == "Q1?"
        assert tc["expected"] == "A1"
        assert "tags" in tc
        # Domain tag should be auto-added
        assert "example.com" in tc["tags"]

    def test_yaml_truncates_long_answers(self, tmp_path: Path):
        """Answers longer than 200 chars should be truncated."""
        long_answer = "A" * 500
        qa_pairs = [
            ScrapedQA(question="Q?", answer=long_answer, source_url="http://example.com"),
        ]
        output = tmp_path / "test_suite.yaml"
        save_scraped_yaml(qa_pairs, output)

        with open(output) as f:
            data = yaml.safe_load(f)

        assert len(data["test_cases"][0]["expected"]) == 200

    def test_yaml_custom_evaluator(self, tmp_path: Path):
        """User can specify a different evaluator."""
        qa_pairs = [
            ScrapedQA(question="Q?", answer="A", source_url="http://example.com"),
        ]
        output = tmp_path / "test_suite.yaml"
        save_scraped_yaml(qa_pairs, output, evaluator="exact_match")

        with open(output) as f:
            data = yaml.safe_load(f)
        assert data["evaluator"] == "exact_match"


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASS 5: JavaScript-Rendered Content
# ══════════════════════════════════════════════════════════════════════════════

class TestJSRenderedContent:
    """Test scraping pages where content is rendered by JavaScript.

    === WHY THIS MATTERS ===
    This is the MAIN reason we use Playwright instead of httpx/BeautifulSoup.
    A regular HTTP GET would see an empty page — Playwright executes the JS
    and waits for the content to appear.
    """

    @pytest.fixture
    def js_rendered_faq(self, tmp_path: Path) -> Path:
        """Create an HTML page where FAQ content is added by JavaScript.

        The initial HTML has no FAQ content — JavaScript adds it after
        the page loads. Only a real browser (Playwright) can scrape this.
        """
        html = """<!DOCTYPE html>
<html>
<head><title>JS FAQ</title></head>
<body>
    <h1>Dynamic FAQ</h1>
    <div id="faq-container"></div>

    <script>
        // Simulate dynamic content loading (like a React/Vue app would)
        const faqs = [
            { q: "What is Docker?", a: "Docker is a containerization platform." },
            { q: "What is Kubernetes?", a: "Kubernetes orchestrates container deployments." },
        ];

        const container = document.getElementById('faq-container');
        faqs.forEach(faq => {
            const item = document.createElement('div');
            item.className = 'faq-item';
            item.innerHTML = `
                <h3 class="question">${faq.q}</h3>
                <div class="answer">${faq.a}</div>
            `;
            container.appendChild(item);
        });
    </script>
</body>
</html>"""
        path = tmp_path / "js_faq.html"
        path.write_text(html)
        return path

    def test_scrape_js_rendered_content(self, js_rendered_faq: Path, scrape):
        """
        === PLAYWRIGHT CONCEPT: JavaScript execution ===

        When page.goto() loads this page, Playwright's browser:
        1. Parses the HTML (faq-container is empty)
        2. Executes the <script> (JS adds .faq-item elements)
        3. Our scraper finds the dynamically-created elements

        A simple HTTP GET with httpx would see an EMPTY faq-container
        because httpx doesn't execute JavaScript.
        """
        results = scrape(
            f"file://{js_rendered_faq}",
            ScrapeConfig(
                qa_container_selector=".faq-item",
                question_selector=".question",
                answer_selector=".answer",
            ),
        )

        assert len(results) == 2
        assert results[0].question == "What is Docker?"
        assert "containerization" in results[0].answer
        assert results[1].question == "What is Kubernetes?"
