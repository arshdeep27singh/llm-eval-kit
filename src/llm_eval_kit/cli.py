"""
CLI entry point — the command users type in their terminal.

Uses Typer, which automatically converts Python function signatures
into CLI arguments. For example:

    def run(suite: Path)  →  llm-eval-kit run <suite>

Typer docs: https://typer.tiangolo.com/

This is the file referenced in pyproject.toml:
    [project.scripts]
    llm-eval-kit = "llm_eval_kit.cli:app"
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from llm_eval_kit import __version__
from llm_eval_kit.reporters.html_report import save_html_report
from llm_eval_kit.reporters.json_report import save_json_report
from llm_eval_kit.runner import load_suite, run_eval

# Create the Typer app — this is what gets called when user types "llm-eval-kit"
app = typer.Typer(
    name="llm-eval-kit",
    help="A lightweight tool for evaluating LLM outputs.",
    no_args_is_help=True,  # Show help if user just types "llm-eval-kit" with no command
)

# Rich console for pretty terminal output
console = Console()


@app.command()
def run(
    suite: Path = typer.Argument(
        ...,  # ... means required (not optional)
        help="Path to the YAML eval suite file.",
        exists=True,  # Typer checks the file exists before running
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Use a mock model (returns expected answers). No real LLM needed.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Save results to a JSON file (e.g., --output results.json).",
    ),
    html: Path | None = typer.Option(
        None,
        "--html",
        help="Save results as an interactive HTML report (e.g., --html report.html).",
    ),
):
    """Run an evaluation suite against an LLM.

    Examples:
        llm-eval-kit run examples/sample_eval.yaml
        llm-eval-kit run examples/sample_eval.yaml --dry-run
        llm-eval-kit run examples/sample_eval.yaml --output results.json
    """
    console.print(f"\n[bold blue]llm-eval-kit v{__version__}[/bold blue]\n")

    # Step 1: Load the YAML config
    console.print(f"Loading suite: [cyan]{suite}[/cyan]")
    config = load_suite(suite)

    if dry_run:
        console.print("[yellow]Dry-run mode:[/yellow] using mock model (returns expected answers)")

    console.print(
        f"Model: [green]{config.model_provider}/{config.model_name}[/green]  "
        f"Evaluator: [green]{config.evaluator}[/green]  "
        f"Test cases: [green]{len(config.test_cases)}[/green]\n"
    )

    # Step 2: Run all evaluations
    console.print("[bold]Running evaluations...[/bold]\n")
    report = run_eval(config, dry_run=dry_run)

    # Step 3: Display results as a table
    table = Table(title="\nEvaluation Results", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Prompt", style="cyan", max_width=50)
    table.add_column("Expected", style="green", max_width=20)
    table.add_column("Response", style="yellow", max_width=30)
    table.add_column("Score", justify="center", width=7)
    table.add_column("Reasoning", style="dim", max_width=40)

    for i, result in enumerate(report.results, start=1):
        # Color the score: green if >= 0.5, red if < 0.5
        score_str = f"[bold green]{result.score:.1f}[/bold green]" if result.score >= 0.5 else f"[bold red]{result.score:.1f}[/bold red]"
        table.add_row(
            str(i),
            result.test_case.prompt[:50],
            result.test_case.expected or "-",
            result.response[:30],
            score_str,
            (result.reasoning or "")[:40],
        )

    console.print(table)

    # Step 4: Print summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Model:    {report.model_name}")
    console.print(f"  Passed:   [green]{report.passed}[/green] / {report.total}")
    avg = report.average_score
    avg_color = "green" if avg >= 0.7 else "yellow" if avg >= 0.5 else "red"
    console.print(f"  Average:  [{avg_color}]{avg:.2f}[/{avg_color}]\n")

    # Step 5: Save JSON report if --output was provided
    if output:
        save_json_report(report, output)
        console.print(f"JSON report saved to: [cyan]{output}[/cyan]")

    # Step 6: Save HTML report if --html was provided
    if html:
        save_html_report(report, html)
        console.print(f"HTML report saved to: [cyan]{html}[/cyan]")

    console.print()


@app.command()
def version():
    """Show the version."""
    console.print(f"llm-eval-kit v{__version__}")


@app.command()
def scrape(
    url: str = typer.Argument(
        ...,
        help="URL to scrape Q&A pairs from (e.g., a FAQ page).",
    ),
    output: Path = typer.Option(
        "scraped_eval.yaml",
        "--output", "-o",
        help="Path to save the generated YAML eval suite.",
    ),
    selector: str = typer.Option(
        ".faq-item",
        "--selector", "-s",
        help="CSS selector for Q&A containers on the page.",
    ),
    tags: str = typer.Option(
        "",
        "--tags", "-t",
        help="Comma-separated tags to add to all test cases.",
    ),
    evaluator: str = typer.Option(
        "contains",
        "--evaluator", "-e",
        help="Evaluator to use in the generated suite (exact_match, contains, llm_judge).",
    ),
):
    """Scrape Q&A pairs from a web page and generate a YAML eval suite.

    Uses Playwright to render the page (handles JavaScript-heavy sites).

    Examples:
        llm-eval-kit scrape "https://example.com/faq" -o faq_eval.yaml
        llm-eval-kit scrape "https://example.com/faq" --selector "details" --tags "faq,web"
    """
    from llm_eval_kit.scraper import ScrapeConfig, scrape_qa_pairs, save_scraped_yaml

    console.print(f"\n[bold blue]llm-eval-kit v{__version__}[/bold blue] — Scraper\n")
    console.print(f"Scraping: [cyan]{url}[/cyan]")

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    config = ScrapeConfig(
        qa_container_selector=selector,
        tags=tag_list,
    )

    qa_pairs = scrape_qa_pairs(url, config)

    if not qa_pairs:
        console.print("[yellow]No Q&A pairs found.[/yellow] Try different CSS selectors with --selector.")
        raise typer.Exit(1)

    console.print(f"Found [green]{len(qa_pairs)}[/green] Q&A pairs\n")

    # Show a preview
    for i, qa in enumerate(qa_pairs[:5], 1):
        console.print(f"  [dim]{i}.[/dim] [cyan]{qa.question[:80]}[/cyan]")
        console.print(f"     → {qa.answer[:80]}...")
        console.print()

    if len(qa_pairs) > 5:
        console.print(f"  [dim]... and {len(qa_pairs) - 5} more[/dim]\n")

    save_scraped_yaml(qa_pairs, output, evaluator=evaluator)
    console.print(f"Saved eval suite to: [cyan]{output}[/cyan]\n")
