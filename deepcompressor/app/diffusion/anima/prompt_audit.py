"""Read-only structural and lexical-diversity audit for calibration prompts."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import typer
import yaml

app = typer.Typer(
    name="deepcompressor-audit-prompts",
    help="Audit a hand-authored prompt corpus without generating or rewriting text.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

ID_PATTERN = re.compile(r"^c(?P<category>\d{3})-p(?P<prompt>\d{3})$")
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[’'][^\W_]+)?", re.UNICODE)
SCORE_TAG_PATTERN = re.compile(r"(?i)(?:^|[\s,])score_[0-9]+(?:_up)?(?:$|[\s,])")


@dataclass(frozen=True)
class SimilarPair:
    left: str
    right: str
    similarity: float
    shared_trigrams: int
    left_trigrams: int
    right_trigrams: int


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(text.casefold()))


def _trigrams(tokens: tuple[str, ...]) -> frozenset[tuple[str, str, str]]:
    return frozenset(zip(tokens, tokens[1:], tokens[2:], strict=False))


def load_prompts(path: Path) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping, got {type(payload).__name__}")
    prompts: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"Every prompt ID and value must be text; invalid entry: {key!r}")
        prompts[key] = value.strip()
    return prompts


def _lexical_similarity(
    prompt_ids: list[str],
    trigram_sets: list[frozenset[tuple[str, str, str]]],
    threshold: float,
    report_limit: int,
) -> tuple[SimilarPair | None, list[SimilarPair]]:
    postings: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    highest: SimilarPair | None = None
    violations: list[SimilarPair] = []
    for right_index, right_set in enumerate(trigram_sets):
        intersections: Counter[int] = Counter()
        for trigram in right_set:
            intersections.update(postings[trigram])
        for left_index, shared in intersections.items():
            left_set = trigram_sets[left_index]
            union = len(left_set) + len(right_set) - shared
            similarity = shared / union if union else 1.0
            pair = SimilarPair(
                left=prompt_ids[left_index],
                right=prompt_ids[right_index],
                similarity=similarity,
                shared_trigrams=shared,
                left_trigrams=len(left_set),
                right_trigrams=len(right_set),
            )
            if highest is None or pair.similarity > highest.similarity:
                highest = pair
            if similarity >= threshold and len(violations) < report_limit:
                violations.append(pair)
        for trigram in right_set:
            postings[trigram].append(right_index)
    violations.sort(key=lambda pair: pair.similarity, reverse=True)
    return highest, violations


def audit_prompts(
    path: Path,
    *,
    expected_count: int,
    prompts_per_category: int,
    max_similarity: float,
    min_tokens: int,
    prefix_words: int,
    max_prefix_repetitions: int,
    report_limit: int,
) -> dict:
    prompts = load_prompts(path)
    errors: list[str] = []
    if len(prompts) != expected_count:
        errors.append(f"expected {expected_count} prompts, found {len(prompts)}")

    categories: Counter[int] = Counter()
    prompt_ids: list[str] = []
    prompt_tokens: list[tuple[str, ...]] = []
    trigram_sets: list[frozenset[tuple[str, str, str]]] = []
    normalized_text: dict[tuple[str, ...], str] = {}
    prefixes: dict[tuple[str, ...], list[str]] = defaultdict(list)
    score_tag_ids: list[str] = []

    for prompt_id, prompt in prompts.items():
        match = ID_PATTERN.fullmatch(prompt_id)
        if match is None:
            errors.append(f"malformed prompt ID: {prompt_id}")
        else:
            category = int(match.group("category"))
            prompt_number = int(match.group("prompt"))
            categories[category] += 1
            if not 1 <= prompt_number <= prompts_per_category:
                errors.append(f"prompt number outside 1..{prompts_per_category}: {prompt_id}")

        tokens = _tokens(prompt)
        if len(tokens) < min_tokens:
            errors.append(f"{prompt_id} has only {len(tokens)} normalized tokens; minimum is {min_tokens}")
        prior = normalized_text.get(tokens)
        if prior is not None:
            errors.append(f"normalized duplicate text: {prior} and {prompt_id}")
        else:
            normalized_text[tokens] = prompt_id
        prefixes[tokens[:prefix_words]].append(prompt_id)
        if SCORE_TAG_PATTERN.search(prompt):
            score_tag_ids.append(prompt_id)
        prompt_ids.append(prompt_id)
        prompt_tokens.append(tokens)
        trigram_sets.append(_trigrams(tokens))

    expected_categories = expected_count // prompts_per_category
    expected_ids = {
        f"c{category:03d}-p{prompt:03d}"
        for category in range(1, expected_categories + 1)
        for prompt in range(1, prompts_per_category + 1)
    }
    missing_ids = sorted(expected_ids - prompts.keys())
    unexpected_ids = sorted(prompts.keys() - expected_ids)
    if missing_ids:
        errors.append(f"missing {len(missing_ids)} expected IDs; first: {missing_ids[:report_limit]}")
    if unexpected_ids:
        errors.append(f"found {len(unexpected_ids)} unexpected IDs; first: {unexpected_ids[:report_limit]}")
    wrong_category_counts = {
        category: count for category, count in sorted(categories.items()) if count != prompts_per_category
    }
    if wrong_category_counts:
        errors.append(f"categories without exactly {prompts_per_category} prompts: {wrong_category_counts}")
    repeated_prefixes = [
        {"prefix": " ".join(prefix), "count": len(ids), "ids": ids[:report_limit]}
        for prefix, ids in prefixes.items()
        if prefix and len(ids) > max_prefix_repetitions
    ]
    repeated_prefixes.sort(key=lambda item: item["count"], reverse=True)
    if repeated_prefixes:
        errors.append(
            f"found {len(repeated_prefixes)} {prefix_words}-word prefixes repeated more than "
            f"{max_prefix_repetitions} times"
        )
    if score_tag_ids:
        errors.append(f"score tags are not valid for Anima Aesthetic: {score_tag_ids[:report_limit]}")

    highest, similar_pairs = _lexical_similarity(prompt_ids, trigram_sets, max_similarity, report_limit)
    if similar_pairs:
        errors.append(f"found prompt pairs with word-trigram Jaccard similarity >= {max_similarity:.3f}")

    return {
        "path": str(path),
        "prompt_count": len(prompts),
        "category_count": len(categories),
        "prompts_per_category": prompts_per_category,
        "metric": "normalized word-trigram Jaccard",
        "maximum_allowed_similarity_exclusive": max_similarity,
        "highest_similarity_pair": asdict(highest) if highest is not None else None,
        "similarity_violations": [asdict(pair) for pair in similar_pairs],
        "repeated_prefixes": repeated_prefixes[:report_limit],
        "score_tag_ids": score_tag_ids[:report_limit],
        "errors": errors,
        "passed": not errors,
    }


@app.command("audit")
def audit_cli(
    prompts: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, resolve_path=True)],
    expected_count: Annotated[int, typer.Option(min=1)] = 10_000,
    prompts_per_category: Annotated[int, typer.Option(min=1)] = 100,
    max_similarity: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.10,
    min_tokens: Annotated[int, typer.Option(min=3)] = 8,
    prefix_words: Annotated[int, typer.Option(min=1)] = 4,
    max_prefix_repetitions: Annotated[int, typer.Option(min=1)] = 5,
    report_limit: Annotated[int, typer.Option(min=1)] = 20,
) -> None:
    """Audit one immutable, hand-authored YAML prompt mapping."""
    report = audit_prompts(
        prompts,
        expected_count=expected_count,
        prompts_per_category=prompts_per_category,
        max_similarity=max_similarity,
        min_tokens=min_tokens,
        prefix_words=prefix_words,
        max_prefix_repetitions=max_prefix_repetitions,
        report_limit=report_limit,
    )
    typer.echo(json.dumps(report, indent=2))
    if not report["passed"]:
        raise typer.Exit(2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
