"""
Temporary automatic QA test registry for DocuBot.

Purpose:
- Match known QA questions automatically.
- Supply test case ID, description, expected result, and run count.
- Evaluate answers without requiring manual sidebar input.

Removal:
- Delete this file together with the temporary QA logging files.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Tuple


NO_RESULT_MESSAGE = (
    "Information not found in company knowledge base."
)


def _normalize_text(
    text: str
) -> str:

    if not text:

        return ""

    value = unicodedata.normalize(
        "NFKD",
        str(text)
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(
            character
        )
    )

    value = value.lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def _case(
    test_case_id: str,
    question: str,
    description: str,
    expected_result: str,
    aliases: Iterable[str] = (),
    required_groups: Iterable[
        Iterable[str]
    ] = (),
    forbidden_terms: Iterable[str] = (),
    minimum_pool_matches: int = 0,
    pool_terms: Iterable[str] = (),
    minimum_bullets: int = 0,
    exact_answer: str = "",
    answer_must_start_with: Iterable[str] = (),
    total_runs: int = 4
) -> Dict[str, Any]:

    return {
        "test_case_id":
            test_case_id,

        "category":
            "Answer Quality Regression",

        "question":
            question,

        "aliases":
            list(
                aliases
            ),

        "description":
            description,

        "expected_result":
            expected_result,

        "required_groups": [
            list(
                group
            )
            for group in required_groups
        ],

        "forbidden_terms":
            list(
                forbidden_terms
            ),

        "minimum_pool_matches":
            int(
                minimum_pool_matches
            ),

        "pool_terms":
            list(
                pool_terms
            ),

        "minimum_bullets":
            int(
                minimum_bullets
            ),

        "exact_answer":
            exact_answer,

        "answer_must_start_with":
            list(
                answer_must_start_with
            ),

        "total_runs":
            int(
                total_runs
            ),
    }


TEST_CASES: List[Dict[str, Any]] = [
    _case(
        test_case_id="Q-001",
        question="Who is Jose Rizal?",
        description=(
            "Verify that DocuBot identifies Jose Rizal and "
            "returns a supported concise overview."
        ),
        expected_result=(
            "The answer identifies Jose Rizal and includes a supported "
            "description such as Filipino nationalist, writer, polymath, "
            "or national hero."
        ),
        required_groups=[
            [
                "Jose Rizal",
                "José Rizal",
            ],
            [
                "Filipino nationalist",
                "writer",
                "polymath",
            ],
        ],
    ),
    _case(
        test_case_id="Q-002",
        question="When is his birthday?",
        description=(
            "Verify pronoun resolution and the correct birth date "
            "for the current Jose Rizal topic."
        ),
        expected_result=(
            "The answer returns June 19, 1861."
        ),
        required_groups=[
            [
                "June 19, 1861",
                "19 June 1861",
            ],
        ],
    ),
    _case(
        test_case_id="Q-003",
        question="When did Rizal die?",
        description=(
            "Verify the correct death date for Jose Rizal."
        ),
        expected_result=(
            "The answer returns December 30, 1896."
        ),
        aliases=[
            "When did rizal die?",
        ],
        required_groups=[
            [
                "December 30, 1896",
                "30 December 1896",
            ],
        ],
    ),
    _case(
        test_case_id="Q-004",
        question=(
            "Who are the ladies that had relationship with Jose Rizal?"
        ),
        description=(
            "Verify complete multi-item relationship retrieval "
            "and bullet formatting."
        ),
        expected_result=(
            "The answer returns the explicitly supported women connected "
            "to Jose Rizal and uses a vertical list."
        ),
        aliases=[
            "Who are the ladies that had relationship with Jose rizal?",
        ],
        required_groups=[
            [
                "Segunda Katigbak",
            ],
            [
                "Leonor Rivera",
            ],
            [
                "Josephine Bracken",
            ],
            [
                "Gertrude Beckett",
            ],
            [
                "Nelly Boustead",
            ],
            [
                "Seiko Usui",
                "O-Sei-San",
                "O Sei San",
            ],
        ],
        minimum_bullets=5,
    ),
    _case(
        test_case_id="Q-005",
        question="Who killed Ferdinand Magellan?",
        description=(
            "Verify exact entity selection for Ferdinand Magellan's death."
        ),
        expected_result=(
            "The answer identifies Lapu-Lapu or Lapulapu and his warriors."
        ),
        required_groups=[
            [
                "Lapu-Lapu",
                "Lapulapu",
                "Lapu Lapu",
            ],
        ],
    ),
    _case(
        test_case_id="Q-006",
        question="Did Lapu Lapu kill Magellan?",
        description=(
            "Verify yes-or-no answer handling for an imperfectly "
            "worded question."
        ),
        expected_result=(
            "The answer begins with Yes and connects Lapu-Lapu "
            "to Magellan's death."
        ),
        aliases=[
            "Did Lapu lapu killed Magellan?",
            "Did Lapu Lapu killed Magellan?",
        ],
        required_groups=[
            [
                "Lapu-Lapu",
                "Lapulapu",
                "Lapu Lapu",
            ],
            [
                "Magellan",
            ],
        ],
        answer_must_start_with=[
            "Yes",
        ],
    ),
    _case(
        test_case_id="Q-007",
        question="Who is the first Philippine president?",
        description=(
            "Verify exact person selection for the first "
            "Philippine president."
        ),
        expected_result=(
            "The answer identifies Emilio Aguinaldo."
        ),
        required_groups=[
            [
                "Emilio Aguinaldo",
            ],
        ],
    ),
    _case(
        test_case_id="Q-008",
        question="Can you explain the Treaty of Paris in detail?",
        description=(
            "Verify a detailed grounded overview of the Treaty of Paris."
        ),
        expected_result=(
            "The answer explains the 1898 Treaty of Paris, the end of the "
            "Spanish-American War, and the transfer of the Philippines "
            "from Spain to the United States."
        ),
        aliases=[
            "Can you explain the treaty of paris in detail?",
        ],
        required_groups=[
            [
                "Treaty of Paris",
            ],
            [
                "1898",
            ],
            [
                "Spain",
            ],
            [
                "United States",
                "U.S.",
                "US",
            ],
            [
                "Philippines",
            ],
        ],
    ),
    _case(
        test_case_id="Q-009",
        question="What did it do?",
        description=(
            "Verify follow-up resolution to the Treaty of Paris."
        ),
        expected_result=(
            "The answer states what the Treaty of Paris did, including "
            "Spain's transfer or cession of the Philippines to the "
            "United States."
        ),
        aliases=[
            "What it did?",
        ],
        required_groups=[
            [
                "Spain",
            ],
            [
                "Philippines",
            ],
            [
                "United States",
                "U.S.",
                "US",
            ],
            [
                "ceded",
                "cession",
                "transferred",
                "relinquished",
            ],
        ],
    ),
    _case(
        test_case_id="Q-010",
        question=(
            "Why did Jose Rizal become the Supremo of the Katipunan?"
        ),
        description=(
            "Verify false-premise handling and prevention of unsupported "
            "reason generation."
        ),
        expected_result=(
            "The exact fallback message is returned because the assumed "
            "role is not explicitly supported."
        ),
        exact_answer=NO_RESULT_MESSAGE,
    ),
    _case(
        test_case_id="Q-011",
        question=(
            "What hardships did Filipinos experience during the "
            "Japanese occupation?"
        ),
        description=(
            "Verify complete multi-item hardship retrieval and list format."
        ),
        expected_result=(
            "The answer lists multiple explicitly supported hardships such "
            "as shortages, inflation, forced labor, violence, disease, "
            "malnutrition, torture, or poor living conditions."
        ),
        pool_terms=[
            "food shortage",
            "food shortages",
            "fuel shortage",
            "fuel shortages",
            "inflation",
            "forced labor",
            "torture",
            "murder",
            "malnutrition",
            "disease",
            "poor living conditions",
            "rationing",
            "sexual slavery",
            "death march",
        ],
        minimum_pool_matches=4,
        minimum_bullets=4,
    ),
    _case(
        test_case_id="Q-012",
        question=(
            "Who founded the Katipunan or KKK on July 7, 1892, "
            "and what was its purpose against Spain?"
        ),
        description=(
            "Verify compound-question completeness for founders and purpose."
        ),
        expected_result=(
            "The answer identifies the founders and explains that the "
            "Katipunan sought Philippine independence from Spain through "
            "revolution or armed struggle."
        ),
        required_groups=[
            [
                "Andres Bonifacio",
                "Andrés Bonifacio",
            ],
            [
                "Deodato Arellano",
            ],
            [
                "Ladislao Diwa",
            ],
            [
                "Spain",
                "Spanish",
            ],
            [
                "independence",
                "revolution",
                "armed",
            ],
        ],
    ),
    _case(
        test_case_id="Q-013",
        question=(
            "Which secret group tried to free Filipinos from Spanish rule "
            "through armed revolution before it was discovered in 1896?"
        ),
        description=(
            "Verify exact entity selection from all constraints."
        ),
        expected_result=(
            "The answer identifies the Katipunan or KKK and must not "
            "identify La Liga Filipina."
        ),
        required_groups=[
            [
                "Katipunan",
                "KKK",
            ],
        ],
        forbidden_terms=[
            "La Liga Filipina",
        ],
    ),
    _case(
        test_case_id="Q-014",
        question=(
            "How did the Treaty of Paris connect the Spanish-American War "
            "to the Philippine-American War?"
        ),
        description=(
            "Verify grounded causal connection across two historical events."
        ),
        expected_result=(
            "The answer explains that the treaty ended the Spanish-American "
            "War, transferred the Philippines to the United States, and "
            "contributed to the Philippine-American War."
        ),
        required_groups=[
            [
                "Treaty of Paris",
            ],
            [
                "Spanish-American War",
            ],
            [
                "Philippine-American War",
            ],
            [
                "Philippines",
            ],
            [
                "United States",
                "U.S.",
                "US",
            ],
        ],
    ),
    _case(
        test_case_id="Q-015",
        question=(
            "What is the difference between the Philippine Revolution "
            "and the Katipunan? Is one an organization and the other "
            "a war/revolution?"
        ),
        description=(
            "Verify comparison of an organization and a revolution."
        ),
        expected_result=(
            "The answer identifies the Katipunan as an organization or "
            "secret society and the Philippine Revolution as a war, "
            "uprising, or revolution."
        ),
        required_groups=[
            [
                "Katipunan",
            ],
            [
                "organization",
                "secret society",
            ],
            [
                "Philippine Revolution",
            ],
            [
                "war",
                "revolution",
                "uprising",
            ],
        ],
    ),
    _case(
        test_case_id="Q-016",
        question=(
            "Kailan ipinagdiriwang ang Araw ng Kalayaan ng Pilipinas "
            "at anong pangyayari ang ginugunita nito?"
        ),
        description=(
            "Verify Tagalog compound answer for Philippine Independence Day."
        ),
        expected_result=(
            "The answer states June 12 and the 1898 declaration of "
            "Philippine independence."
        ),
        required_groups=[
            [
                "June 12",
                "Hunyo 12",
            ],
            [
                "1898",
            ],
            [
                "independence",
                "kalayaan",
            ],
        ],
    ),
    _case(
        test_case_id="Q-017",
        question="When is Philippines Independence Day?",
        description=(
            "Verify the annual date of Philippine Independence Day."
        ),
        expected_result=(
            "The answer returns June 12."
        ),
        aliases=[
            "When is Philippine Independence Day?",
            "When is Philippines independence day?",
        ],
        required_groups=[
            [
                "June 12",
                "12 June",
            ],
        ],
    ),
]


_CASE_LOOKUP: Dict[
    str,
    Dict[str, Any]
] = {}


for _test_case in TEST_CASES:

    variants = [
        _test_case["question"],
        *_test_case.get(
            "aliases",
            []
        ),
    ]

    for _variant in variants:

        _CASE_LOOKUP[
            _normalize_text(
                _variant
            )
        ] = _test_case


def get_test_case(
    question: str
) -> Dict[str, Any]:

    """
    Match a known question automatically.

    Unknown questions are still logged automatically
    as ad-hoc evidence with MANUAL CHECK status.
    """

    normalized_question = _normalize_text(
        question
    )

    matched = _CASE_LOOKUP.get(
        normalized_question
    )

    if matched:

        result = dict(
            matched
        )

        result["matched"] = True
        result["normalized_question"] = (
            normalized_question
        )

        return result

    digest = hashlib.sha1(
        normalized_question.encode(
            "utf-8"
        )
    ).hexdigest()[:8].upper()

    return {
        "test_case_id":
            f"AUTO-{digest}",

        "category":
            "Ad Hoc Question Evidence",

        "question":
            question,

        "description":
            "Automatically capture evidence for an unregistered question.",

        "expected_result":
            (
                "No predefined expected answer. "
                "Technical evidence is recorded for review."
            ),

        "required_groups":
            [],

        "forbidden_terms":
            [],

        "minimum_pool_matches":
            0,

        "pool_terms":
            [],

        "minimum_bullets":
            0,

        "exact_answer":
            "",

        "answer_must_start_with":
            [],

        "total_runs":
            1,

        "matched":
            False,

        "normalized_question":
            normalized_question,
    }


def get_next_run_number(
    question: str,
    counters: Dict[str, int]
) -> Tuple[int, int, int]:

    """
    Increment the automatic run counter.

    Returns:
        run_number,
        total_runs,
        cycle_number
    """

    test_case = get_test_case(
        question
    )

    key = test_case[
        "normalized_question"
    ]

    previous_count = int(
        counters.get(
            key,
            0
        )
    )

    current_count = (
        previous_count
        + 1
    )

    counters[
        key
    ] = current_count

    total_runs = max(
        1,
        int(
            test_case.get(
                "total_runs",
                1
            )
        )
    )

    run_number = (
        (
            current_count
            - 1
        )
        % total_runs
    ) + 1

    cycle_number = (
        (
            current_count
            - 1
        )
        // total_runs
    ) + 1

    return (
        run_number,
        total_runs,
        cycle_number
    )


def _contains_any(
    normalized_answer: str,
    alternatives: Iterable[str]
) -> bool:

    return any(
        _normalize_text(
            alternative
        )
        in normalized_answer
        for alternative in alternatives
        if str(
            alternative
        ).strip()
    )


def _count_markdown_bullets(
    answer: str
) -> int:

    return len(
        re.findall(
            r"(?m)^\s*[-*]\s+\S",
            answer or ""
        )
    )


def evaluate_answer(
    answer: str,
    test_case: Dict[str, Any]
) -> Tuple[str, str]:

    """
    Evaluate a known QA answer automatically.

    Unknown questions remain MANUAL CHECK, but their
    technical evidence is still fully recorded.
    """

    if not test_case.get(
        "matched",
        False
    ):

        return (
            "MANUAL CHECK",
            (
                "Question was not found in the automatic QA registry. "
                "Evidence was recorded without guessing correctness."
            )
        )

    answer_text = (
        answer
        or ""
    ).strip()

    normalized_answer = _normalize_text(
        answer_text
    )

    exact_answer = test_case.get(
        "exact_answer",
        ""
    ).strip()

    if exact_answer:

        if (
            normalized_answer
            == _normalize_text(
                exact_answer
            )
        ):

            return (
                "PASS",
                "The answer exactly matches the expected fallback."
            )

        return (
            "FAIL",
            (
                "Expected exact answer: "
                f"{exact_answer}"
            )
        )

    failures: List[str] = []

    for group in test_case.get(
        "required_groups",
        []
    ):

        if not _contains_any(
            normalized_answer,
            group
        ):

            failures.append(
                "Missing required fact group: "
                + " OR ".join(
                    group
                )
            )

    forbidden_hits = [
        term
        for term in test_case.get(
            "forbidden_terms",
            []
        )
        if _normalize_text(
            term
        ) in normalized_answer
    ]

    if forbidden_hits:

        failures.append(
            "Forbidden term(s) found: "
            + ", ".join(
                forbidden_hits
            )
        )

    pool_terms = test_case.get(
        "pool_terms",
        []
    )

    minimum_pool_matches = int(
        test_case.get(
            "minimum_pool_matches",
            0
        )
    )

    if minimum_pool_matches > 0:

        matched_pool_terms = {
            term
            for term in pool_terms
            if _normalize_text(
                term
            ) in normalized_answer
        }

        if (
            len(
                matched_pool_terms
            )
            < minimum_pool_matches
        ):

            failures.append(
                (
                    "Only "
                    f"{len(matched_pool_terms)} "
                    "expected content indicators were found; "
                    f"{minimum_pool_matches} required."
                )
            )

    minimum_bullets = int(
        test_case.get(
            "minimum_bullets",
            0
        )
    )

    if minimum_bullets > 0:

        bullet_count = _count_markdown_bullets(
            answer_text
        )

        if bullet_count < minimum_bullets:

            failures.append(
                (
                    f"Only {bullet_count} Markdown bullet(s) found; "
                    f"{minimum_bullets} required."
                )
            )

    starts_with = test_case.get(
        "answer_must_start_with",
        []
    )

    if starts_with:

        if not any(
            normalized_answer.startswith(
                _normalize_text(
                    prefix
                )
            )
            for prefix in starts_with
        ):

            failures.append(
                "Answer must start with: "
                + " OR ".join(
                    starts_with
                )
            )

    if failures:

        return (
            "FAIL",
            "\n".join(
                failures
            )
        )

    return (
        "PASS",
        (
            "Automatic expected-result checks passed for "
            f"{test_case['test_case_id']}."
        )
    )
