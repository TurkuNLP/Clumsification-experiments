# This script has been co-created, refactored, and cleaned using GPT 5.6.

"""
Multilingual rule-based perturbations backed by UniMorph.

Available templates
-------------------
jumble
    Swap two word tokens. When focus offsets are supplied, one swapped word
    must overlap the focus. Lightweight heuristics avoid merely reordering two
    members of the same comma-separated list.
subject_verb_dis
    Replace a finite verb with a conflicting form from the same lemma.
random_inflection
    Replace an inflectable word with another form of the same lemma and POS.
typos
    Introduce one Unicode-aware character-level typo.

Optional item fields
--------------------
focus_start, focus_end:
    Character offsets relative to item["text"].
focus_word_radius:
    For non-jumble rules, the number of neighboring words that may be changed.
    The MCQ builder supplies 1. Jumble always requires an overlapping focus
    word, regardless of this radius.

Items without focus metadata retain the original unrestricted behavior.

Supported languages: Finnish, Danish, Czech, German, Modern Greek, Italian.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from tqdm.auto import tqdm

try:
    from unimorph import Store as _UniMorphStore
    from unimorph import download as _unimorph_download

    _UNIMORPH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    _UniMorphStore = None  # type: ignore[assignment]
    _unimorph_download = None  # type: ignore[assignment]
    _UNIMORPH_IMPORT_ERROR = exc


RULE_BASED_MODEL_LABEL = "UniMorph-rule-based"
RULE_BASED_OUTPUT_DIR = "trad_perturbed_layers"

_LANGUAGE_ALIASES = {
    "en": "eng", "eng": "eng", "english": "eng",
    "fi": "fin", "fin": "fin", "finnish": "fin",
    "da": "dan", "dan": "dan", "danish": "dan",
    "cs": "ces", "cz": "ces", "ces": "ces", "cze": "ces", "czech": "ces",
    "de": "deu", "deu": "deu", "ger": "deu", "german": "deu",
    "el": "ell", "ell": "ell", "gre": "ell", "greek": "ell",
    "modern-greek": "ell", "modern_greek": "ell",
    "it": "ita", "ita": "ita", "italian": "ita",
}
SUPPORTED_UNIMORPH_LANGUAGES = frozenset(
    {"fin", "dan", "ces", "deu", "ell", "ita"}
)
SUPPORTED_RULE_LANGUAGES = frozenset({"eng", *SUPPORTED_UNIMORPH_LANGUAGES})

_POS_ATOMS = {
    "ADJ", "ADP", "ADV", "ART", "AUX", "CONJ", "DET", "INTJ", "N",
    "NUM", "PART", "PRO", "PRON", "PROPN", "V",
}
_INFLECTABLE_POS = {"ADJ", "ART", "AUX", "DET", "N", "NUM", "PRO", "PRON", "V"}
_PERSON_ATOMS = {"1", "2", "3"}
_NUMBER_ATOMS = {"SG", "PL", "DU", "TRI", "PAUC", "GRPL"}
_GENDER_ATOMS = {"MASC", "FEM", "NEUT", "COM"}
_TENSE_ATOMS = {
    "PRS", "PST", "FUT", "NPST", "RPST", "REMPST", "IMMEDPST",
    "HODPST", "REMFUT", "IMMEDFUT", "HODFUT",
}
_MOOD_ATOMS = {
    "ADM", "COND", "IMP", "IND", "INT", "JUS", "NEC", "OPT", "POT",
    "PURP", "SBJV", "SUBJ",
}
_ASPECT_ATOMS = {"HAB", "IPFV", "ITER", "PFV", "PRF", "PROG", "PROSP"}
_VOICE_ATOMS = {"ACT", "ANTIP", "CAUS", "MID", "PASS", "RECP", "REFL"}
_POLARITY_ATOMS = {"NEG", "POS"}
_FINITE_ATOMS = {"FIN"}
_NONFINITE_ATOMS = {
    "GER", "INF", "NFIN", "PTCP", "SUP", "V.CVB", "V.GER", "V.INF",
    "V.MSDR", "V.PTCP",
}
_AGREEMENT_ATOMS = _PERSON_ATOMS | _NUMBER_ATOMS | _GENDER_ATOMS
_TENSE_MOOD_ATOMS = _TENSE_ATOMS | _MOOD_ATOMS
_PRESERVE_AGREEMENT_DIMENSIONS = (
    _TENSE_ATOMS,
    _MOOD_ATOMS,
    _ASPECT_ATOMS,
    _VOICE_ATOMS,
    _POLARITY_ATOMS,
    _FINITE_ATOMS,
)
_FEATURE_ATOM_RE = re.compile(
    r"[A-Z][A-Z0-9_.-]*|(?<![A-Z0-9])[123](?![A-Z0-9])"
)
_VALID_OUTPUT_MODES = {"all", "first_success", "random_success"}


class UniMorphEntryLike(Protocol):
    lemma: str
    form: str
    features: str


class UniMorphStoreLike(Protocol):
    def analyze(self, lang: str, form: str) -> list[UniMorphEntryLike]: ...
    def inflect(self, lang: str, lemma: str) -> list[UniMorphEntryLike]: ...
    def has_language(self, lang: str) -> bool: ...


@dataclass(frozen=True)
class WordSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class MorphEntry:
    lemma: str
    form: str
    features: str


@dataclass(frozen=True)
class ScoredForm:
    score: int
    form: str


@dataclass(frozen=True)
class RuleTemplate:
    name: str
    criteria: str
    task: str
    fn: Callable[[dict], str]


def normalize_language(language: str) -> str:
    normalized = str(language or "").strip().lower().replace(" ", "-")
    if normalized not in _LANGUAGE_ALIASES:
        normalized = re.split(r"[-_]", normalized, maxsplit=1)[0]
    try:
        return _LANGUAGE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported rule-based perturbation language {language!r}. "
            "Supported languages: English, Finnish, Danish, Czech, German, "
            "Greek, Italian."
        ) from exc


def _require_unimorph_bindings() -> None:
    if _UniMorphStore is not None and _unimorph_download is not None:
        return
    message = (
        "Morphological perturbations require 'unimorph-rs'. Install it with "
        "'pip install unimorph-rs'; it is imported as 'unimorph'."
    )
    if _UNIMORPH_IMPORT_ERROR is not None:
        message += f" Original import error: {_UNIMORPH_IMPORT_ERROR}"
    raise ImportError(message)


def load_unimorph_store(language: str) -> UniMorphStoreLike:
    """Open a UniMorph store, downloading the requested non-English data if needed."""
    language = normalize_language(language)
    if language == "eng":
        raise ValueError("English morphology uses Lemminflect, not UniMorph")
    _require_unimorph_bindings()
    assert _UniMorphStore is not None
    assert _unimorph_download is not None
    store = _UniMorphStore()
    try:
        available = bool(store.has_language(language))
    except Exception as exc:
        raise RuntimeError(
            f"Could not inspect the local UniMorph store for {language!r}."
        ) from exc

    if not available:
        print(f"Downloading and indexing UniMorph data for '{language}'...")
        try:
            _unimorph_download(language)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download/index UniMorph data for {language!r}."
            ) from exc
        store = _UniMorphStore()

    try:
        if not store.has_language(language):
            raise RuntimeError(
                f"UniMorph language {language!r} is absent after download."
            )
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            f"Could not verify the UniMorph dataset for {language!r}."
        ) from exc
    return store


# Transitional private name for archived callers. New code uses
# ``load_unimorph_store`` through the canonical backend selector.
_ensure_language_downloaded = load_unimorph_store


def _open_downloaded_store(language: str) -> UniMorphStoreLike:
    _require_unimorph_bindings()
    assert _UniMorphStore is not None
    store = _UniMorphStore()
    if not store.has_language(language):
        raise RuntimeError(
            f"UniMorph language {language!r} was not initialized by the parent."
        )
    return store


def _feature_atoms(features: str) -> frozenset[str]:
    return frozenset(_FEATURE_ATOM_RE.findall(str(features).upper()))


def _part_of_speech(atoms: frozenset[str]) -> str | None:
    for atom in atoms:
        root = atom.split(".", maxsplit=1)[0]
        if root in _POS_ATOMS:
            return root
    return None


def _is_finite_verb(atoms: frozenset[str]) -> bool:
    if _part_of_speech(atoms) not in {"V", "AUX"}:
        return False
    if atoms & _NONFINITE_ATOMS:
        return False
    if any(
        atom.startswith(("V.PTCP", "V.CVB", "V.INF", "V.GER", "V.MSDR"))
        for atom in atoms
    ):
        return False
    return bool(
        atoms
        & (
            _FINITE_ATOMS
            | _PERSON_ATOMS
            | _NUMBER_ATOMS
            | _TENSE_ATOMS
            | _MOOD_ATOMS
        )
    )


def _agreement_signature(
    atoms: frozenset[str],
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    return (
        frozenset(atoms & _PERSON_ATOMS),
        frozenset(atoms & _NUMBER_ATOMS),
        frozenset(atoms & _GENDER_ATOMS),
    )


def _same_marked_dimensions(
    current: frozenset[str],
    candidate: frozenset[str],
    dimensions: tuple[set[str], ...],
) -> bool:
    for dimension in dimensions:
        current_values = current & dimension
        if current_values and candidate & dimension != current_values:
            return False
    return True


def _similarity_score(
    current: frozenset[str],
    candidate: frozenset[str],
    ignored: set[str] | frozenset[str] = frozenset(),
) -> int:
    current_core = current - ignored
    candidate_core = candidate - ignored
    return (
        5 * len(current_core & candidate_core)
        - 6 * len(current_core - candidate_core)
        - 2 * len(candidate_core - current_core)
    )


def _is_letter(character: str) -> bool:
    return bool(character) and unicodedata.category(character).startswith("L")


def _is_mark(character: str) -> bool:
    return bool(character) and unicodedata.category(character).startswith("M")


def _word_spans(text: str) -> list[WordSpan]:
    spans: list[WordSpan] = []
    i = 0
    while i < len(text):
        if not _is_letter(text[i]):
            i += 1
            continue
        start = i
        i += 1
        while i < len(text):
            char = text[i]
            if _is_letter(char) or _is_mark(char):
                i += 1
            elif char in {"'", "’"} and i + 1 < len(text) and _is_letter(text[i + 1]):
                i += 1
            else:
                break
        spans.append(WordSpan(start, i, text[start:i]))
    return spans


def _coerce_focus_offset(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _focused_word_indices(
    item: dict,
    spans: list[WordSpan],
    *,
    radius: int,
) -> list[int]:
    """
    Find focus-overlapping words and optionally include neighboring words.

    Missing focus metadata means all words, preserving compatibility with
    callers that do not use focused perturbations.
    """
    if not spans:
        return []
    if "focus_start" not in item and "focus_end" not in item:
        return list(range(len(spans)))

    text_length = len(str(item.get("text", "")))
    focus_start = min(
        max(_coerce_focus_offset(item.get("focus_start"), 0), 0),
        text_length,
    )
    focus_end = min(
        max(_coerce_focus_offset(item.get("focus_end"), focus_start), 0),
        text_length,
    )
    if focus_end < focus_start:
        focus_start, focus_end = focus_end, focus_start

    overlapping = [
        index
        for index, span in enumerate(spans)
        if span.end > focus_start and span.start < focus_end
    ]
    if not overlapping:
        focus_center_twice = focus_start + focus_end
        nearest = min(
            range(len(spans)),
            key=lambda index: abs(
                spans[index].start + spans[index].end - focus_center_twice
            ),
        )
        overlapping = [nearest]

    first = max(0, overlapping[0] - max(0, radius))
    last = min(len(spans) - 1, overlapping[-1] + max(0, radius))
    return list(range(first, last + 1))


def _focus_radius(item: dict) -> int:
    return max(
        0, _coerce_focus_offset(item.get("focus_word_radius"), 1)
    )


def _comma_adjacent(text: str, spans: list[WordSpan], index: int) -> bool:
    left_gap = (
        text[spans[index - 1].end : spans[index].start]
        if index > 0
        else text[: spans[index].start]
    )
    right_gap = (
        text[spans[index].end : spans[index + 1].start]
        if index + 1 < len(spans)
        else text[spans[index].end :]
    )
    return "," in left_gap or "," in right_gap


def _jumble_pair_score(
    text: str,
    spans: list[WordSpan],
    focus_index: int,
    other_index: int,
) -> int | None:
    """
    Prefer swaps likely to affect structure instead of list-item ordering.

    This is deliberately a transparent heuristic and can later be replaced by
    parser-based scoring without changing the public perturbation interface.
    """
    if focus_index == other_index:
        return None
    if spans[focus_index].text == spans[other_index].text:
        return None

    left_index, right_index = sorted((focus_index, other_index))
    between = text[spans[left_index].end : spans[right_index].start]
    focus_in_list = _comma_adjacent(text, spans, focus_index)
    other_in_list = _comma_adjacent(text, spans, other_index)
    only_weak_list_punctuation = bool(
        "," in between and not re.search(r"[;:!?…()\[\]{}]", between)
    )

    # Avoid merely permuting two apparent members of one comma-delimited list.
    if focus_in_list and other_in_list and only_weak_list_punctuation:
        return None

    distance = abs(focus_index - other_index)
    moves_out_of_list = focus_in_list != other_in_list
    crosses_strong_boundary = bool(
        re.search(r"[;:!?…()\[\]{}]", between)
    )

    score = 20 * distance
    score += 1000 if moves_out_of_list else 0
    score += 250 if crosses_strong_boundary else 0
    score += 100 if distance > 1 else 0
    score -= 100 if only_weak_list_punctuation else 0
    return score


def _replace_spans(
    text: str,
    replacements: list[tuple[int, int, str]],
) -> str:
    output = text
    for start, end, replacement in sorted(
        replacements, key=lambda row: row[0], reverse=True
    ):
        output = output[:start] + replacement + output[end:]
    return output


def _preserve_case(source: str, replacement: str) -> str:
    if not replacement:
        return replacement
    if source.isupper():
        return replacement.upper()
    if source.islower():
        return replacement.lower()
    if source[:1].isupper() and source[1:].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _grapheme_like_clusters(word: str) -> list[str]:
    clusters: list[str] = []
    for character in word:
        if _is_mark(character) and clusters:
            clusters[-1] += character
        else:
            clusters.append(character)
    return clusters


def _cluster_is_letter(cluster: str) -> bool:
    return bool(cluster) and _is_letter(cluster[0])


def _single_word_form(form: str) -> bool:
    candidate = str(form).strip()
    spans = _word_spans(candidate)
    return (
        len(spans) == 1
        and spans[0].start == 0
        and spans[0].end == len(candidate)
    )


def _stable_rng(base_seed: int | None, item: dict, salt: str) -> Any:
    if base_seed is None:
        return random
    identity = "\x1f".join(
        [
            str(base_seed),
            str(item.get("_source_ds", "")),
            str(item.get("_source_index", "")),
            str(item.get("_original_index", "")),
            str(item.get("text", "")),
            salt,
        ]
    )
    digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big", signed=False))


class MultilingualRulePerturber:
    def __init__(
        self,
        language: str,
        store: UniMorphStoreLike | None,
        random_seed: int | None = None,
    ):
        self.language = normalize_language(language)
        self.store = store
        self.random_seed = random_seed
        self._analysis_cache: dict[str, tuple[MorphEntry, ...]] = {}
        self._paradigm_cache: dict[str, tuple[MorphEntry, ...]] = {}

    @staticmethod
    def _text(item: dict) -> str:
        return str(item.get("text", ""))

    def _rng(self, item: dict, salt: str) -> Any:
        return _stable_rng(self.random_seed, item, salt)

    def _analyze(self, word: str) -> tuple[MorphEntry, ...]:
        if self.store is None:
            return ()
        if word in self._analysis_cache:
            return self._analysis_cache[word]

        normalized = unicodedata.normalize("NFC", word)
        queries: list[str] = []
        for query in (
            word, normalized, word.lower(), normalized.lower(),
            word.casefold(), normalized.casefold(),
        ):
            if query and query not in queries:
                queries.append(query)

        entries: list[MorphEntry] = []
        seen: set[tuple[str, str, str]] = set()
        for query in queries:
            try:
                raw_entries = self.store.analyze(self.language, query)
            except Exception:
                continue
            for raw in raw_entries:
                entry = MorphEntry(
                    str(raw.lemma), str(raw.form), str(raw.features)
                )
                key = (entry.lemma, entry.form, entry.features)
                if key not in seen:
                    seen.add(key)
                    entries.append(entry)

        result = tuple(entries)
        self._analysis_cache[word] = result
        return result

    def _paradigm(self, lemma: str) -> tuple[MorphEntry, ...]:
        if self.store is None:
            return ()
        if lemma in self._paradigm_cache:
            return self._paradigm_cache[lemma]
        try:
            raw_entries = self.store.inflect(self.language, lemma)
        except Exception:
            raw_entries = []

        entries: list[MorphEntry] = []
        seen: set[tuple[str, str, str]] = set()
        for raw in raw_entries:
            entry = MorphEntry(
                str(raw.lemma), str(raw.form), str(raw.features)
            )
            key = (entry.lemma, entry.form, entry.features)
            if key not in seen:
                seen.add(key)
                entries.append(entry)

        result = tuple(entries)
        self._paradigm_cache[lemma] = result
        return result

    @staticmethod
    def _form_differs(source: str, candidate: str) -> bool:
        return (
            _single_word_form(candidate)
            and unicodedata.normalize("NFC", source).casefold()
            != unicodedata.normalize("NFC", candidate).casefold()
        )

    @staticmethod
    def _choose_best(
        candidates: list[ScoredForm],
        source: str,
        rng: Any,
    ) -> str | None:
        if not candidates:
            return None
        best_by_form: dict[str, ScoredForm] = {}
        for candidate in candidates:
            key = unicodedata.normalize("NFC", candidate.form).casefold()
            previous = best_by_form.get(key)
            if previous is None or candidate.score > previous.score:
                best_by_form[key] = candidate
        highest = max(candidate.score for candidate in best_by_form.values())
        forms = sorted(
            candidate.form
            for candidate in best_by_form.values()
            if candidate.score == highest
        )
        return _preserve_case(source, rng.choice(forms))

    def _agreement_candidates(
        self, analysis: MorphEntry, source: str
    ) -> list[ScoredForm]:
        current_atoms = _feature_atoms(analysis.features)
        if not _is_finite_verb(current_atoms):
            return []
        current_signature = _agreement_signature(current_atoms)
        if not any(current_signature):
            return []

        current_core = current_atoms - _AGREEMENT_ATOMS
        current_pos = _part_of_speech(current_atoms)
        candidates: list[ScoredForm] = []
        for candidate in self._paradigm(analysis.lemma):
            if not self._form_differs(source, candidate.form):
                continue
            candidate_atoms = _feature_atoms(candidate.features)
            if _part_of_speech(candidate_atoms) != current_pos:
                continue
            if not _is_finite_verb(candidate_atoms):
                continue
            candidate_signature = _agreement_signature(candidate_atoms)
            if candidate_signature == current_signature:
                continue
            if any(
                current_values and not candidate_values
                for current_values, candidate_values in zip(
                    current_signature, candidate_signature
                )
            ):
                continue
            if not _same_marked_dimensions(
                current_atoms,
                candidate_atoms,
                _PRESERVE_AGREEMENT_DIMENSIONS,
            ):
                continue
            candidate_core = candidate_atoms - _AGREEMENT_ATOMS
            score = (
                (1000 if candidate_core == current_core else 0)
                + _similarity_score(
                    current_atoms, candidate_atoms, _AGREEMENT_ATOMS
                )
            )
            candidates.append(ScoredForm(score, candidate.form))
        return candidates

    def _danish_tense_mood_candidates(
        self, analysis: MorphEntry, source: str
    ) -> list[ScoredForm]:
        current_atoms = _feature_atoms(analysis.features)
        if not _is_finite_verb(current_atoms):
            return []
        current_pos = _part_of_speech(current_atoms)
        current_tense = current_atoms & _TENSE_ATOMS
        current_mood = current_atoms & _MOOD_ATOMS
        if not (current_tense or current_mood):
            return []

        preserve = (_ASPECT_ATOMS, _VOICE_ATOMS, _POLARITY_ATOMS)
        candidates: list[ScoredForm] = []
        for candidate in self._paradigm(analysis.lemma):
            if not self._form_differs(source, candidate.form):
                continue
            candidate_atoms = _feature_atoms(candidate.features)
            if _part_of_speech(candidate_atoms) != current_pos:
                continue
            if not _is_finite_verb(candidate_atoms):
                continue
            if not _same_marked_dimensions(
                current_atoms, candidate_atoms, preserve
            ):
                continue

            candidate_tense = candidate_atoms & _TENSE_ATOMS
            candidate_mood = candidate_atoms & _MOOD_ATOMS
            tense_changed = bool(
                current_tense
                and candidate_tense
                and candidate_tense != current_tense
            )
            mood_changed = bool(
                current_mood
                and candidate_mood
                and candidate_mood != current_mood
            )
            if not (tense_changed or mood_changed):
                continue

            if tense_changed and candidate_mood == current_mood:
                bonus = 600
            elif mood_changed and candidate_tense == current_tense:
                bonus = 500
            else:
                bonus = 400
            score = bonus + _similarity_score(
                current_atoms, candidate_atoms, _TENSE_MOOD_ATOMS
            )
            candidates.append(ScoredForm(score, candidate.form))
        return candidates

    def _random_inflection_candidates(
        self, analysis: MorphEntry, source: str
    ) -> list[ScoredForm]:
        current_atoms = _feature_atoms(analysis.features)
        current_pos = _part_of_speech(current_atoms)
        if current_pos not in _INFLECTABLE_POS:
            return []

        current_is_finite = _is_finite_verb(current_atoms)
        candidates: list[ScoredForm] = []
        for candidate in self._paradigm(analysis.lemma):
            if not self._form_differs(source, candidate.form):
                continue
            candidate_atoms = _feature_atoms(candidate.features)
            if _part_of_speech(candidate_atoms) != current_pos:
                continue
            if candidate_atoms == current_atoms:
                continue

            distance = len(current_atoms ^ candidate_atoms)
            common = len(current_atoms & candidate_atoms)
            finiteness_bonus = 0
            if current_pos in {"V", "AUX"}:
                finiteness_bonus = (
                    50
                    if _is_finite_verb(candidate_atoms) == current_is_finite
                    else 0
                )
            candidates.append(
                ScoredForm(
                    finiteness_bonus + 4 * common - 5 * distance,
                    candidate.form,
                )
            )
        return candidates

    def jumble(self, item: dict) -> str:
        """
        Swap two words; one must overlap the focus when focus is supplied.
        """
        text = self._text(item)
        spans = _word_spans(text)
        if len(spans) < 2:
            return text

        focus_indices = _focused_word_indices(item, spans, radius=0)
        scored_pairs: list[tuple[int, int, int]] = []
        for focus_index in focus_indices:
            for other_index in range(len(spans)):
                score = _jumble_pair_score(
                    text, spans, focus_index, other_index
                )
                if score is not None:
                    scored_pairs.append(
                        (score, focus_index, other_index)
                    )
        if not scored_pairs:
            return text

        highest = max(score for score, _, _ in scored_pairs)
        best_pairs = sorted(
            (first, second)
            for score, first, second in scored_pairs
            if score == highest
        )
        rng = self._rng(item, "jumble")
        first_index, second_index = rng.choice(best_pairs)
        first = spans[first_index]
        second = spans[second_index]
        return _replace_spans(
            text,
            [
                (first.start, first.end, second.text),
                (second.start, second.end, first.text),
            ],
        )

    def subject_verb_dis(self, item: dict) -> str:
        text = self._text(item)
        spans = _word_spans(text)
        rng = self._rng(item, "subject_verb_dis")
        eligible = _focused_word_indices(
            item, spans, radius=_focus_radius(item)
        )
        rng.shuffle(eligible)

        for index in eligible:
            span = spans[index]
            scored: list[ScoredForm] = []
            for analysis in self._analyze(span.text):
                if self.language == "dan":
                    scored.extend(
                        self._danish_tense_mood_candidates(
                            analysis, span.text
                        )
                    )
                else:
                    scored.extend(
                        self._agreement_candidates(analysis, span.text)
                    )
            replacement = self._choose_best(scored, span.text, rng)
            if replacement is not None:
                return _replace_spans(
                    text, [(span.start, span.end, replacement)]
                )
        return text

    def random_inflection(self, item: dict) -> str:
        text = self._text(item)
        spans = _word_spans(text)
        rng = self._rng(item, "random_inflection")
        eligible = _focused_word_indices(
            item, spans, radius=_focus_radius(item)
        )
        rng.shuffle(eligible)

        for index in eligible:
            span = spans[index]
            scored: list[ScoredForm] = []
            for analysis in self._analyze(span.text):
                scored.extend(
                    self._random_inflection_candidates(
                        analysis, span.text
                    )
                )
            replacement = self._choose_best(scored, span.text, rng)
            if replacement is not None:
                return _replace_spans(
                    text, [(span.start, span.end, replacement)]
                )
        return text

    def typos(self, item: dict) -> str:
        text = self._text(item)
        spans = _word_spans(text)
        if not spans:
            return text

        eligible_indices = set(
            _focused_word_indices(
                item, spans, radius=_focus_radius(item)
            )
        )
        preferred = [
            span
            for index, span in enumerate(spans)
            if index in eligible_indices
            and sum(
                _cluster_is_letter(cluster)
                for cluster in _grapheme_like_clusters(span.text)
            ) >= 4
        ]
        fallback = [
            span
            for index, span in enumerate(spans)
            if index in eligible_indices
            and sum(
                _cluster_is_letter(cluster)
                for cluster in _grapheme_like_clusters(span.text)
            ) >= 2
        ]
        candidates = preferred or fallback
        if not candidates:
            return text

        rng = self._rng(item, "typos")
        span = rng.choice(candidates)
        clusters = _grapheme_like_clusters(span.text)
        letter_indices = [
            index
            for index, cluster in enumerate(clusters)
            if _cluster_is_letter(cluster)
        ]
        if len(letter_indices) < 2:
            return text

        operations = ["delete", "duplicate", "insert"]
        transposable = [
            index
            for index in range(len(clusters) - 1)
            if _cluster_is_letter(clusters[index])
            and _cluster_is_letter(clusters[index + 1])
            and clusters[index] != clusters[index + 1]
        ]
        if transposable:
            operations.append("transpose")

        operation = rng.choice(operations)
        changed = list(clusters)
        if operation == "delete":
            internal = [
                index
                for index in letter_indices
                if index not in {letter_indices[0], letter_indices[-1]}
            ]
            del changed[rng.choice(internal or letter_indices)]
        elif operation == "duplicate":
            index = rng.choice(letter_indices)
            changed.insert(index, changed[index])
        elif operation == "transpose":
            index = rng.choice(transposable)
            changed[index], changed[index + 1] = (
                changed[index + 1], changed[index]
            )
        else:
            inserted = rng.choice(
                [clusters[index] for index in letter_indices]
            )
            changed.insert(
                rng.choice(list(range(1, len(changed) + 1))),
                inserted,
            )

        replacement = "".join(changed)
        if replacement == span.text:
            return text
        return _replace_spans(
            text, [(span.start, span.end, replacement)]
        )


def build_rule_templates(
    perturber: MultilingualRulePerturber,
) -> list[RuleTemplate]:
    return [
        RuleTemplate("jumble", "Fluency", "ALL", perturber.jumble),
        RuleTemplate(
            "subject_verb_dis",
            "Fluency",
            "ALL",
            perturber.subject_verb_dis,
        ),
        RuleTemplate(
            "random_inflection",
            "Fluency",
            "ALL",
            perturber.random_inflection,
        ),
        RuleTemplate("typos", "Fluency", "ALL", perturber.typos),
    ]


def select_rule_templates(
    templates: list[RuleTemplate],
    rule_task: str = "all",
    rule_criteria: str = "all",
    rule_template_names: list[str] | None = None,
) -> list[RuleTemplate]:
    task = (rule_task or "all").upper()
    criterion = (rule_criteria or "all").lower()
    name_filter = set(rule_template_names or [])

    selected: list[RuleTemplate] = []
    for template in templates:
        if task == "COMMON_FLUENCY":
            if (
                template.task != "ALL"
                or template.criteria.lower() != "fluency"
            ):
                continue
        else:
            if task != "ALL" and template.task not in {"ALL", task}:
                continue
            if (
                criterion != "all"
                and template.criteria.lower() != criterion
            ):
                continue
        if name_filter and template.name not in name_filter:
            continue
        selected.append(template)

    result: list[RuleTemplate] = []
    seen: set[tuple[str, str, str]] = set()
    for template in selected:
        key = (template.name, template.criteria, template.task)
        if key not in seen:
            seen.add(key)
            result.append(template)
    return result


def _valid_perturbation(
    original: str, perturbed: str | None
) -> bool:
    if perturbed is None:
        return False
    normalized = str(perturbed).strip()
    return bool(normalized) and normalized != str(original).strip()


def _apply_rule_templates_to_item(
    item: dict,
    templates: list[RuleTemplate],
    output_mode: str,
    model_label: str,
    random_seed: int | None,
) -> tuple[list[dict], int]:
    original = str(item.get("text", ""))
    successes: list[tuple[RuleTemplate, str]] = []
    failures = 0

    for template in templates:
        try:
            output = template.fn(item)
        except Exception:
            failures += 1
            continue
        if _valid_perturbation(original, output):
            successes.append((template, str(output).strip()))

    if output_mode == "first_success":
        successes = successes[:1]
    elif output_mode == "random_success" and successes:
        rng = _stable_rng(random_seed, item, "random_success")
        successes = [rng.choice(successes)]
    elif output_mode != "all":
        raise ValueError(f"Unknown output_mode: {output_mode}")

    return (
        [
            {
                "perturbation_type": template.name,
                "model": model_label,
                "head_id": item.get(
                    "_source_index", item.get("_original_index")
                ),
                "text": text,
                "max_length": item.get(
                    "max_length", max(len(original), len(text))
                ),
                "_source_ds": item.get("_source_ds"),
            }
            for template, text in successes
        ],
        failures,
    )


_WORKER_TEMPLATES: list[RuleTemplate] | None = None
_WORKER_OUTPUT_MODE = "all"
_WORKER_MODEL_LABEL = RULE_BASED_MODEL_LABEL
_WORKER_RANDOM_SEED: int | None = None


def _init_rule_worker(
    language: str,
    needs_unimorph: bool,
    rule_task: str,
    rule_criteria: str,
    rule_template_names: list[str] | None,
    output_mode: str,
    model_label: str,
    random_seed: int | None,
) -> None:
    global _WORKER_TEMPLATES
    global _WORKER_OUTPUT_MODE
    global _WORKER_MODEL_LABEL
    global _WORKER_RANDOM_SEED

    random.seed(
        os.getpid() ^ time.time_ns()
        if random_seed is None
        else random_seed
    )
    store = (
        _open_downloaded_store(language)
        if needs_unimorph
        else None
    )
    perturber = MultilingualRulePerturber(
        language, store, random_seed
    )
    _WORKER_TEMPLATES = select_rule_templates(
        build_rule_templates(perturber),
        rule_task,
        rule_criteria,
        rule_template_names,
    )
    _WORKER_OUTPUT_MODE = output_mode
    _WORKER_MODEL_LABEL = model_label
    _WORKER_RANDOM_SEED = random_seed


def _rule_worker_apply_item(
    item: dict,
) -> tuple[list[dict], int]:
    if _WORKER_TEMPLATES is None:
        raise RuntimeError("Worker templates were not initialized.")
    return _apply_rule_templates_to_item(
        item,
        _WORKER_TEMPLATES,
        _WORKER_OUTPUT_MODE,
        _WORKER_MODEL_LABEL,
        _WORKER_RANDOM_SEED,
    )


def rule_based_perturbation(
    ds_items: list[dict],
    rule_task: str = "all",
    rule_criteria: str = "all",
    rule_template_names: list[str] | None = None,
    output_mode: str = "all",
    model_label: str = RULE_BASED_MODEL_LABEL,
    n_jobs: int | None = None,
    chunksize: int | None = None,
    random_seed: int | None = None,
    language: str = "fi",
) -> list[dict]:
    if output_mode not in _VALID_OUTPUT_MODES:
        raise ValueError(f"Unknown output_mode: {output_mode}")

    language_code = normalize_language(language)
    prototype = MultilingualRulePerturber(
        language_code, None, random_seed
    )
    prototype_templates = select_rule_templates(
        build_rule_templates(prototype),
        rule_task,
        rule_criteria,
        rule_template_names,
    )
    if not prototype_templates:
        raise ValueError(
            "No templates selected. Available templates: jumble, "
            "subject_verb_dis, random_inflection, typos."
        )

    if not ds_items:
        print(
            f"Rule-based perturbation ({language_code}): selected "
            f"{len(prototype_templates)} template(s), produced 0 rows."
        )
        return []

    morphological_names = {
        "subject_verb_dis", "random_inflection"
    }
    needs_unimorph = any(
        template.name in morphological_names
        for template in prototype_templates
    )
    store = (
        _ensure_language_downloaded(language_code)
        if needs_unimorph
        else None
    )
    perturber = MultilingualRulePerturber(
        language_code, store, random_seed
    )
    templates = select_rule_templates(
        build_rule_templates(perturber),
        rule_task,
        rule_criteria,
        rule_template_names,
    )

    n_jobs_effective = (
        os.cpu_count() or 1
        if n_jobs is None
        else max(1, int(n_jobs))
    )
    n_jobs_effective = min(n_jobs_effective, len(ds_items))
    chunksize_effective = (
        max(1, len(ds_items) // max(1, n_jobs_effective * 4))
        if chunksize is None
        else max(1, int(chunksize))
    )

    rows: list[dict] = []
    failures = 0
    if n_jobs_effective == 1:
        if random_seed is not None:
            random.seed(random_seed)
        for item in tqdm(ds_items, desc="Generating perturbations..."):
            item_rows, item_failures = _apply_rule_templates_to_item(
                item,
                templates,
                output_mode,
                model_label,
                random_seed,
            )
            rows.extend(item_rows)
            failures += item_failures
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs_effective,
            initializer=_init_rule_worker,
            initargs=(
                language_code,
                needs_unimorph,
                rule_task,
                rule_criteria,
                rule_template_names,
                output_mode,
                model_label,
                random_seed,
            ),
        ) as executor:
            results = executor.map(
                _rule_worker_apply_item,
                ds_items,
                chunksize=chunksize_effective,
            )
            for item_rows, item_failures in tqdm(
                results,
                total=len(ds_items),
                desc=(
                    "Generating perturbations with "
                    f"{n_jobs_effective} workers..."
                ),
            ):
                rows.extend(item_rows)
                failures += item_failures

    print(
        f"Rule-based perturbation ({language_code}): selected "
        f"{len(templates)} template(s), produced {len(rows)} successful rows "
        f"from {len(ds_items)} source item(s); {failures} template "
        f"application(s) raised and were skipped. Used "
        f"{n_jobs_effective} worker process(es)."
    )
    return rows
