"""
Multilingual rule-based perturbations backed by UniMorph.

This module is an alternative to rule_based_evaleval.py. It keeps the same
output schema and public entry point while limiting the perturbation set to
four language-independent templates:

* jumble: swap two word tokens;
* subject_verb_dis: replace one finite verb with a conflicting form from the
  same lemma;
* random_inflection: replace one inflectable word with another form of the
  same lemma and part of speech;
* typos: introduce one Unicode-aware character-level typo.

Supported languages
-------------------
Finnish (fi/fin), Danish (da/dan), Czech (cs/ces), German (de/deu),
Modern Greek (el/ell), and Italian (it/ita).

The subject--verb perturbation intentionally uses UniMorph only; it does not
run a syntactic parser. For languages with person/number/gender agreement, it
changes those features while preserving tense, mood, aspect, voice, polarity,
and finiteness as closely as possible. Danish does not productively mark
subject agreement on finite verbs, so its fallback changes tense or mood.

UniMorph is a lexical resource rather than an unrestricted generator. A
morphological perturbation is therefore skipped when the surface form or its
lemma is absent from the downloaded dataset.

Installation
------------
The package is installed as unimorph-rs but imported as unimorph:

    pip install unimorph-rs

The selected language dataset is downloaded automatically in the parent
process before worker processes are started.
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
    # unimorph-rs exposes its Python bindings through this module name.
    from unimorph import Store as _UniMorphStore
    from unimorph import download as _unimorph_download

    _UNIMORPH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on the environment
    _UniMorphStore = None  # type: ignore[assignment]
    _unimorph_download = None  # type: ignore[assignment]
    _UNIMORPH_IMPORT_ERROR = exc


RULE_BASED_MODEL_LABEL = "UniMorph-rule-based"
RULE_BASED_OUTPUT_DIR = "trad_perturbed_layers"

_LANGUAGE_ALIASES = {
    "fi": "fin",
    "fin": "fin",
    "finnish": "fin",
    "da": "dan",
    "dan": "dan",
    "danish": "dan",
    "cs": "ces",
    "cz": "ces",
    "ces": "ces",
    "cze": "ces",
    "czech": "ces",
    "de": "deu",
    "deu": "deu",
    "ger": "deu",
    "german": "deu",
    "el": "ell",
    "ell": "ell",
    "gre": "ell",
    "greek": "ell",
    "modern-greek": "ell",
    "modern_greek": "ell",
    "it": "ita",
    "ita": "ita",
    "italian": "ita",
}

SUPPORTED_UNIMORPH_LANGUAGES = frozenset(
    {"fin", "dan", "ces", "deu", "ell", "ita"}
)

# UniMorph feature atoms relevant to candidate selection.
_POS_ATOMS = {
    "ADJ",
    "ADP",
    "ADV",
    "ART",
    "AUX",
    "CONJ",
    "DET",
    "INTJ",
    "N",
    "NUM",
    "PART",
    "PRO",
    "PRON",
    "PROPN",
    "V",
}
_INFLECTABLE_POS = {"ADJ", "ART", "AUX", "DET", "N", "NUM", "PRO", "PRON", "V"}

_PERSON_ATOMS = {"1", "2", "3"}
_NUMBER_ATOMS = {"SG", "PL", "DU", "TRI", "PAUC", "GRPL"}
_GENDER_ATOMS = {"MASC", "FEM", "NEUT", "COM"}

_TENSE_ATOMS = {
    "PRS",
    "PST",
    "FUT",
    "NPST",
    "RPST",
    "REMPST",
    "IMMEDPST",
    "HODPST",
    "REMFUT",
    "IMMEDFUT",
    "HODFUT",
}
_MOOD_ATOMS = {
    "ADM",
    "COND",
    "IMP",
    "IND",
    "INT",
    "JUS",
    "NEC",
    "OPT",
    "POT",
    "PURP",
    "SBJV",
    "SUBJ",
}
_ASPECT_ATOMS = {
    "HAB",
    "IPFV",
    "ITER",
    "PFV",
    "PRF",
    "PROG",
    "PROSP",
}
_VOICE_ATOMS = {"ACT", "ANTIP", "CAUS", "MID", "PASS", "RECP", "REFL"}
_POLARITY_ATOMS = {"NEG", "POS"}
_FINITE_ATOMS = {"FIN"}
_NONFINITE_ATOMS = {
    "GER",
    "INF",
    "NFIN",
    "PTCP",
    "SUP",
    "V.CVB",
    "V.GER",
    "V.INF",
    "V.MSDR",
    "V.PTCP",
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
    def analyze(self, lang: str, form: str) -> list[UniMorphEntryLike]:
        ...

    def inflect(self, lang: str, lemma: str) -> list[UniMorphEntryLike]:
        ...

    def has_language(self, lang: str) -> bool:
        ...


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
    """Return the ISO 639-3 code expected by UniMorph."""
    normalized = str(language or "").strip().lower().replace(" ", "-")
    # Accept common locale-like values, for example de-DE and fi_FI.
    if normalized not in _LANGUAGE_ALIASES:
        base = re.split(r"[-_]", normalized, maxsplit=1)[0]
        normalized = base
    try:
        return _LANGUAGE_ALIASES[normalized]
    except KeyError as exc:
        supported = "Finnish, Danish, Czech, German, Greek, Italian"
        raise ValueError(
            f"Unsupported rule-based perturbation language {language!r}. "
            f"Supported languages: {supported}."
        ) from exc


def _require_unimorph_bindings() -> None:
    if _UniMorphStore is not None and _unimorph_download is not None:
        return

    message = (
        "Morphological rule-based perturbations require the Python bindings "
        "from the 'unimorph-rs' package. Install them with "
        "'pip install unimorph-rs'. The package is imported as 'unimorph'."
    )
    if _UNIMORPH_IMPORT_ERROR is not None:
        message += f" Original import error: {_UNIMORPH_IMPORT_ERROR}"
    raise ImportError(message)


def _ensure_language_downloaded(language: str) -> UniMorphStoreLike:
    """
    Ensure that one UniMorph dataset is present, then return a fresh Store.

    Downloading is deliberately performed only in the parent process. Worker
    processes open the resulting SQLite-backed store for lookups.
    """
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
                f"Could not download/index UniMorph data for '{language}'. "
                "Check network access and the UniMorph dataset repository."
            ) from exc
        # Re-open after the separate download() call has updated the cache DB.
        store = _UniMorphStore()

    try:
        if not store.has_language(language):
            raise RuntimeError(
                f"UniMorph download completed, but language {language!r} "
                "is still not present in the local store."
            )
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            f"Could not verify the UniMorph dataset for {language!r}."
        ) from exc

    return store


def _open_downloaded_store(language: str) -> UniMorphStoreLike:
    """Open a dataset that the parent process has already downloaded."""
    _require_unimorph_bindings()
    assert _UniMorphStore is not None
    store = _UniMorphStore()
    if not store.has_language(language):
        raise RuntimeError(
            f"UniMorph language {language!r} was not initialized by the parent."
        )
    return store


def _feature_atoms(features: str) -> frozenset[str]:
    """
    Extract flat atoms from flat or hierarchical UniMorph feature strings.

    For example, V;IND;PRS;3;SG and V;IND;PRS;NOM(3,SG) expose the same
    person/number atoms.
    """
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
    if any(atom in _NONFINITE_ATOMS for atom in atoms):
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
    """
    Preserve every explicitly marked current value in the supplied dimensions.

    If the current bundle marks a tense, for example, the candidate must mark
    exactly the same tense. A dimension absent from the current bundle does
    not constrain the candidate.
    """
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
    common = len(current_core & candidate_core)
    missing = len(current_core - candidate_core)
    extra = len(candidate_core - current_core)
    return 5 * common - 6 * missing - 2 * extra


def _is_letter(character: str) -> bool:
    return bool(character) and unicodedata.category(character).startswith("L")


def _is_mark(character: str) -> bool:
    return bool(character) and unicodedata.category(character).startswith("M")


def _word_spans(text: str) -> list[WordSpan]:
    """
    Tokenize alphabetic words while retaining exact character offsets.

    This scanner is intentionally independent of language-specific tokenizers.
    It accepts Unicode letters and combining marks and keeps an internal
    apostrophe when it joins two letter sequences.
    """
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
                continue
            if (
                char in {"'", "’"}
                and i + 1 < len(text)
                and _is_letter(text[i + 1])
            ):
                i += 1
                continue
            break
        spans.append(WordSpan(start=start, end=i, text=text[start:i]))
    return spans


def _replace_spans(
    text: str,
    replacements: list[tuple[int, int, str]],
) -> str:
    """Replace non-overlapping spans without changing untouched whitespace."""
    output = text
    for start, end, replacement in sorted(
        replacements, key=lambda row: row[0], reverse=True
    ):
        output = output[:start] + replacement + output[end:]
    return output


def _preserve_case(source: str, replacement: str) -> str:
    """Transfer the source token's coarse casing pattern to a replacement."""
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
    """
    Approximate grapheme clusters using a base code point plus combining marks.

    This is sufficient for the Latin and Greek orthographies targeted here and
    prevents a typo operation from detaching a combining accent from its base.
    """
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
    """Accept exactly one alphabetic token, with no surrounding punctuation."""
    candidate = str(form).strip()
    spans = _word_spans(candidate)
    return (
        len(spans) == 1
        and spans[0].start == 0
        and spans[0].end == len(candidate)
    )


def _stable_rng(
    base_seed: int | None,
    item: dict,
    salt: str,
) -> Any:
    """
    Return deterministic per-item randomness when a seed is supplied.

    The result is independent of worker count and process IDs. Without a
    supplied seed, the process-local random module is used.
    """
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
    return random.Random(int.from_bytes(digest, byteorder="big", signed=False))


class MultilingualRulePerturber:
    """Implement the four perturbation templates for one language."""

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

        cache_key = word
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        normalized = unicodedata.normalize("NFC", word)
        queries = []
        for query in (
            word,
            normalized,
            word.lower(),
            normalized.lower(),
            word.casefold(),
            normalized.casefold(),
        ):
            if query and query not in queries:
                queries.append(query)

        collected: list[MorphEntry] = []
        seen: set[tuple[str, str, str]] = set()
        for query in queries:
            try:
                raw_entries = self.store.analyze(self.language, query)
            except Exception:
                continue
            for raw in raw_entries:
                entry = MorphEntry(
                    lemma=str(raw.lemma),
                    form=str(raw.form),
                    features=str(raw.features),
                )
                key = (entry.lemma, entry.form, entry.features)
                if key not in seen:
                    seen.add(key)
                    collected.append(entry)

        result = tuple(collected)
        self._analysis_cache[cache_key] = result
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

        collected: list[MorphEntry] = []
        seen: set[tuple[str, str, str]] = set()
        for raw in raw_entries:
            entry = MorphEntry(
                lemma=str(raw.lemma),
                form=str(raw.form),
                features=str(raw.features),
            )
            key = (entry.lemma, entry.form, entry.features)
            if key not in seen:
                seen.add(key)
                collected.append(entry)

        result = tuple(collected)
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

        highest_score = max(candidate.score for candidate in best_by_form.values())
        best = [
            candidate.form
            for candidate in best_by_form.values()
            if candidate.score == highest_score
        ]
        replacement = rng.choice(sorted(best))
        return _preserve_case(source, replacement)

    def _agreement_candidates(
        self,
        analysis: MorphEntry,
        source: str,
    ) -> list[ScoredForm]:
        current_atoms = _feature_atoms(analysis.features)
        if not _is_finite_verb(current_atoms):
            return []

        current_signature = _agreement_signature(current_atoms)
        if not any(current_signature):
            return []

        candidates: list[ScoredForm] = []
        current_core = current_atoms - _AGREEMENT_ATOMS
        current_pos = _part_of_speech(current_atoms)

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

            # Do not compare a fully specified current dimension to a target
            # that simply omits that dimension.
            incomplete = False
            for current_values, candidate_values in zip(
                current_signature, candidate_signature
            ):
                if current_values and not candidate_values:
                    incomplete = True
                    break
            if incomplete:
                continue

            if not _same_marked_dimensions(
                current_atoms,
                candidate_atoms,
                _PRESERVE_AGREEMENT_DIMENSIONS,
            ):
                continue

            candidate_core = candidate_atoms - _AGREEMENT_ATOMS
            exact_core_bonus = 1000 if candidate_core == current_core else 0
            score = exact_core_bonus + _similarity_score(
                current_atoms,
                candidate_atoms,
                ignored=_AGREEMENT_ATOMS,
            )
            candidates.append(ScoredForm(score=score, form=candidate.form))

        return candidates

    def _danish_tense_mood_candidates(
        self,
        analysis: MorphEntry,
        source: str,
    ) -> list[ScoredForm]:
        current_atoms = _feature_atoms(analysis.features)
        if not _is_finite_verb(current_atoms):
            return []

        current_pos = _part_of_speech(current_atoms)
        current_tense = current_atoms & _TENSE_ATOMS
        current_mood = current_atoms & _MOOD_ATOMS
        if not (current_tense or current_mood):
            return []

        # Preserve dimensions that are not the intended Danish fallback.
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
            if not _same_marked_dimensions(current_atoms, candidate_atoms, preserve):
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

            # Prefer a tense-only change, then a mood-only change, then a
            # candidate that changes both dimensions.
            if tense_changed and candidate_mood == current_mood:
                change_bonus = 600
            elif mood_changed and candidate_tense == current_tense:
                change_bonus = 500
            else:
                change_bonus = 400

            score = change_bonus + _similarity_score(
                current_atoms,
                candidate_atoms,
                ignored=_TENSE_MOOD_ATOMS,
            )
            candidates.append(ScoredForm(score=score, form=candidate.form))

        return candidates

    def _random_inflection_candidates(
        self,
        analysis: MorphEntry,
        source: str,
    ) -> list[ScoredForm]:
        current_atoms = _feature_atoms(analysis.features)
        current_pos = _part_of_speech(current_atoms)
        if current_pos not in _INFLECTABLE_POS:
            return []

        candidates: list[ScoredForm] = []
        current_is_finite = _is_finite_verb(current_atoms)

        for candidate in self._paradigm(analysis.lemma):
            if not self._form_differs(source, candidate.form):
                continue

            candidate_atoms = _feature_atoms(candidate.features)
            if _part_of_speech(candidate_atoms) != current_pos:
                continue
            if candidate_atoms == current_atoms:
                continue

            # Prefer a small feature-bundle change. For verbs, additionally
            # prefer retaining finite/non-finite status when alternatives exist.
            distance = len(current_atoms ^ candidate_atoms)
            common = len(current_atoms & candidate_atoms)
            finiteness_bonus = 0
            if current_pos in {"V", "AUX"}:
                finiteness_bonus = (
                    50
                    if _is_finite_verb(candidate_atoms) == current_is_finite
                    else 0
                )
            score = finiteness_bonus + 4 * common - 5 * distance
            candidates.append(ScoredForm(score=score, form=candidate.form))

        return candidates

    def jumble(self, item: dict) -> str:
        """Swap exactly two different word tokens while preserving punctuation."""
        text = self._text(item)
        spans = _word_spans(text)
        if len(spans) < 2:
            return text

        rng = self._rng(item, "jumble")
        indices = list(range(len(spans)))
        rng.shuffle(indices)

        first_index: int | None = None
        second_index: int | None = None
        for i, left_index in enumerate(indices):
            for right_index in indices[i + 1 :]:
                if spans[left_index].text != spans[right_index].text:
                    first_index = left_index
                    second_index = right_index
                    break
            if first_index is not None:
                break

        if first_index is None or second_index is None:
            return text

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
        """
        Change one finite verb's agreement features.

        Danish uses the agreed tense/mood fallback because its finite verbs do
        not productively distinguish subject person or number.
        """
        text = self._text(item)
        spans = _word_spans(text)
        rng = self._rng(item, "subject_verb_dis")
        rng.shuffle(spans)

        for span in spans:
            scored: list[ScoredForm] = []
            for analysis in self._analyze(span.text):
                if self.language == "dan":
                    scored.extend(
                        self._danish_tense_mood_candidates(analysis, span.text)
                    )
                else:
                    scored.extend(self._agreement_candidates(analysis, span.text))

            replacement = self._choose_best(scored, span.text, rng)
            if replacement is not None:
                return _replace_spans(
                    text, [(span.start, span.end, replacement)]
                )

        return text

    def random_inflection(self, item: dict) -> str:
        """Change one random eligible word to another inflection of its lemma."""
        text = self._text(item)
        spans = _word_spans(text)
        rng = self._rng(item, "random_inflection")
        rng.shuffle(spans)

        for span in spans:
            scored: list[ScoredForm] = []
            for analysis in self._analyze(span.text):
                scored.extend(
                    self._random_inflection_candidates(analysis, span.text)
                )

            replacement = self._choose_best(scored, span.text, rng)
            if replacement is not None:
                return _replace_spans(
                    text, [(span.start, span.end, replacement)]
                )

        return text

    def typos(self, item: dict) -> str:
        """Introduce one deletion, duplication, transposition, or insertion."""
        text = self._text(item)
        spans = _word_spans(text)
        if not spans:
            return text

        preferred = [
            span
            for span in spans
            if sum(
                _cluster_is_letter(cluster)
                for cluster in _grapheme_like_clusters(span.text)
            )
            >= 4
        ]
        fallback = [
            span
            for span in spans
            if sum(
                _cluster_is_letter(cluster)
                for cluster in _grapheme_like_clusters(span.text)
            )
            >= 2
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
            # Prefer an internal letter so the typo remains recognizable.
            internal = [
                index
                for index in letter_indices
                if index not in {letter_indices[0], letter_indices[-1]}
            ]
            delete_at = rng.choice(internal or letter_indices)
            del changed[delete_at]
        elif operation == "duplicate":
            duplicate_at = rng.choice(letter_indices)
            changed.insert(duplicate_at, changed[duplicate_at])
        elif operation == "transpose":
            transpose_at = rng.choice(transposable)
            changed[transpose_at], changed[transpose_at + 1] = (
                changed[transpose_at + 1],
                changed[transpose_at],
            )
        else:  # insert
            inserted_cluster = rng.choice(
                [clusters[index] for index in letter_indices]
            )
            insertion_points = list(range(1, len(changed) + 1))
            changed.insert(rng.choice(insertion_points), inserted_cluster)

        replacement = "".join(changed)
        if replacement == span.text:
            return text
        return _replace_spans(text, [(span.start, span.end, replacement)])


def build_rule_templates(
    perturber: MultilingualRulePerturber,
) -> list[RuleTemplate]:
    """Return the four shared fluency templates."""
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
    """
    Keep the selection behavior of rule_based_evaleval.py.

    All templates are shared (task ALL), so selecting an individual NLG task
    still includes these fluency perturbations.
    """
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

    seen: set[tuple[str, str, str]] = set()
    deduplicated: list[RuleTemplate] = []
    for template in selected:
        key = (template.name, template.criteria, template.task)
        if key not in seen:
            seen.add(key)
            deduplicated.append(template)
    return deduplicated


def _valid_perturbation(original: str, perturbed: str | None) -> bool:
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

    rows: list[dict] = []
    for template, text in successes:
        rows.append(
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
        )
    return rows, failures


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
    """Initialize process-local Store, caches, templates, and randomness."""
    global _WORKER_TEMPLATES
    global _WORKER_OUTPUT_MODE
    global _WORKER_MODEL_LABEL
    global _WORKER_RANDOM_SEED

    if random_seed is None:
        random.seed(os.getpid() ^ time.time_ns())
    else:
        # Template methods use stable per-item RNGs. This process seed is only
        # a fallback for code paths called without a stable item seed.
        random.seed(random_seed)

    store = _open_downloaded_store(language) if needs_unimorph else None
    perturber = MultilingualRulePerturber(
        language=language,
        store=store,
        random_seed=random_seed,
    )
    _WORKER_TEMPLATES = select_rule_templates(
        build_rule_templates(perturber),
        rule_task=rule_task,
        rule_criteria=rule_criteria,
        rule_template_names=rule_template_names,
    )
    _WORKER_OUTPUT_MODE = output_mode
    _WORKER_MODEL_LABEL = model_label
    _WORKER_RANDOM_SEED = random_seed


def _rule_worker_apply_item(item: dict) -> tuple[list[dict], int]:
    if _WORKER_TEMPLATES is None:
        raise RuntimeError("Worker templates were not initialized.")
    return _apply_rule_templates_to_item(
        item=item,
        templates=_WORKER_TEMPLATES,
        output_mode=_WORKER_OUTPUT_MODE,
        model_label=_WORKER_MODEL_LABEL,
        random_seed=_WORKER_RANDOM_SEED,
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
    """
    Apply multilingual rule-based perturbations.

    The returned rows use the existing pipeline schema: perturbation_type,
    model, head_id, text, max_length, and _source_ds.

    Parameters
    ----------
    ds_items:
        Source item dictionaries containing at least text.
    language:
        A supported two- or three-letter language code. It is placed last in
        the signature to preserve positional compatibility with the older
        implementation.
    output_mode:
        all, first_success, or random_success.
    n_jobs:
        None uses all available CPUs; 1 runs sequentially.
    random_seed:
        When set, results are reproducible independently of worker count.
    """
    if output_mode not in _VALID_OUTPUT_MODES:
        raise ValueError(f"Unknown output_mode: {output_mode}")

    language_code = normalize_language(language)

    # Validate filters before downloading a potentially large dataset.
    prototype = MultilingualRulePerturber(
        language=language_code,
        store=None,
        random_seed=random_seed,
    )
    prototype_templates = select_rule_templates(
        build_rule_templates(prototype),
        rule_task=rule_task,
        rule_criteria=rule_criteria,
        rule_template_names=rule_template_names,
    )
    if not prototype_templates:
        raise ValueError(
            "No rule-based templates selected. Available templates: "
            "jumble, subject_verb_dis, random_inflection, typos."
        )

    if not ds_items:
        print(
            f"Rule-based perturbation ({language_code}): selected "
            f"{len(prototype_templates)} template(s), produced 0 successful "
            "rows from 0 source item(s); 0 template application(s) raised "
            "and were skipped."
        )
        return []

    morphological_names = {"subject_verb_dis", "random_inflection"}
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
        language=language_code,
        store=store,
        random_seed=random_seed,
    )
    templates = select_rule_templates(
        build_rule_templates(perturber),
        rule_task=rule_task,
        rule_criteria=rule_criteria,
        rule_template_names=rule_template_names,
    )

    if n_jobs is None:
        n_jobs_effective = os.cpu_count() or 1
    else:
        n_jobs_effective = max(1, int(n_jobs))
    n_jobs_effective = min(n_jobs_effective, len(ds_items))

    if chunksize is None:
        chunksize_effective = max(
            1, len(ds_items) // max(1, n_jobs_effective * 4)
        )
    else:
        chunksize_effective = max(1, int(chunksize))

    rows: list[dict] = []
    failures = 0

    if n_jobs_effective == 1:
        if random_seed is not None:
            random.seed(random_seed)
        for item in tqdm(ds_items, desc="Generating perturbations..."):
            item_rows, item_failures = _apply_rule_templates_to_item(
                item=item,
                templates=templates,
                output_mode=output_mode,
                model_label=model_label,
                random_seed=random_seed,
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
