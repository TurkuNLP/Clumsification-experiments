# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""Adapter exposing LemmInflect's English dictionary through the UniMorph API."""
from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class LemminflectEntry:
    lemma: str
    form: str
    features: str


_TAG_TO_POS = {
    "NN": "N", "NNS": "N", "NNP": "PROPN", "NNPS": "PROPN",
    "VB": "V", "VBD": "V", "VBG": "V", "VBN": "V", "VBP": "V", "VBZ": "V",
    "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ", "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
    "PRP": "PRON", "PRP$": "PRON", "DT": "DET", "PDT": "DET", "WDT": "DET",
    "IN": "ADP", "CC": "CONJ", "CD": "NUM", "MD": "AUX",
}

_UPOS_TO_POS = {
    "NOUN": "N", "PROPN": "PROPN", "VERB": "V", "AUX": "AUX",
    "ADJ": "ADJ", "ADV": "ADV", "PRON": "PRON", "DET": "DET",
    "ADP": "ADP", "CCONJ": "CONJ", "SCONJ": "CONJ", "NUM": "NUM",
}

_TAG_TO_FEATURES = {
    "NN": "N;SG", "NNS": "N;PL", "NNP": "PROPN;SG", "NNPS": "PROPN;PL",
    "VB": "V;NFIN", "VBD": "V;FIN;PST", "VBG": "V;V.PTCP;PRS",
    "VBN": "V;V.PTCP;PST", "VBP": "V;FIN;PRS;1;2;PL",
    "VBZ": "V;FIN;PRS;3;SG",
    "JJ": "ADJ", "JJR": "ADJ;CMPR", "JJS": "ADJ;SPRL",
    "RB": "ADV", "RBR": "ADV;CMPR", "RBS": "ADV;SPRL",
}


class LemminflectStore:
    """Fast, process-local store backed by LemmInflect's prebuilt lexicon."""

    def __init__(self):
        try:
            from lemminflect import getAllInflections, getAllLemmas
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "English morphology requires lemminflect; install it with "
                "'pip install lemminflect'."
            ) from exc
        self._get_all_inflections = getAllInflections
        self._get_all_lemmas = getAllLemmas
        self._analysis_cache: dict[str, list[LemminflectEntry]] = {}
        self._inflection_cache: dict[str, list[LemminflectEntry]] = {}

    @staticmethod
    def _features(tag: str) -> str:
        return _TAG_TO_FEATURES.get(tag, f"{_TAG_TO_POS.get(tag, 'X')};{tag}")

    def analyze(self, lang: str, form: str) -> list[LemminflectEntry]:
        if form not in self._analysis_cache:
            entries = []
            seen: set[tuple[str, str]] = set()
            for upos, lemmas in (self._get_all_lemmas(form) or {}).items():
                for lemma in lemmas:
                    matching_tags = [
                        tag
                        for tag, forms in (self._get_all_inflections(lemma) or {}).items()
                        if any(str(candidate).casefold() == form.casefold() for candidate in forms)
                    ]
                    features = (
                        [self._features(tag) for tag in matching_tags]
                        or [f"{_UPOS_TO_POS.get(str(upos), 'X')};{upos}"]
                    )
                    for feature_string in features:
                        key = (str(lemma), feature_string)
                        if key not in seen:
                            seen.add(key)
                            entries.append(
                                LemminflectEntry(str(lemma), form, feature_string)
                            )
            self._analysis_cache[form] = entries
        return self._analysis_cache[form]

    def inflect(self, lang: str, lemma: str) -> list[LemminflectEntry]:
        if lemma not in self._inflection_cache:
            entries = []
            for tag, forms in (self._get_all_inflections(lemma) or {}).items():
                for form in forms:
                    entries.append(LemminflectEntry(lemma, str(form), self._features(tag)))
            self._inflection_cache[lemma] = entries
        return self._inflection_cache[lemma]

    def has_language(self, lang: str) -> bool:
        return lang == "eng"

    def change_lemma(self, form: str, *, allow_pos_change: bool = False,
                     rng: random.Random | None = None) -> tuple[str, str] | None:
        """Return a form belonging to a different lemma, if available.

        The original POS is retained by default. ``allow_pos_change`` permits
        the deliberately broader cross-part-of-speech variant.
        """
        rng = rng or random
        analyses = self.analyze("eng", form)
        if not analyses:
            return None
        source = analyses[0]
        source_pos = source.features.split(";", 1)[0]
        lemmas = sorted({entry.lemma for entry in analyses if entry.lemma != source.lemma})
        if not lemmas:
            return None
        rng.shuffle(lemmas)
        for lemma in lemmas:
            candidates = self.inflect("eng", lemma)
            candidates = [entry for entry in candidates
                          if entry.form.casefold() != form.casefold()
                          and (allow_pos_change or entry.features.split(";", 1)[0] == source_pos)]
            if candidates:
                choice = rng.choice(candidates).form
                if form[:1].isupper():
                    choice = choice[:1].upper() + choice[1:]
                return choice, lemma
        return None
