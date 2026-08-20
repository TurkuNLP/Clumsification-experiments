# This script has been co-created, refactored, and cleaned using GPT 5.6.
"""
EvalEval-style non-LLM perturbations for the perturbation pipeline.

This module implements the rule/traditional perturbation templates described in
Sai et al. (2021), "Perturbation CheckLists for Evaluating NLG Evaluation
Metrics", and mirrored by https://github.com/iitmnlp/EvalEval.

It intentionally uses only stdlib by default, with optional NLTK/WordNet and
num2words support when those packages and their data are available.
"""

import random
import re
from dataclasses import dataclass
from typing import Callable
from tqdm.auto import tqdm

import os
import time
from concurrent.futures import ProcessPoolExecutor

try:
    import nltk  # type: ignore
    from nltk.corpus import stopwords as nltk_stopwords  # type: ignore
    from nltk.corpus import wordnet as wn  # type: ignore
except Exception:  # pragma: no cover
    nltk = None
    nltk_stopwords = None
    wn = None

try:
    from num2words import num2words as _num2words  # type: ignore
except Exception:  # pragma: no cover
    _num2words = None


RULE_BASED_MODEL_LABEL = "EvalEval-rule-based"
RULE_BASED_OUTPUT_DIR = "trad_perturbed_layers"

_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "don", "should", "now", "is",
    "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "have", "has", "had", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "their", "our",
}

_COMMON_NAMES = [
    "James", "Mary", "John", "Kate", "Raj", "Phillips", "Cameron", "Tesla",
    "Aron", "Alice", "Bob", "Maria", "David", "Sarah", "Michael", "Priya",
]

_GENERIC_WRONG_PHRASES = [
    "I forgot.",
    "This happened on the moon.",
    "The answer is a blue triangle.",
    "My father is my grandmother's father.",
    "The cricketer was born in 1990.",
]

_QUESTION_WORDS = {"what", "when", "where", "why", "how", "whose", "whom", "who", "which"}

_AUX_FLIPS = {
    "was": "were",
    "were": "was",
    "is": "are",
    "are": "is",
    "has": "have",
    "have": "has",
    "does": "do",
    "do": "does",
    "doesn't": "don't",
    "don't": "doesn't",
    "isn't": "aren't",
    "aren't": "isn't",
    "wasn't": "weren't",
    "weren't": "wasn't",
}

_CONTRACTIONS = {
    "are not": "aren't",
    "cannot": "can't",
    "can not": "can't",
    "could not": "couldn't",
    "did not": "didn't",
    "does not": "doesn't",
    "do not": "don't",
    "had not": "hadn't",
    "has not": "hasn't",
    "have not": "haven't",
    "is not": "isn't",
    "it is": "it's",
    "they are": "they're",
    "was not": "wasn't",
    "we are": "we're",
    "were not": "weren't",
    "will not": "won't",
    "would not": "wouldn't",
    "you are": "you're",
    "there is": "there's",
}

_EXPANSIONS = {v: k for k, v in _CONTRACTIONS.items()}
_EXPANSIONS.update({
    "I'm": "I am", "i'm": "I am", "I've": "I have", "i've": "I have",
    "I'll": "I will", "i'll": "I will", "I'd": "I would", "i'd": "I would",
})

_FALLBACK_SYNONYMS = {
    "delicious": "tasty", "big": "large", "small": "little", "quick": "fast",
    "happy": "glad", "sad": "unhappy", "smart": "clever", "good": "fine",
    "bad": "poor", "beautiful": "pretty", "begin": "start", "end": "finish",
}
_FALLBACK_ANTONYMS = {
    "good": "bad", "bad": "good", "happy": "sad", "sad": "happy",
    "large": "small", "big": "small", "small": "large", "hot": "cold",
    "cold": "hot", "old": "new", "new": "old", "inspiring": "uninspiring",
    "correct": "incorrect", "true": "false", "false": "true",
}
_FALLBACK_HYPONYMS = {
    "musician": "architect", "brother": "friend", "girl": "person",
    "dog": "poodle", "animal": "dog", "vehicle": "car", "fruit": "mango",
    "person": "child", "city": "village", "school": "market",
}
_FALLBACK_RELATED = {
    "red": "green", "green": "red", "small": "tall", "tall": "small",
    "young": "old", "old": "young", "black": "white", "white": "black",
    "blue": "yellow", "yellow": "blue",
}
_GENDER_SWAP = {
    "man": "woman", "woman": "man", "men": "women", "women": "men",
    "boy": "girl", "girl": "boy", "boys": "girls", "girls": "boys",
    "male": "female", "female": "male", "he": "she", "she": "he",
    "his": "her", "her": "his", "father": "mother", "mother": "father",
    "son": "daughter", "daughter": "son", "king": "queen", "queen": "king",
}


def _word_tokenize(text: str) -> list[str]:
    if nltk is not None:
        try:
            return nltk.word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]", text, flags=re.UNICODE)


def _sent_tokenize(text: str) -> list[str]:
    if nltk is not None:
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            pass
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _detokenize(tokens: list[str]) -> str:
    text = " ".join(t for t in tokens if t is not None and t != "")
    text = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+n(['’]t)\b", r"n\1", text)
    text = re.sub(r"\s+(['’](?:s|re|ve|ll|d|m))\b", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    if nltk is not None:
        try:
            return nltk.pos_tag(tokens)
        except Exception:
            pass
    tagged = []
    adjectives = set(_FALLBACK_SYNONYMS) | set(_FALLBACK_ANTONYMS) | set(_FALLBACK_RELATED)
    verbs = {"is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
             "do", "does", "did", "go", "goes", "went", "run", "runs", "ran", "play",
             "plays", "played", "make", "makes", "made", "say", "says", "said"}
    for tok in tokens:
        low = tok.lower()
        if not re.search(r"[A-Za-z0-9]", tok):
            tag = "."
        elif low in _AUX_FLIPS or low in verbs or low.endswith("ing") or low.endswith("ed"):
            tag = "VB"
        elif low in adjectives or low.endswith(("ous", "ful", "able", "ible", "ive", "al", "ic")):
            tag = "JJ"
        elif tok[:1].isupper() and tok.lower() not in _FALLBACK_STOPWORDS:
            tag = "NNP"
        elif low.endswith("s") and len(low) > 3:
            tag = "NNS"
        else:
            tag = "NN"
        tagged.append((tok, tag))
    return tagged


def _get_stopwords() -> set[str]:
    if nltk_stopwords is not None:
        try:
            return set(nltk_stopwords.words("english"))
        except Exception:
            pass
    return set(_FALLBACK_STOPWORDS)


def _wordnet_related(word: str, relation: str) -> list[str]:
    out = []
    if wn is not None:
        try:
            synsets = wn.synsets(word)
            if relation == "synonym":
                for syn in synsets:
                    out.extend(lemma.name().replace("_", " ") for lemma in syn.lemmas())
            elif relation == "antonym":
                for syn in synsets:
                    for lemma in syn.lemmas():
                        out.extend(a.name().replace("_", " ") for a in lemma.antonyms())
            elif relation == "hyponym":
                for syn in synsets:
                    for h in syn.hyponyms():
                        out.extend(lemma.name().replace("_", " ") for lemma in h.lemmas())
            elif relation == "sibling":
                for syn in synsets:
                    for hyper in syn.hypernyms():
                        for sibling in hyper.hyponyms():
                            if sibling != syn:
                                out.extend(lemma.name().replace("_", " ") for lemma in sibling.lemmas())
            out = [x for x in out if x.lower() != word.lower() and re.fullmatch(r"[A-Za-z][A-Za-z -]*", x)]
        except Exception:
            out = []
    if not out:
        mapping = {
            "synonym": _FALLBACK_SYNONYMS,
            "antonym": _FALLBACK_ANTONYMS,
            "hyponym": _FALLBACK_HYPONYMS,
            "sibling": _FALLBACK_RELATED,
        }[relation]
        if word.lower() in mapping:
            out = [mapping[word.lower()]]
    return list(dict.fromkeys(out))


def _preserve_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _replace_first_by_pos(text: str, tags: set[str], relation: str) -> str:
    toks = _word_tokenize(text)
    pos = _pos_tag(toks)
    for idx, (tok, tag) in enumerate(pos):
        if tag in tags and re.search(r"[A-Za-z]", tok):
            repls = _wordnet_related(tok, relation)
            if repls:
                toks[idx] = _preserve_case(tok, repls[0])
                return _detokenize(toks)
    return text


def _simple_num2words(n: int) -> str:
    if _num2words is not None:
        try:
            return _num2words(n)
        except Exception:
            pass
    small = ["zero","one","two","three","four","five","six","seven","eight","nine","ten",
             "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
             "eighteen","nineteen"]
    tens = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
    if 0 <= n < 20:
        return small[n]
    if n < 100:
        return tens[n // 10] + ("" if n % 10 == 0 else " " + small[n % 10])
    if n < 1000:
        return small[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _simple_num2words(n % 100))
    return str(n)


@dataclass(frozen=True)
class RuleTemplate:
    name: str
    criteria: str
    task: str
    fn: Callable[[dict], str]


class EvalEvalRulePerturber:
    def __init__(self, corpus_texts: list[str] | None = None):
        self.stopwords = _get_stopwords()
        self.corpus_texts = [t for t in (corpus_texts or []) if isinstance(t, str) and t.strip()]

    def _text(self, item: dict) -> str:
        return str(item.get("text", ""))

    def remove_punct(self, item: dict) -> str:
        return re.sub(r"[^\w\s]", " ", self._text(item)).strip()

    def misplaced_punctuation(self, item: dict) -> str:
        text = self._text(item)
        if not text.strip():
            return text
        if re.search(r"[,.!?;:]", text):
            return re.sub(r"[,.!?;:]", "", text, count=1).strip()
        toks = _word_tokenize(text)
        if len(toks) > 4:
            toks.insert(len(toks) // 2, ",")
            return _detokenize(toks)
        return text + " ,"

    def typos(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        for i, tok in enumerate(toks):
            if tok.isalpha() and len(tok) > 4:
                j = random.randint(1, len(tok) - 2)
                toks[i] = tok[:j] + tok[j + 1:]
                return _detokenize(toks)
        return self._text(item)

    def contractions(self, item: dict) -> str:
        text = self._text(item)
        for src, dst in _CONTRACTIONS.items():
            pattern = re.compile(r"\b" + re.escape(src) + r"\b", flags=re.IGNORECASE)
            if pattern.search(text):
                return pattern.sub(dst, text, count=1)
        return text

    def expansions(self, item: dict) -> str:
        text = self._text(item)
        for src, dst in _EXPANSIONS.items():
            pattern = re.compile(r"\b" + re.escape(src) + r"\b", flags=re.IGNORECASE)
            if pattern.search(text):
                return pattern.sub(dst, text, count=1)
        return text

    def add_negation(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        for i, tok in enumerate(toks):
            if tok.lower() in {"is", "are", "was", "were", "has", "have", "had", "do", "does", "did", "will", "can", "could", "should", "would"}:
                toks.insert(i + 1, "not")
                return _detokenize(toks)
        for i, tok in enumerate(toks):
            if tok.isalpha():
                toks.insert(i, "not")
                return _detokenize(toks)
        return self._text(item)

    def jumble(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        word_idx = [i for i, t in enumerate(toks) if re.search(r"\w", t)]
        if len(word_idx) < 3:
            return self._text(item)
        words = [toks[i] for i in word_idx]
        shuffled = words[:]
        for _ in range(5):
            random.shuffle(shuffled)
            if shuffled != words:
                break
        for i, w in zip(word_idx, shuffled):
            toks[i] = w
        return _detokenize(toks)

    def drop_stopwords(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        kept = [t for t in toks if t.lower() not in self.stopwords]
        return _detokenize(kept) if len(kept) < len(toks) else self._text(item)

    def synonym_adjective(self, item: dict) -> str:
        return _replace_first_by_pos(self._text(item), {"JJ", "JJR", "JJS"}, "synonym")

    def antonym_adjective(self, item: dict) -> str:
        return _replace_first_by_pos(self._text(item), {"JJ", "JJR", "JJS"}, "antonym")

    def hyponyms(self, item: dict) -> str:
        return _replace_first_by_pos(self._text(item), {"NN", "NNP", "NNS", "VB", "VBP", "VBZ"}, "hyponym")

    def subject_verb_dis(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        for i, tok in enumerate(toks):
            low = tok.lower()
            if low in _AUX_FLIPS:
                repl = _AUX_FLIPS[low]
                toks[i] = _preserve_case(tok, repl)
                return _detokenize(toks)
        return self._text(item)

    def number2words(self, item: dict) -> str:
        text = self._text(item)
        def repl(m: re.Match) -> str:
            return _simple_num2words(int(m.group(0)))
        return re.sub(r"\b\d+\b", repl, text, count=1)

    def repeat_phrases(self, item: dict) -> str:
        toks = [t for t in _word_tokenize(self._text(item)) if re.search(r"\w", t)]
        if len(toks) < 2:
            return self._text(item)
        n = max(1, min(4, round(len(toks) * 0.25)))
        phrase = " ".join(toks[:n])
        sep = "," if not self._text(item).rstrip().endswith((",", ".")) else ""
        return self._text(item).rstrip() + sep + " " + phrase

    def drop_adjectives(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        pos = _pos_tag(toks)
        kept = []
        changed = False
        for i, (w, p) in enumerate(pos):
            nxt = pos[i + 1][1] if i + 1 < len(pos) else ""
            if p in {"JJ", "JJR", "JJS"} and nxt in {"NN", "NNP", "NNS", "NNPS"}:
                changed = True
                continue
            kept.append(w)
        return _detokenize(kept) if changed else self._text(item)

    def only_stop(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        kept = [t for t in toks if t.lower() in self.stopwords or not re.search(r"\w", t)]
        return _detokenize(kept) if kept and len(kept) < len(toks) else self._text(item)

    def change_numeric(self, item: dict) -> str:
        text = self._text(item)
        def repl(m: re.Match) -> str:
            raw = m.group(0)
            val = int(raw)
            delta = random.choice([-50, -10, -5, 5, 10, 50])
            return str(max(0, val + delta) if val + delta != val else val + 1)
        return re.sub(r"\b\d+\b", repl, text, count=1)

    def change_names(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        for i, tok in enumerate(toks):
            if tok[:1].isupper() and tok.isalpha() and tok.lower() not in self.stopwords:
                choices = [n for n in _COMMON_NAMES if n.lower() != tok.lower()]
                toks[i] = random.choice(choices)
                return _detokenize(toks)
        return self.perturb_named_entities_nouns_verbs(item)

    def drop_phrases(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        word_positions = [i for i, t in enumerate(toks) if re.search(r"\w", t)]
        if len(word_positions) < 3:
            return self._text(item)
        n_drop = max(1, round(len(word_positions) * 0.2))
        start = random.randint(0, max(0, len(word_positions) - n_drop))
        drop_positions = set(word_positions[start:start + n_drop])
        kept = [t for i, t in enumerate(toks) if i not in drop_positions]
        return _detokenize(kept)

    def sentence_reorder(self, item: dict) -> str:
        sents = _sent_tokenize(self._text(item))
        if len(sents) < 2:
            return self._text(item)
        reordered = sents[:]
        random.shuffle(reordered)
        if reordered == sents:
            reordered = sents[1:] + sents[:1]
        return " ".join(reordered)

    def repeat_sentences(self, item: dict) -> str:
        sents = _sent_tokenize(self._text(item))
        if not sents:
            return self._text(item)
        return self._text(item).rstrip() + " " + sents[0]

    def replace_nouns_pronouns(self, item: dict) -> str:
        sents = _sent_tokenize(self._text(item))
        if not sents:
            return self._text(item)
        out = []
        changed = False
        for sent in sents:
            toks = _word_tokenize(sent)
            pos = _pos_tag(toks)
            for i, (_, tag) in enumerate(pos):
                if tag in {"NNS", "NNPS"}:
                    toks[i] = "They" if i == 0 else "they"
                    changed = True
                    break
                if tag in {"NN", "NNP"}:
                    toks[i] = "It" if i == 0 else "it"
                    changed = True
                    break
            out.append(_detokenize(toks))
        return " ".join(out) if changed else self._text(item)

    def remove_question_word(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        kept = [w for w in toks if w.lower() not in _QUESTION_WORDS]
        return _detokenize(kept) if len(kept) < len(toks) else self._text(item)

    def change_question_word(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        for i, w in enumerate(toks):
            if w.lower() in _QUESTION_WORDS:
                choices = [q for q in _QUESTION_WORDS if q != w.lower()]
                toks[i] = _preserve_case(w, random.choice(sorted(choices)))
                return _detokenize(toks)
        return self._text(item)

    def change_question_to_assertion(self, item: dict) -> str:
        text = self._text(item).strip()
        text = re.sub(r"\?\s*$", ".", text)
        toks = _word_tokenize(text)
        if not toks:
            return self._text(item)
        if toks[0].lower() in {"who", "what"} and len(toks) > 2 and toks[1].lower() in {"is", "are", "was", "were"}:
            pred = _detokenize(toks[2:]).rstrip(".")
            filler = "someone" if toks[0].lower() == "who" else "something"
            return f"{pred} {toks[1].lower()} {filler}."
        if toks[0].lower() in _QUESTION_WORDS:
            toks = toks[1:]
            if toks and toks[-1] == "?":
                toks[-1] = "."
            out = _detokenize(toks)
            return out if out.endswith(".") else out + "."
        return text

    def mask_words_predict(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        pos = _pos_tag(toks)
        replacements = ["market", "beach", "experiment", "artist", "teacher", "river", "book"]
        for i, (w, tag) in enumerate(pos):
            if tag in {"NN", "NNP", "NNS", "VB", "VBP", "VBZ"} and w.isalpha() and w.lower() not in self.stopwords:
                choices = [r for r in replacements if r.lower() != w.lower()]
                toks[i] = _preserve_case(w, random.choice(choices))
                return _detokenize(toks)
        return self._text(item)

    def perturb_named_entities_nouns_verbs(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        pos = _pos_tag(toks)
        pool = ["Raj", "Mary", "market", "school", "wrote", "bought", "city", "teacher"]
        for i, (w, tag) in enumerate(pos):
            if tag in {"NN", "NNP", "NNS", "VB", "VBP", "VBZ"} and w.isalpha() and w.lower() not in self.stopwords:
                choices = [p for p in pool if p.lower() != w.lower()]
                toks[i] = _preserve_case(w, random.choice(choices))
                return _detokenize(toks)
        return self._text(item)

    def change_attributes(self, item: dict) -> str:
        return _replace_first_by_pos(self._text(item), {"JJ", "JJR", "JJS"}, "sibling")

    def change_gender(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        for i, tok in enumerate(toks):
            low = tok.lower()
            if low in _GENDER_SWAP:
                toks[i] = _preserve_case(tok, _GENDER_SWAP[low])
                return _detokenize(toks)
        return self._text(item)

    def drop_objects(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        pos = _pos_tag(toks)
        kept = []
        changed = False
        for w, tag in pos:
            if tag in {"NN", "NNP", "NNS", "NNPS"} and w.isalpha():
                changed = True
                continue
            kept.append(w)
        return _detokenize(kept) if changed and kept else self._text(item)

    def replace_object_with_synonym(self, item: dict) -> str:
        return _replace_first_by_pos(self._text(item), {"NN", "NNP", "NNS", "NNPS"}, "synonym")

    def repeat_object(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        pos = _pos_tag(toks)
        for w, tag in pos:
            if tag in {"NN", "NNP", "NNS", "NNPS"} and w.isalpha():
                return self._text(item).rstrip() + " and " + w + " ."
        return self._text(item)

    def change_object_order(self, item: dict) -> str:
        toks = _word_tokenize(self._text(item))
        pos = _pos_tag(toks)
        noun_positions = [i for i, (_, tag) in enumerate(pos) if tag in {"NN", "NNS", "NNP", "NNPS"}]
        if len(noun_positions) > 1:
            i, j = noun_positions[0], noun_positions[-1]
            toks[i], toks[j] = toks[j], toks[i]
            return _detokenize(toks)
        return self._text(item)

    def add_extra_wrong_information(self, item: dict) -> str:
        phrase = random.choice(_GENERIC_WRONG_PHRASES)
        text = self._text(item).rstrip()
        if not text:
            return phrase
        sep = " " if text.endswith((".", "!", "?")) else ", "
        return text + sep + phrase

    def random_text(self, item: dict) -> str:
        text = self._text(item)
        choices = [t for t in self.corpus_texts if t != text]
        if choices:
            return random.choice(choices)
        return random.choice(_GENERIC_WRONG_PHRASES)

    def _context_utterances(self, item: dict) -> list[str]:
        context = item.get("context") or item.get("dialogue_context") or item.get("source") or item.get("prompt")
        if not isinstance(context, str) or not context.strip():
            return []
        return [re.sub(r"^[A-Za-z]*S?\s*:\s*", "", x).strip() for x in context.strip().split("\n") if x.strip()]

    def negate_previous_utterance(self, item: dict) -> str:
        utts = self._context_utterances(item)
        if utts:
            return self.add_negation({"text": random.choice(utts)})
        return self.add_negation(item)

    def repeat_itself(self, item: dict) -> str:
        utts = self._context_utterances(item)
        if utts:
            return random.choice(utts)
        return self._text(item)

    def repeat_last_speaker(self, item: dict) -> str:
        utts = self._context_utterances(item)
        if utts:
            return utts[-1]
        return self.repeat_phrases(item)

    def sorry_reply(self, item: dict) -> str:
        return "I'm sorry, can you repeat?"

    def generic(self, item: dict) -> str:
        return random.choice(["Yes.", "Ok.", "Thank you.", "Bye."])

    def nonsensible_reply(self, item: dict) -> str:
        return random.choice([
            "Yes, my father is my grandmother's father.",
            "The square tomato sings loudly.",
            "I parked the ocean under my bed.",
        ])


def build_rule_templates(perturber: EvalEvalRulePerturber) -> list[RuleTemplate]:
    P = perturber
    common_fluency = [
        ("misplaced_punctuation", "Fluency", P.misplaced_punctuation),
        ("jumble", "Fluency", P.jumble),
        ("subject_verb_dis", "Fluency", P.subject_verb_dis),
        ("drop_stopwords", "Fluency", P.drop_stopwords),
        ("typos", "Fluency", P.typos),
        ("remove_punct", "Fluency", P.remove_punct),
        ("drop_adjectives", "Fluency", P.drop_adjectives),
    ]
    common_invariance = [
        ("synonym_adjective", "Invariance", P.synonym_adjective),
        ("contractions", "Invariance", P.contractions),
        ("expansions", "Invariance", P.expansions),
        ("number2words", "Invariance", P.number2words),
    ]
    templates: list[RuleTemplate] = []
    for name, crit, fn in common_fluency + common_invariance:
        templates.append(RuleTemplate(name, crit, "ALL", fn))

    task_specific = {
        "MT": [
            ("drop_phrases", "Adequacy", P.drop_phrases),
            ("add_extra_wrong_information", "Adequacy", P.add_extra_wrong_information),
            ("add_negation", "Adequacy", P.add_negation),
            ("antonym_adjective", "Adequacy", P.antonym_adjective),
            ("repeat_phrases", "Adequacy", P.repeat_phrases),
            ("only_stop", "Adequacy", P.only_stop),
            ("hyponyms", "Informativeness", P.hyponyms),
        ],
        "AS": [
            ("hyponyms", "Informativeness", P.hyponyms),
            ("sentence_reorder", "Coherence", P.sentence_reorder),
            ("repeat_sentences", "Non-redundancy", P.repeat_sentences),
            ("replace_nouns_pronouns", "Clarity", P.replace_nouns_pronouns),
            ("drop_phrases", "Coverage", P.drop_phrases),
            ("change_names", "Relevance", P.change_names),
        ],
        "QG": [
            ("change_question_word", "Answerability", P.change_question_word),
            ("remove_question_word", "Answerability", P.remove_question_word),
            ("change_question_to_assertion", "Answerability", P.change_question_to_assertion),
            ("only_stop", "Answerability", P.only_stop),
            ("mask_words_predict", "Relevance", P.mask_words_predict),
            ("perturb_named_entities_nouns_verbs", "Relevance", P.perturb_named_entities_nouns_verbs),
            ("change_names", "Relevance", P.change_names),
        ],
        "DG": [
            ("negate_previous_utterance", "Making-sense", P.negate_previous_utterance),
            ("nonsensible_reply", "Making-sense", P.nonsensible_reply),
            ("repeat_itself", "Avoid-repetition", P.repeat_itself),
            ("repeat_last_speaker", "Avoid-repetition", P.repeat_last_speaker),
            ("repeat_phrases", "Avoid-repetition", P.repeat_phrases),
            ("sorry_reply", "Listening", P.sorry_reply),
            ("random_text", "Relevance", P.random_text),
            ("generic", "Interesting", P.generic),
        ],
        "IC": [
            ("change_object_order", "Correctness", P.change_object_order),
            ("change_gender", "Correctness", P.change_gender),
            ("change_attributes", "Correctness", P.change_attributes),
            ("replace_object_with_synonym", "Correctness", P.replace_object_with_synonym),
            ("drop_objects", "Thoroughness", P.drop_objects),
            ("repeat_object", "Thoroughness", P.repeat_object),
            ("hyponyms", "Correctness", P.hyponyms),
        ],
        "D2T": [
            ("change_numeric", "Correctness", P.change_numeric),
            ("add_negation", "Correctness", P.add_negation),
            ("antonym_adjective", "Correctness", P.antonym_adjective),
            ("drop_phrases", "Coverage", P.drop_phrases),
            ("repeat_phrases", "Coverage", P.repeat_phrases),
            ("random_text", "Coverage", P.random_text),
            ("change_names", "Relevance", P.change_names),
        ],
    }
    for task, rows in task_specific.items():
        for name, crit, fn in rows:
            templates.append(RuleTemplate(name, crit, task, fn))
    return templates


def select_rule_templates(
    templates: list[RuleTemplate],
    rule_task: str = "all",
    rule_criteria: str = "all",
    rule_template_names: list[str] | None = None,
) -> list[RuleTemplate]:
    task = (rule_task or "all").upper()
    crit = (rule_criteria or "all").lower()
    name_filter = set(rule_template_names or [])
    selected = []
    for t in templates:
        # ── NEW: COMMON_FLUENCY picks only the shared fluency templates ──
        if task == "COMMON_FLUENCY":
            if t.task != "ALL" or t.criteria.lower() != "fluency":
                continue
        else:
            if task != "ALL" and t.task not in {"ALL", task}:
                continue
            if crit != "all" and t.criteria.lower() != crit:
                continue
        # ── end of changed block ──
        if name_filter and t.name not in name_filter:
            continue
        selected.append(t)

    seen = set()
    deduped = []
    for t in selected:
        key = (t.name, t.criteria) if task == "ALL" else (t.name, t.criteria, t.task)
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    return deduped


def _valid_perturbation(original: str, perturbed: str) -> bool:
    if perturbed is None:
        return False
    perturbed = str(perturbed).strip()
    return bool(perturbed) and perturbed != str(original).strip()

_VALID_OUTPUT_MODES = {"all", "first_success", "random_success"}

_WORKER_TEMPLATES: list[RuleTemplate] | None = None
_WORKER_OUTPUT_MODE: str = "all"
_WORKER_MODEL_LABEL: str = RULE_BASED_MODEL_LABEL


def _apply_rule_templates_to_item(
    item: dict,
    templates: list[RuleTemplate],
    output_mode: str,
    model_label: str,
) -> tuple[list[dict], int]:
    """
    Apply selected templates to a single item.

    Returns:
        rows, failures
    """
    original = str(item.get("text", ""))
    successes: list[tuple[RuleTemplate, str]] = []
    failures = 0

    for tmpl in templates:
        try:
            out = tmpl.fn(item)
        except Exception:
            failures += 1
            continue

        if _valid_perturbation(original, out):
            successes.append((tmpl, str(out).strip()))

    if output_mode == "first_success":
        successes = successes[:1]
    elif output_mode == "random_success" and successes:
        successes = [random.choice(successes)]
    elif output_mode != "all":
        raise ValueError(f"Unknown output_mode: {output_mode}")

    rows = []
    for tmpl, text in successes:
        rows.append({
            "perturbation_type": tmpl.name,
            "model": model_label,
            "head_id": item.get("_source_index", item.get("_original_index")),
            "text": text,
            "max_length": item.get("max_length", max(len(original), len(text))),
            "_source_ds": item.get("_source_ds"),
        })

    return rows, failures


def _init_rule_worker(
    corpus_texts: list[str],
    rule_task: str,
    rule_criteria: str,
    rule_template_names: list[str] | None,
    output_mode: str,
    model_label: str,
    random_seed: int | None,
) -> None:
    """
    Initializer run once per worker process.
    Builds the perturber/templates inside each process so bound methods are local
    to that process.
    """
    global _WORKER_TEMPLATES
    global _WORKER_OUTPUT_MODE
    global _WORKER_MODEL_LABEL

    if random_seed is None:
        random.seed(os.getpid() ^ time.time_ns())
    else:
        random.seed(random_seed + os.getpid())

    perturber = EvalEvalRulePerturber(corpus_texts)
    _WORKER_TEMPLATES = select_rule_templates(
        build_rule_templates(perturber),
        rule_task=rule_task,
        rule_criteria=rule_criteria,
        rule_template_names=rule_template_names,
    )

    _WORKER_OUTPUT_MODE = output_mode
    _WORKER_MODEL_LABEL = model_label


def _rule_worker_apply_item(item: dict) -> tuple[list[dict], int]:
    """
    Worker function for ProcessPoolExecutor.
    Must be top-level so it can be pickled by multiprocessing.
    """
    if _WORKER_TEMPLATES is None:
        raise RuntimeError("Worker templates were not initialized.")

    return _apply_rule_templates_to_item(
        item,
        _WORKER_TEMPLATES,
        _WORKER_OUTPUT_MODE,
        _WORKER_MODEL_LABEL,
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
) -> list[dict]:
    """
    Apply EvalEval-style non-LLM perturbations and return rows with the same
    schema as the vLLM perturbation path:

      perturbation_type, model, head_id, text, max_length, _source_ds

    output_mode:
      - all: one row per successful template per item
      - first_success: at most one row per item
      - random_success: at most one random successful row per item

    Parallelism:
      - n_jobs=None: use all available CPUs
      - n_jobs=1: run sequentially
      - n_jobs=N: use N worker processes
    """
    if output_mode not in _VALID_OUTPUT_MODES:
        raise ValueError(f"Unknown output_mode: {output_mode}")

    corpus_texts = [x.get("text", "") for x in ds_items]

    # Build once in parent for validation and for sequential mode.
    perturber = EvalEvalRulePerturber(corpus_texts)
    templates = select_rule_templates(
        build_rule_templates(perturber),
        rule_task=rule_task,
        rule_criteria=rule_criteria,
        rule_template_names=rule_template_names,
    )

    if not templates:
        raise ValueError(
            "No rule-based templates selected. "
            "Check --rule-task/--rule-criteria/--rule-templates."
        )

    if not ds_items:
        print(
            f"Rule-based perturbation: selected {len(templates)} template(s), "
            f"produced 0 successful rows from 0 source item(s); "
            f"0 template application(s) raised and were skipped."
        )
        return []

    if n_jobs is None:
        n_jobs_effective = os.cpu_count() or 1
    else:
        n_jobs_effective = max(1, int(n_jobs))

    n_jobs_effective = min(n_jobs_effective, len(ds_items))

    if chunksize is None:
        chunksize = max(1, len(ds_items) // max(1, n_jobs_effective * 4))
    else:
        chunksize = max(1, int(chunksize))

    rows: list[dict] = []
    failures = 0

    # Sequential path. Useful for debugging and avoids multiprocessing overhead
    # on tiny datasets.
    if n_jobs_effective == 1:
        if random_seed is not None:
            random.seed(random_seed)

        for item in tqdm(ds_items, desc="Generating perturbations..."):
            item_rows, item_failures = _apply_rule_templates_to_item(
                item,
                templates,
                output_mode,
                model_label,
            )
            rows.extend(item_rows)
            failures += item_failures

    # Parallel path.
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs_effective,
            initializer=_init_rule_worker,
            initargs=(
                corpus_texts,
                rule_task,
                rule_criteria,
                rule_template_names,
                output_mode,
                model_label,
                random_seed,
            ),
        ) as executor:
            results_iter = executor.map(
                _rule_worker_apply_item,
                ds_items,
                chunksize=chunksize,
            )

            for item_rows, item_failures in tqdm(
                results_iter,
                total=len(ds_items),
                desc=f"Generating perturbations with {n_jobs_effective} workers...",
            ):
                rows.extend(item_rows)
                failures += item_failures

    print(
        f"Rule-based perturbation: selected {len(templates)} template(s), "
        f"produced {len(rows)} successful rows from {len(ds_items)} source item(s); "
        f"{failures} template application(s) raised and were skipped. "
        f"Used {n_jobs_effective} worker process(es)."
    )

    return rows
