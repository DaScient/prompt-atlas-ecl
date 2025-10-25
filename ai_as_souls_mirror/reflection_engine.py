from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json, re
from collections import Counter

_WORD = re.compile(r"[a-zA-Z']+")

class ReflectionEngine:
    def __init__(self, config_path: str | Path, use_llm: bool = False):
        self.cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.archetypes = self.cfg["archetypes"]
        self.sentiment = self.cfg["sentiment"]
        self.arch_kw = {a["id"]: set(w.lower() for w in a["keywords"]) for a in self.archetypes}
        self.arch_meta = {a["id"]: a for a in self.archetypes}
        self.sent_lex = {k: set(v) for k,v in self.sentiment.items()}
        self.use_llm = use_llm

    def tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in _WORD.findall(text)]

    def score_archetypes(self, tokens: List[str]) -> Tuple[str, float, Dict[str, int]]:
        counts = {aid: 0 for aid in self.arch_kw}
        bag = Counter(tokens)
        for aid, kws in self.arch_kw.items():
            for k in kws: counts[aid] += bag.get(k, 0)
        best_id = max(counts, key=lambda k: counts[k])
        total = sum(counts.values()) or 1
        conf = counts[best_id] / total
        return best_id, conf, counts

    def score_sentiment(self, tokens: List[str]) -> Tuple[str, float]:
        pos = sum(1 for t in tokens if t in self.sent_lex["positive"])
        neg = sum(1 for t in tokens if t in self.sent_lex["negative"])
        neu = sum(1 for t in tokens if t in self.sent_lex["neutral"])
        total = max(pos + neg + neu, 1)
        if pos > neg and pos >= neu: return "positive", pos/total
        if neg > pos and neg >= neu: return "negative", neg/total
        return "neutral", neu/total

    def reflect_llm(self, text: str) -> Optional[Dict]:
        return None

    def reflect(self, text: str) -> Dict:
        tokens = self.tokenize(text)
        arch_id, arch_conf, arch_counts = self.score_archetypes(tokens)
        ar = self.arch_meta[arch_id]
        sent_label, sent_conf = self.score_sentiment(tokens)
        lines = [f"{ar['name']} speaks.", ar['affirmation']]
        if sent_label == "negative": lines.append("Your language suggests strain. Slow your breath and widen the frame.")
        elif sent_label == "positive": lines.append("There’s traction here; let curiosity carry momentum.")
        else: lines.append("You’re mapping the terrain—stay with the details that matter.")
        return {
           "archetype":{"id":arch_id,"name":ar["name"],"color":ar["color"],"confidence":round(arch_conf,3)},
           "sentiment":{"label":sent_label,"confidence":round(sent_conf,3)},
           "mirror_text":" ".join(lines),
           "debug":{"archetype_counts":arch_counts}
        }
