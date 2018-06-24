import collections

import spacy
from spacy import displacy
import re

Attribute = collections.namedtuple("Attribute", ["attribute", "description", "position"])


class AttributeChunker:
    def __init__(self, exclutions=None):
        if exclutions is None:
            exclutions = {"bird"}
        self.nlp = spacy.load("en_core_web_sm")

    def chunk(self, text):
        doc = self.nlp(text)
        attrs = []
        for index, possible_subject in enumerate(doc):
            if (
                possible_subject.dep == spacy.symbols.amod
                and possible_subject.head.pos == spacy.symbols.NOUN
            ):
                attrs.append(
                    Attribute(possible_subject.head.text, possible_subject.text, index)
                )
        return attrs


class CounterFactualGenerator:
    # Have to display: same nouns and different adjectives
    # Show others anyway (not common)

    def __init__(self):
        
        self.ch = AttributeChunker()
        self.chunks = {}

    def to_dict(self, chunks):
        n = collections.defaultdict(list)
        for ch in chunks:
            n[ch.attribute].append(ch.description)
        return n

    def generate(self, expl, other_expl, addtn_limit = 3):
        chunks = self.to_dict(self.ch.chunk(expl))
        other_ch = self.to_dict(self.ch.chunk(other_expl))

        cf_expl = "this bird "
        # Case 1: Overlapping nouns
        nouns = set(chunks.keys())
        other_nouns = set(other_ch.keys())
        overlap = nouns.intersection(other_nouns)

        print(chunks, other_ch)

        added_attr = []
        added_attr_other = []

        seen_nouns = set()
        added_first = False
        if len(overlap) > 0:
            for noun in nouns:
                if len(added_attr) + len(added_attr_other) >= addtn_limit:
                    break
                # see if the adjectives are diffent
                diff_adj = set(other_ch[noun]) - set(chunks[noun])
                if len(diff_adj) == 0:
                    continue
                added_first = True
                added_attr.extend([Attribute(noun, adj, -1) for adj in chunks[noun]])
                added_attr_other.extend([Attribute(noun, adj, -1) for adj in diff_adj])
                cf_expl += "has a {} {} and not a ".format(" ".join(chunks[noun]), noun)
                cf_expl += "{} {} and ".format(" ".join(diff_adj), noun)

                seen_nouns.add(noun)
        
        cf_expl = cf_expl.strip()
        cf_expl = re.sub("(\s)*and$", "\\1", cf_expl)
        # Case 2: Other attributes
        missing_nouns = other_nouns - seen_nouns 
        if len(missing_nouns) > 0:
            cf_expl += " and " if added_first else " "
            for noun in missing_nouns:
                if len(added_attr) + len(added_attr_other) >= addtn_limit:
                    break
                missing_ = set(other_ch[noun]) - set(chunks[noun])
                if len(missing_) == 0:
                    continue
                added_attr_other.extend([Attribute(noun, adj, -1) for adj in missing_])
                cf_expl += "doesn't have a {} {} and ".format(" ".join(missing_), noun)
        
        cf_expl = cf_expl.strip()
        cf_expl = re.sub("(\s)*and$", "\\1", cf_expl)

        return cf_expl, added_attr, added_attr_other


if __name__ == "__main__":
    chunker = AttributeChunker()
    cfe = CounterFactualGenerator()

    cf, added, added_other = cfe.generate(
         "this bird is black with white on its wings and has a long pointy beak",
        "this bird has a yellow belly and breast with a short pointy bill", addtn_limit=10)
    print(cf)
    print(added)
    print(added_other)
