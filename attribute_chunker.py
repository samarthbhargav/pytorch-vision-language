import collections

import spacy
from spacy import displacy

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
        # self.explanations = explanations
        # print("Precomputing chunks")
        # for count, (img_id, expl) in enumerate(self.explanations.items()):
        #     self.chunks[img_id] = self.to_dict(self.ch.chunk(expl))
        #     if count % 100 == 0:
        #         print("Count: {}".format(count))

        # self.cf_explanations = {}
        # print("Computing CF explanations")
        # self._pre_compute()

    def to_dict(self, chunks):
        n = collections.defaultdict(list)
        for ch in chunks:
            n[ch.attribute].append(ch.description)
        return n

    def _pre_compute(self):
        for count, (img_id, chunks) in enumerate(self.chunks.items()):
            nouns = set(chunks.keys())
            accepted = []
            for other_id, other_chunks in self.chunks.items():
                common_nouns = nouns.intersection(other_chunks.keys())
                if len(common_nouns) == 0:
                    continue
                diff_count = 0
                for cn in common_nouns:
                    # TODO: this is wrong. use in op
                    if chunks[cn] != other_chunks[cn]:
                        diff_count += 1
                if diff_count > 0:
                    accepted.append((other_id, diff_count))

            accepted.sort(key=lambda _: -_[1])
            self.cf_explanations[img_id] = [a[0] for a in accepted]
            if count % 100 == 0:
                print("Count: {}".format(count))

            if count == 0:
                break

    def generate_cf(self, expl, other_expl, addtn_limit = 3):
        chunks = self.to_dict(self.ch.chunk(expl))
        other_ch = self.to_dict(self.ch.chunk(other_expl))

        cf_expl = "because this bird "
        # Case 1: Overlapping nouns
        nouns = set(chunks.keys())
        other_nouns = set(other_ch.keys())
        overlap = nouns.intersection(other_nouns)

        added_attr = []
        added_attr_other = []

        seen_nouns = set()
        if len(overlap) > 0:
            for noun in nouns:
                if len(added_attr) + len(added_attr_other) >= addtn_limit:
                    break
                # see if the adjectives are diffent
                diff_adj = set(other_ch[noun]) - set(chunks[noun])

                if len(diff_adj) == 0:
                    continue

                added_attr.extend([Attribute(noun, adj) for adj in chunks[noun]])
                added_attr_other.extend([Attribute(noun, adj) for adj in diff_adj])
                cf_expl += "has a {} {} and not a ".format(" ".join(chunks[noun]), noun)
                cf_expl += "{} {} and ".format(" ".join(diff_adj), noun)

                seen_nouns.add(noun)

        cf_expl = cf_expl.strip().strip("and").strip()

        # Case 2: Other attributes
        missing_nouns = other_nouns - seen_nouns 
        if len(missing_nouns) > 0:
            cf_expl += " and "
            for noun in missing_nouns:
                if len(added_attr) + len(added_attr_other) >= addtn_limit:
                    break
                added_attr_other.extend([Attribute(noun, adj) for adj in diff_adj])
                cf_expl += "doesn't have a {} {}".format(" ".join(other_ch[noun]), noun)
        
        cf_expl = cf_expl.strip().strip("and").strip()

        return cf_expl, added_attr, added_attr_other


if __name__ == "__main__":
    chunker = AttributeChunker()
    attrs = chunker.chunk(
        "this bird has a long grey head and a downward curving beak with brown feathers covering the rest of its body"
    )

    from data_api import DataApi

    data = DataApi()
    explanations = {img["id"]: img["caption"] for img in data.data}

    cfe = CounterFactualExplanations(explanations)

    cf, added, added_other = cfe.generate_cf(
        "this bird has a brown crown and a black belly",
        "this bird has a pointy brown crown and a white belly and yellow beak"
    , addtn_limit=10)
    print(cf)
    print(added)
    print(added_other)
