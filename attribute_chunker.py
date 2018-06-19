import collections

import spacy
from spacy import displacy

Attribute = collections.namedtuple("Attribute", ["attribute", "description"])


class AttributeChunker:
    def __init__(self, exclutions=None):
        if exclutions is None:
            exclutions = {"bird"}
        self.nlp = spacy.load("en_core_web_sm")

    def chunk(self, text):
        doc = self.nlp(text)
        attrs = []
        for possible_subject in doc:
            if (
                possible_subject.dep == spacy.symbols.amod
                and possible_subject.head.pos == spacy.symbols.NOUN
            ):
                attrs.append(
                    Attribute(possible_subject.head.text, possible_subject.text)
                )
        return attrs


class CounterFactualExplanations:
    # Have to display: same nouns and different adjectives
    # Show others anyway (not common)
    
    def __init__(self, explanations):
        self.explanations = explanations
        self.ch = AttributeChunker()
        self.chunks = {}
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

    def generate_cf(self, expl, other_expl):
        chunks = self.to_dict(self.ch.chunk(expl))
        other_ch = self.to_dict(self.ch.chunk(other_expl))

        cf_expl = "because this bird has "
        # Case 1: Overlapping nouns
        nouns = set(chunks.keys())
        other_nouns = set(other_ch.keys())
        overlap = nouns.intersection(other_nouns)

        added_attr = []
        added_attr_other = []

        if len(overlap) > 0:
            for noun in nouns:
                # see if the adjectives are diffent
                diff_adj = set(other_ch[noun]) - set(chunks[noun])

                if len(diff_adj) == 0:
                    continue
                
                added_attr.extend([Attribute(noun, adj) for adj in chunks[noun]])
                added_attr_other.extend([Attribute(noun, adj) for adj in diff_adj])
                cf_expl += "a {} {} and not a ".format(" ".join(chunks[noun]), noun)
                cf_expl += "{} {} and ".format(" ".join(diff_adj), noun)
        
        cf_expl.rstrip("and ")
        # Case 2: Other attributes


        return cf_expl
        


        

if __name__ == "__main__":
    chunker = AttributeChunker()
    attrs = chunker.chunk(
        "this bird has a long grey head and a downward curving beak with brown feathers covering the rest of its body"
    )

    from data_api import DataApi

    data = DataApi()
    explanations = {img["id"]: img["caption"] for img in data.data}
    # print(explanations)
    # doc = chunker.chunk("this bird has a black belly and crown")
    cfe = CounterFactualExplanations(explanations)

    # for img_id in cfe.cf_explanations:
    #     print(img_id, explanations[img_id])
    #     for expl in cfe.cf_explanations[img_id]:
    #         print("\t", explanations[expl])
    #     print()

    cf = cfe.generate_cf("this bird has a brown crown and a black belly", "this bird has a pointy brown crown and a white belly")  
    print(cf)