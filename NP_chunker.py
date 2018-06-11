import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm')

def read_data(filename):
    """ Reads in the data to be processed.
        Input: Text file containing explanations.
        Output: Nested list where each sublist represents a sentence. """
    with open(filename, "r") as filename:
            data = filename.readlines()
    return data

def get_NP(text):
    """ Extracts noun phrases from text.
        Input: Nested list where each sublist represents a sentence.
        Output: Nested list where each sublist represents the noun phrases detected
                in the input sentence at the same index."""
    noun_chunks = []
    for i, sent in enumerate(text):
        token_sent = nlp(sent)
        current_nps = []
        for np in token_sent.noun_chunks:
            current_nps.append(np)
        if current_nps:
            noun_chunks.append(current_nps)

    return noun_chunks

def gen_counterfactual_expl(text, noun_chunks, label):



    return conterfactual_expl

# Read in explanations
explanations = read_data('bird_example')

# Retrieve Noun Phrases
NPs = get_NP(explanations)
print(NPs[0])
print(NPs[1])

a = [str(x) for x in NPs[0]]
b = [str(x) for x in NPs[1]]

for np in a:
    if np not in b:
        print(str(np))



