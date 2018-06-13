import spacy

nlp = spacy.load('en_core_web_sm')

def read_data(filename):
    """ Reads in the data to be processed.
        Input: Text file containing explanations.
        Output: Nested list where each sublist represents a sentence. """
    with open(filename, "r") as filename:
            data = filename.readlines()
    return data

def get_NP(sent):
    """ Extracts noun phrases from text.
        Input: List representing a sentence.
        Output: List of strings representing the noun phrases detected
                in the input sentence."""
    noun_chunks = []
    token_sent = nlp(sent)
    current_nps = []
    for np in token_sent.noun_chunks:
        current_nps.append(np)
    if current_nps:
        noun_chunks.append(current_nps)

    return [str(x) for sublist in noun_chunks for x in sublist]

def gen_counterfactual_expl(label_pos, label_neg, factual_explanations):
    conterfactual_expl = []

    # Retrieve Noun Phrases
    NP_pos = get_NP(factual_explanations[0])
    NP_neg = get_NP(factual_explanations[1])

    counterfactual_NPs = []
    for np in NP_pos:
        if np not in NP_neg:
            counterfactual_NPs.append(np)

    print(NP_pos)
    start = " This is not a "
    connector = " because it does not have "
    EOS = "."



    conterfactual_expl = start + label_neg + connector + \
                         counterfactual_NPs[1] + EOS\



    return conterfactual_expl

# Read in explanations

explanations = read_data('bird_example1')

label_neg = "Crested Auklet"
label_pos = "Red Faced Cormorant"

# Generate counterfactual explanation
print(gen_counterfactual_expl(label_pos, label_neg, explanations))





