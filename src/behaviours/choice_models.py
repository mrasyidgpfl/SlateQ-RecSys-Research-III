from recsim_ng.lib.tensorflow import selectors

def multinomial_choice(logits):
    return selectors.MultinomialLogitChoiceModel(logits)