import spacy as sp
from spacy.matcher import PhraseMatcher
import pandas as pd

nlp = sp.load("en_core_web_lg")

matcher = PhraseMatcher(nlp.vocab)
matcher.add("Ireland", [nlp("ireland"),nlp("irish")])
matcher.add("EU", [nlp("arack Obama"),nlp("Baracko")])
matcher.add("World", [nlp("Scatter plot"), nlp("scatterplot"),nlp("scat")])
matcher.add("Who",[nlp("Scatter plot"), nlp("scatterplot"),nlp("scat")])
doc = nlp("Ireland is in ireland which is irish")
matches = matcher(doc)
matchesS = [nlp.vocab.strings[s[0]] for s in matches]
print(nlp.vocab.strings)
print(matchesS)
target = "Ireland"
if target in matchesS:
    print(target," Found!")
print(matches)

# #initilize the matcher with a shared vocab
# matched = PhraseMatcher(nlp.vocab)
# #create the list of words to match
# plot_types = ['scatter plot','bar chart','histogram',"bar"]
# #obtain doc object for each word in the list and store it in a list
# patterns = [nlp(plots) for plots in plot_types]
# #add the pattern to the matcher
# matched.add("plot_patern", patterns)
# #process some text
# doc = nlp("in this document there seems to be a scatter plot and a bar chart")
# matching = matched(doc)
# print(matching)
# for match_id, start, end in matching:
#  span = doc[start:end]
#  print(span.text)