import sys
import pickle

f = open(sys.argv[1])
pronouns = "i|me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours|yourself|yourselves|he|his|him|himself|she|her|hers|herself|it|its|itself|they|them|their|themselves|that|this|these|those|what|whatever|which|whichever|who|whoever|whom|whomever|whose".split("|")


tokens = f.readlines()
print(len(tokens))
prn_2_idx = {}
idx_2_prn = {}

for i,t in enumerate(tokens):
	word = t.split()
	if word[0] in pronouns:
		prn_2_idx[word[0]] = i+4
		idx_2_prn[i+4] = word[0]

pickle.dump(prn_2_idx, open("pronoun_to_index.pkl", 'wb'))
pickle.dump(idx_2_prn, open("index_to_pronoun.pkl", 'wb'))
	


