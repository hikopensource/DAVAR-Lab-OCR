The NER dataset CoNLL2003(https://www.clips.uantwerpen.be/conll2003/ner/) and Resume(https://github.com/jiesutd/LatticeLSTM).
## Data Format
We adopt BIEOS scheme as the ground truth format.
```Annotation example
{'0':
	{'nlp_ann': {'tokens': [['SOCCER',
							'-',
							'JAPAN',
							'GET',
							'LUCKY',
							'WIN',
							',',
							'CHINA',
							'IN',
							'SURPRISE',
							'DEFEAT',
							'.']],
						  'tokens_labels': [[['O'],
							['O'],
							['S-LOC'],
							['O'],
							['O'],
							['O'],
							['O'],
							['S-PER'],
							['O'],
							['O'],
							['O'],
							['O']]]
				}
	}
	...
}
```
The `tokens` and `tokens_labels` are two lists of length 1, and tokens[0] and tokens_labels[0] must be of equal length. The `tokens_labels` represents the label of tokens, using BIOES sheme.
