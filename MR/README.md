This is the Movie Review(MR)[<a href="http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz">DownLoad</a>] dataset. It contains 10662 sentences, half of which is positive instances and others are negative ones. We split the dataset into 10 subsets randomly and use cross-validation policy to train our model.

Simply run the command `python parserMR.py` in this folder to generate file `data`. The format of this file is shown as follows:
```
list[0]: list, each entry is a dict: {'label': category, 'text': raw sentence, 'setLabel': subset ID, 'len': length of this sentence}.
list[1]: dict, each entry is a word and its corresponding document frequency.
```
