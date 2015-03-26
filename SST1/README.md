This is the folder of Stanford Sentiment Treebank[<a href="http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip">DownLoad</a>] dataset, which contains 11855 sentences. There are five categories in total: very positive, positive, neutral, negative, very negative.

Just run the command `python parseSST.py` to generate the file `data`. This file contains a list, the format is shown as follows: 
```
list[0]: list, each entry represents a sentence, the form of which is {'label': category, 'text': sentences, 'setLabel': train/dev/test set, 'len': length of this sentence}.
list[1]: dict, each entry is a word and the corresponding value is the document frequency of this word.
list[2]: dict, config information, 'all'/'train'/'test'/'dev': all/train/test/dev subsets ID, 'cross': whether or not to use cross-validation
```
