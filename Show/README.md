This module aims to show the performance of a model with help of open-source framework <a href='http://www.highcharts.com/'>highcharts</a>.

The input file is list dumped by python-cPickle. Its format is shown as follows:

```
list[0]:records, a list; each entry is a dict representing a record {'x':epoch,'trainAcc':acc on train set,'validateAcc':acc on validate set,'testAcc': acc on test set}
list[1]:results, a dict {'minError':minimum error rate on test set,'finalAcc':final accuracy,'bestValAcc':best accuracy on validate set}
```
