
import pandas as pd

annotations = pd.read_table("/home/cz/czf/retrieval/reasoning/datasets/results_20130124.token", sep='\t', header=None,
                            names=['image', 'caption'])

for i in range(len(annotations['caption'])):
    if annotations['caption'][i] == 'Group gathered to go snowmobiling .':
        print(annotations['image'][i])