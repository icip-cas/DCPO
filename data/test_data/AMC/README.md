---
dataset_info:
  features:
  - name: problem
    dtype: string
  - name: solution
    dtype: string
  - name: answer
    dtype: string
  - name: url
    dtype: string
  - name: year
    dtype: int64
  splits:
  - name: test
    num_bytes: 158599
    num_examples: 134
  - name: amc22
    num_bytes: 50893.70895522388
    num_examples: 43
  - name: amc23
    num_bytes: 54444.432835820895
    num_examples: 46
  - name: amc24
    num_bytes: 53260.85820895522
    num_examples: 45
  download_size: 182617
  dataset_size: 317198.0
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
  - split: amc22
    path: data/amc22-*
  - split: amc23
    path: data/amc23-*
  - split: amc24
    path: data/amc24-*
---
