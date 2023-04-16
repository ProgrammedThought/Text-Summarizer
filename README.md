# Text-Summarizer

## Introduction
We will be creating a Natural Language Processing Transformer Model that performs Abstractive Text Summarization. Given a document, article, or body of text, the model will return a concise summary of the provided text. Since we are dealing with sequential data, we will be using a transformer provided by Pytorch, with a bidirectional encoder and an autoregressive decoder, and fine-tuning it to enhance performance as required for the task. 

## Model
### Model Figure

### Model Parameters

### Model Examples

## Data
### Data Sources
Our dataset is a subset of the "Gigaword" dataset, acquired from the Hugging Face Dataset library. Gigaword is the largest dataset available to be used for the summarization task, with 3.8 million training samples, 189k validation samples and 1951 test samples. 

### Data Split
As per our initial idea, we planned to combine and use 3 complete datasets from the Hugging Face dataset library, "Gigaword", "Multi_news", and "CNN_dailymail",  but due to the lack of computational resources, we quickly realized that is not viable and we have to subset the data. We decided to use a subset of the Gigaword Dataset, as it is the largest of the 3, to ensure variability of characteristics and context in our data to help the model generalize and learn appropriately.


**Gigaword:**
Train: 3803957 (95%), Validation: 189651(4.75%), Test: 1951(0.25%)

**Subset Dataset:**
Train: 76079 (91%), Validation: 5690 (6%), Test: 1951(3%)

We believe this subset accurately represents the dataset and has a sufficient number of samples for the model to be trained and perform well. 

### Data Summary
To accurately interpret our results, we collected summary statistics on our dataset. 

#### Training Data
There are 76079 training samples in our dataset. 
The average document length is: 31.42806819227382 words, and the average summary length is:   8.206430158125107. There are 42986 unique words, and a total of 2391016 words, in the documents and 25275 unique words, and a total of 624337 words in the summary of the training set. The most commonly used words and their frequencies are as follows:


| Word (Document) | Frequency (%) | Word (Summary)     | Frequency (%) |
| --------------- | ------------- | ------------------ | ------------- |
| the             | 4.87          | in                 | 3.29          |
| .               | 3.18          | to                 | 3.27          |
| ,               | 3.07          | of                 | 1.62          |
| a               | 2.75          | for                | 1.59          |
| of              | 2.56          | on                 | 1.32          |
| in              | 2.39          | 's                 | 1.30          |
| to              | 2.35          | us                 | 0.97          |
| on              | 1.60          | unk                | 0.61          |
| said            | 1.37          | percent            | 0.60          |
| and             | 1.29          | as                 | 0.58          |


#### Validation Data
There are 5690 validation samples in our dataset. 
The average document length is: 31.138137082601055 words, and the average summary length is: 7.9195079086116 words, in the validation set. There are 14086 unique words, and a total of 177176 words in the documents, and 7873 unique words, and a total of 45062 words in the summary of the validation set. The most commonly used words in the validation set and their frequencies are as follows:

| Word (Document) | Frequency (%) | Word (Summary)     | Frequency (%) |
| --------------- | ------------- | ------------------ | ------------- |
| the             | 4.83          | to                 | 3.35          |
| .               | 3.21          | in                 | 2.98          |
| ,               | 3.05          | for                | 1.78          |
| a               | 2.65          | of                 | 1.49          |
| of              | 2.49          | 's                 | 1.44          |
| to              | 2.44          | on                 | 1.18          |
| in              | 2.25          | us                 | 1.07          |
| on              | 1.68          | unk                | 0.84          |
| 's              | 1.36          | over               | 0.59          |
| said            | 1.32          | ##                 | 0.53          |

#### Test Data
There are 1951 validation samples in our dataset. 
The average document length is: 29.69656586365966 words, and the average summary length is: 8.791901588925754 words, in the test set. There are 9445 unique words, and a total of 57938 words in the documents, and 5096 unique words, and a total of 17153 words in the summary of the test set. The most commonly used words in the test set and their frequencies are as follows:

| Word (Document) | Frequency (%) | Word (Summary)     | Frequency (%) |
| --------------- | ------------- | ------------------ | ------------- |
| the             | 4.96          | unk                | 4.89          |
| ,               | 3.52          | to                 | 2.83          |
| .               | 3.21          | in                 | 2.81          |
| a               | 2.63          | :                  | 2.10          |
| of              | 2.54          | of                 | 1.61          |
| to              | 2.30          | for                | 1.31          |
| in              | 2.23          | 's                 | 1.07          |
| and             | 1.61          | on                 | 0.99          |
| on              | 1.37          | by                 | 0.89          |
| 's              | 1.18          | with               | 0.73          |


### Data Transformation
Since our data is Text. To prepare our data for input to the model, we used the pretrained BertTokenizer ('bert-base-uncased') to convert the text into a sequence of tokens. Followed by an embedding layer to create numerical representation of the tokens and assemble them into tensors, with the shape (30522, 256).

```python

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_function(examples):
    model_inputs = tokenizer(examples['document'], padding='max_length', \
                             max_length=75, add_special_tokens=True)
    labels = tokenizer(text_target=examples["summary"], padding='max_length', \
                       max_length=70, add_special_tokens=True)
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["labels_mask"] = labels["attention_mask"]
    return model_inputs
    
```

The tokenization and embedding of an example sentence:

```python
example = {'document': "a powerful rally tuesday helped wall street recoup some losses from monday 's rout , 
            amid renewed hope for us passage of a financial rescue package .",
            'summary': 'big rally helps wall street recoup part of record plunge'}
ex_token = preprocess_function(example)

print(ex_token['input_ids']) # [101, 1037, 3928, 8320, 9857, 3271, 2813, 2395, 28667, 7140, 2361, 2070, 6409, 2013, 6928, 1005, 1055, 20996, 4904, 1010, 13463, 9100,   3246, 2005, 2149, 6019, 1997, 1037, 3361, 5343, 7427, 1012, 102]
print(ex_token['labels']) #[101, 2502, 8320, 7126, 2813, 2395, 28667, 7140, 2361, 2112, 1997, 2501, 25912, 102]


example_emb = nn.Embedding(30522, 256)
inp_emb = example_emb(torch.tensor(ex_token['input_ids']))
print(inp_emb) # tensor([[ 0.9791,  0.8341,  0.0265,  ...,  0.1341,  0.3989,  2.0854],
                        #[-0.1030,  2.0953, -1.2685,  ..., -1.3507,  1.0663, -0.5291],
                        #[ 2.4593, -0.0891, -0.9599,  ..., -0.4411,  1.6811, -0.5325],
                        #...,
                        #[ 0.2698, -1.0205,  0.5904,  ...,  0.5692, -0.6299,  0.2383],
                        #[ 0.2698, -1.0205,  0.5904,  ...,  0.5692, -0.6299,  0.2383],
                        #[ 0.2698, -1.0205,  0.5904,  ...,  0.5692, -0.6299,  0.2383]])

```



## Training
### Training Curve

### Hyperparameter Tuning

## Results

## Ethical Considerations

## Authors
