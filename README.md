# Text-Summarizer

## Introduction
We will be creating a Natural Language Processing Transformer Model that performs Abstractive Text Summarization. Given a document, article, or body of text, the model will return a concise summary of the provided text. Since we are dealing with sequential data, we will be using a transformer provided by Pytorch, with a bidirectional encoder and an autoregressive decoder, and fine-tuning it to enhance performance as required for the task. 

## Model
### Model Figure

### Model Parameters

### Model Examples

## Data
### Data Sources
Our dataset is a subset of the "Gigaword" dataset, acquired from the Hugging Face Dataset library. Gigaword is the largest dataset available to be used for the task of summarization, with 3.8 million training samples, 189k validation samples and 1951 test samples. 

### Data Split
As per our initial idea, we planned to combine and use 3 complete datasets from the Hugging Face dataset library, "Gigaword", "Multi_news", and "CNN_dailymail",  but due to the lack of computational resources, we quickly realized that is not viable and we have to subset the data. We decided to use a subset of the Gigaword Dataset's training samples, as it is the largest of the 3, to ensure variability of characteristics and context in our data to help the model generalize and learn appropriately.


**Gigaword:**
Train: 3803957 (95%), Validation: 189651(4.75%), Test: 1951(0.25%)

**Subset Dataset:**
Train: 114119 (85%), Validation: 18965 (14%), Test: 1951(1%)

Although the percentage of test samples looks quite small, given the number of observations we believe it is sufficient to estimate the performance the model well. 

### Data Summary
To accurately interpret our results, we collected summary statistics on our dataset. 


### Data Transformation
Since our data is Text. To prepare our data for input to the model, we used the pretrained GPT2 Autotokenizer to convert the text into a sequence of tokens. 

```python

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2", special_tokens=[['BOS']])
tokenizer.pad_token = tokenizer.eos_token
inp_lst = []

def preprocess_function(examples):
    inputs = [doc for doc in examples['document']]
    model_inputs = tokenizer(inputs, padding=True, return_tensors="pt")
    labels = tokenizer(text_target=examples["summary"], padding=True, return_tensors='pt')
    model_inputs["labels"] = labels["input_ids"]
    inp_lst.append((model_inputs["input_ids"], labels["input_ids"]))
    return model_inputs
```

The tokenization of an example sentence:

```python
sents = "He was able to train it without any problems."
ids =  tokenizer(sents, padding=True, return_tensors='pt')
print(ids) 
# tensor([[1544,  373, 1498,  284, 4512,  340, 1231,  597, 2761,   13]])
```


Followed by a glove embedding layer to create numerical representation of the tokens and assemble them into tensors, using Glove 6B with dimension 50 (6 Billion tokens and 50 Features).


```python
from torchtext.vocab import GloVe
glove = GloVe(name='6B', dim=50)

def glove_embed(data):
    max_length = 0 
    for i in range(len(data)):
        max_length = max(max_length, len(data[i]))
    tensor = torch.empty(len(data), max_length, EMB_DIM)
    for i in range(len(data)):
        words = tokenizer.convert_ids_to_tokens(data[i])
        emb = glove.get_vecs_by_tokens(words, lower_case_backup=True)
        tensor[i] = emb
    return tensor

```



## Training
### Training Curve

### Hyperparameter Tuning

## Results

## Ethical Considerations

## Authors
