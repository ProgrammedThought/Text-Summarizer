# Text-Summarizer

## Introduction
We will be creating a Natural Language Processing Transformer Model that performs Abstractive Text Summarization. Given a document, article, or body of text, the model will return a concise summary of the provided text. Since we are dealing with sequential data, we will be using a transformer provided by Pytorch, with a bidirectional encoder and an autoregressive decoder, and fine-tuning it to enhance performance as required for the task. 

## Model
### Model Figure

### Model Parameters

### Model Examples

## Data
### Data Sources
Our data is comprised of multiple datasets acquired from Hugging Face. We collected and combined the following datasets for the model: "Gigaword", "Multi_news", and "CNN_dailymail". This was done to ensure variability of characteristics, length, and context in our data to help the model generalize and learn appropriately. 
### Data Split
The datasets originally available on Hugging Face are already split into training, validation, and test sets with the following splits:

**Gigaword:**
Train: 3803957 (95%), Validation: 189651(4.75%), Test: 1951(0.25%)

**Multi_news:**
Train: 44972 (80%), Validation: 5622(10%), Test: 5633(10%)

**CNN_dailymail:**
Train: 287113(92%), Validation: 13368(4.4%), Test: 11490(3.6%)

As per our initial idea, we planned to combine and use the 3 complete datasets, but due to the lack of computational resources, we quickly realized that is not viable and we have to subset the data. We decided to use the complete Multi_news and CNN_dailymail datasets, with a subset of training samples from the Gigaword dataset to ensure a practical split. Our final data split looked like this: 

**Dataset:**
Train: 1000000 (81%), Validation: 208641(17%), Test: 19074(2%)

Although the percentage of test samples looks quite small, given the number of observations we believe it is sufficient to estimate the performance the model well. 

### Data Summary


### Data Transformation
The datasets we used were acquired directly from the Hugging Face Datasets library. The Gigaword and Multi_news datasets have two string fields, document and summary. The CNN_dailymail has 3 string fields: id, article, and highlights, so prior to the concatenation of datasets, we renamed the 'article' column to 'document', the 'highlights' column to 'summary', and removed the 'id' column from the CNN_dailymail dataset. Then the concatenation of the 3 datasets was performed, into the predetermined split of Training, Validation, and Test sets.




## Training
### Training Curve

### Hyperparameter Tuning

## Results

## Ethical Considerations

## Authors
