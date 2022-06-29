# Description
Your task is to create an HTTP service for identifying the country an address belongs to. 
You can assume that the address will always contain the city.

# Questions
- What is the problem of the method that index each city to its corresponding country?
- How can we improve the performance of the method?
- Can I assmue that the service will only be used in the European Union?

# Useful Resources
- https://results.openaddresses.io/
  - Data source contains addresses and their corresponding countries.
- https://towardsdatascience.com/can-we-predict-country-of-origin-just-using-name-fb5126e2c128
  - Translate name into english
  - Prepare a dataset with name as input and country as output
  - Train a XLNet model for classification
- https://developers.arcgis.com/python/samples/identifying-country-names-from-incomplete-house-addresses/
  - Identifying country names from incomplete house addresses
  - Train a XLM-roberta model for country identification
  - Get misclassified records and anaylze problems
- https://arxiv.org/pdf/2007.03020.pdf
  - Discover some preprocessing techniques for addresses
  - Pre-trained a Roberta model then fine-tune it for country identification
- https://medium.com/@albarrentine/statistical-nlp-on-openstreetmap-b9d573e6cc86
  - Multilinual address normalization
  - Address parsing

# Plan
- 28.06 Read resources, do EDA and understand the problem
- 29.06 Solve imbalanced data, implement rule-based and statistical method
  - Solve imbalanced data by a data generator
  - Rule-based method to identify the country
  - Statistical method to identify the country
- 30.06 Develop a service and test it 

# Goal
- Develop a solution combining rule-based and statistical methods
  - Rule-based: indexing the city to its corresponding country
  - Statistical: machine learning model for country identification
  - Combine the two methods to get a better result

# Problems

## Data
- Imbalance in the data:
  - Expand current data using data augmentation methods
  - Write a data generator using collected data from openaddr
![img.png](res/img.png)
- Length of address:
![img_1.png](res/img_1.png)

## Model
- Rule-based method:
  - The city name may contain a comma
- Multilingual address
  - Fine-tune a multilingual model directly on the data?
- Pre-train a model
  - Fine-tune the pre-trained model on the data?

## Configs
- Choices of models
- Supported countries


## Service
- FastAPI?
- Test cases?

