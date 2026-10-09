# E-Commerce Product Attribute Extraction Using NLP

## Project Overview

This project focuses on extracting product attributes from e-commerce product descriptions using Natural Language Processing (NLP) and Named Entity Recognition (NER).

The system identifies relevant attributes from product descriptions and evaluates the model's performance using a dedicated evaluation and auditing workflow.

## Project Objectives

- Develop an NLP pipeline for processing product descriptions.
- Train an NLP model to identify product attributes.
- Extract attribute spans from product descriptions.
- Evaluate model performance using precision, recall, and F1 score.
- Audit model predictions to identify errors and limitations.

## Supported Attributes

The current labelled dataset supports the following attributes:

- SLEEVE_STYLE
- NECKLINE
- LENGTH
- PATTERN

## Project Components

### 1. NLP Pipeline
Processes product descriptions and prepares text for model prediction.

### 2. NLP Model and Attribute Extraction
Uses Named Entity Recognition to identify attribute spans in product descriptions.

### 3. Model Evaluation
Measures model performance using precision, recall, and F1 score.

### 4. Model Audit
Examines prediction errors, missed attributes, incorrect labels, and potential performance differences across data subgroups.

## Technologies Used

- Python
- spaCy
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Dataset

The project uses labelled product descriptions divided into training, validation, and test datasets.

The training dataset is used to train the model, the validation dataset supports model development, and the test dataset is reserved for final evaluation.

## Evaluation Metrics

The model evaluation process includes:

- **Precision:** Measures how many predicted attributes are correct.
- **Recall:** Measures how many actual attributes the model successfully identifies.
- **F1 Score:** Combines precision and recall into a single metric.

Final scores must be generated from the trained model's predictions on the held-out test dataset.

## Model Auditing

The audit process investigates:

- Missed attributes
- Incorrect attribute predictions
- Incorrect entity boundaries
- False-positive predictions
- Performance across relevant data subgroups

## Limitations

The model is limited to the attribute labels represented in its labelled dataset. It should not be assumed to reliably extract unsupported attributes such as brand, colour, size, or material.

Model performance must be established through actual training and evaluation rather than assumed in advance.

## Reproducibility

The repository will contain the source code, dependency list, datasets where permitted, and instructions required to reproduce the training, evaluation, and auditing workflow.

## Academic Project

Developed as part of the NLP CA3 project on E-Commerce Product Attribute Extraction.
