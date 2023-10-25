# PGraphDTA

Code repository for "Improving Drug Target Interaction Prediction using Protein Language Models and Contact Maps".

## Introduction
PGraphDTA is a computational tool designed for predicting Drug-Target Interactions (DTIs) using advanced graph neural networks. This code is based on the research paper titled "Improving Drug Target Interaction Prediction using Protein Language Models and Contact Maps".

## Project Structure
- `data_processing`: Contains scripts and utilities for pre-processing and preparing the data for training and inference.
- `dti_inference_dist.py`: Script for DTI inference.
- `dti_inference_dist_contact_map.py`: Script for DTI inference with contact map integration.
- `dti_train_dist.py`: Script for training the DTI model.
- `dti_train_dist_contact_map.py`: Script for training the DTI model with contact map integration.
- `models`: Directory containing pre-trained models and architecture definitions.

## Setup
1. Clone this repository to your local machine.
2. Set up a virtual environment (Anaconda3 recommended).
3. Create environment: conda env create --file environment.yml.

## Usage
### Training
To train the model, run:
```
python dti_train_dist.py [arguments]
```
For training with contact map integration, run:
```
python dti_train_dist_contact_map.py [arguments]
```

### Inference
To infer using the trained model, run:
```
python dti_inference_dist.py [arguments]
```
For inference with contact map integration, run:
```
python dti_inference_dist_contact_map.py [arguments]
```

## Data
Ensure your data is placed in the appropriate directories and follows the expected formats. Refer to the `data_processing` directory for utilities and scripts that can help in this regard.

## Models
The `models` directory contains pre-trained models and their architecture definitions. You can use these for direct inference or as a starting point for further training.

## Citation
If you find our repository helpful or used it, please cite our [paper](https://arxiv.org/abs/2310.04017).
```
@misc{bal2023pgraphdta,
      title={PGraphDTA: Improving Drug Target Interaction Prediction using Protein Language Models and Contact Maps}, 
      author={Rakesh Bal and Yijia Xiao and Wei Wang},
      year={2023},
      eprint={2310.04017},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
