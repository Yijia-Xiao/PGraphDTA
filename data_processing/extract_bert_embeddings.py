from transformers import BertModel, BertTokenizer
from tdc.multi_pred import DTI
import torch
import numpy as np
from tqdm import tqdm
import re
import os

prot_models = ["Rostlab/prot_bert", "yarongef/DistilProtBert"]
model_choice = 0
PROT_MODEL = prot_models[model_choice]

model_names = ["ProtBERT", "DistilProtBERT"]
model_name = model_names[model_choice]

datasets = ["DAVIS", "KIBA"]
dataset_choice = 0
DATASET = datasets[dataset_choice]

dataset = DTI(name = DATASET)

print("Extracting Embeddings for " + DATASET + " dataset using " + model_name + " model")

split = dataset.get_split()
train_data = split["train"]
valid_data = split["valid"]
test_data = split["test"]

train_targets = train_data.Target.tolist()
valid_targets = valid_data.Target.tolist()
test_targets = test_data.Target.tolist()


def preprocess_protein(sequence):
    processProtein = [aa for aa in sequence] # aa is a single amino acid
    processProtein = " ".join(processProtein)
    processProtein = re.sub(r"[UZOB]", "X", processProtein)
    return processProtein

train_targets = [preprocess_protein(train_target) for train_target in train_targets]
valid_targets = [preprocess_protein(valid_target) for valid_target in valid_targets]
test_targets = [preprocess_protein(test_target) for test_target in test_targets]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

prot_tokenizer = BertTokenizer.from_pretrained(PROT_MODEL, do_lower_case=False)
prot_model = BertModel.from_pretrained(PROT_MODEL)
prot_model.to(device)

BATCH_SIZE = 64
MAX_PROT_LEN = 1024

save_dir = "../embeddings/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("Extracting protein embeddings for train set")
train_batches = [train_targets[i:i+BATCH_SIZE] for i in range(0, len(train_targets), BATCH_SIZE)]

protein_emb_train = []
for train_batch in tqdm(train_batches):
    encoded_proteins = prot_tokenizer(train_batch, 
                                  return_tensors='pt', 
                                  max_length=MAX_PROT_LEN, 
                                  truncation=True, 
                                  padding=True)
    
    encoded_proteins = encoded_proteins.to(device)
    with torch.no_grad():
        train_target_embeddings = prot_model(**encoded_proteins).last_hidden_state[:, 0, :]
    for train_target_embedding in train_target_embeddings.cpu().detach().numpy():
        protein_emb_train.append(train_target_embedding)

# Save protein embeddings train to a file
PATH = save_dir + "protein_emb_train_" + model_name + "_" + DATASET + ".npy"
np.save(PATH, protein_emb_train)

print("Extracting protein embeddings for validation set")
valid_batches = [valid_targets[i:i+BATCH_SIZE] for i in range(0, len(valid_targets), BATCH_SIZE)]

protein_emb_valid = []
for valid_batch in tqdm(valid_batches):
    encoded_proteins = prot_tokenizer(valid_batch, 
                                  return_tensors='pt', 
                                  max_length=MAX_PROT_LEN, 
                                  truncation=True,
                                  padding=True)

    encoded_proteins = encoded_proteins.to(device)
    with torch.no_grad():
        valid_target_embeddings = prot_model(**encoded_proteins).last_hidden_state[:, 0, :]
    for valid_target_embedding in valid_target_embeddings.cpu().detach().numpy():
        protein_emb_valid.append(valid_target_embedding)

# Save protein embeddings valid to a file
PATH = save_dir + "protein_emb_valid_" + model_name + "_" + DATASET + ".npy"
np.save(PATH, protein_emb_valid)

print("Extracting protein embeddings for test set")
test_batches = [test_targets[i:i+BATCH_SIZE] for i in range(0, len(test_targets), BATCH_SIZE)]

protein_emb_test = []
for test_batch in tqdm(test_batches):
    encoded_proteins = prot_tokenizer(test_batch,
                                    return_tensors='pt',
                                    max_length=MAX_PROT_LEN,
                                    truncation=True,
                                    padding=True)

    encoded_proteins = encoded_proteins.to(device)
    with torch.no_grad():
        test_target_embeddings = prot_model(**encoded_proteins).last_hidden_state[:, 0, :]
    for test_target_embedding in test_target_embeddings.cpu().detach().numpy():
        protein_emb_test.append(test_target_embedding)

# Save protein embeddings test to a file
PATH = save_dir + "protein_emb_test_" + model_name + "_" + DATASET + ".npy"
np.save(PATH, protein_emb_test)