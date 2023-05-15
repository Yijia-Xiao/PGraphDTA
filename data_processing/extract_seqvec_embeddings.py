from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
from tdc.multi_pred import DTI
import torch
import numpy as np
import os
from tqdm import tqdm

model_name = "SeqVec"

datasets = ["DAVIS", "KIBA"]
dataset_choice = 0
dataset_name = datasets[dataset_choice]

dataset = DTI(name = dataset_name)

split = dataset.get_split()
train_data = split["train"]
valid_data = split["valid"]
test_data = split["test"]

train_targets = train_data.Target.tolist()
valid_targets = valid_data.Target.tolist()
test_targets = test_data.Target.tolist()

train_targets = [list(train_target) for train_target in train_targets]
valid_targets = [list(valid_target) for valid_target in valid_targets]
test_targets = [list(test_target) for test_target in test_targets]

model_dir = Path('uniref50_v2/')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
device = 0 if torch.cuda.is_available() else -1
embedder = ElmoEmbedder(options,weights, cuda_device=device)

BATCH_SIZE = 16

save_dir = "../embeddings/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Extract batches from train_targest with BATCH_SIZE
train_batches = [train_targets[i:i+BATCH_SIZE] for i in range(0, len(train_targets), BATCH_SIZE)]

protein_emb_train = []
for train_batch in tqdm(train_batches):
    train_target_embeddings = embedder.embed_sentences(train_batch)
    for train_target_embedding in train_target_embeddings:
        protein_embd = torch.tensor(train_target_embedding).sum(dim=0).mean(dim=0)
        protein_embd = protein_embd.detach().cpu().numpy()
        protein_emb_train.append(protein_embd)

# Save protein embeddings train to a file
PATH = save_dir + "protein_emb_train_" + model_name + "_" + dataset_name + ".npy"
np.save(PATH, protein_emb_train)

valid_batches = [valid_targets[i:i+BATCH_SIZE] for i in range(0, len(valid_targets), BATCH_SIZE)]

protein_emb_valid = []
for valid_batch in tqdm(valid_batches):
    valid_target_embeddings = embedder.embed_sentences(valid_batch)
    for valid_target_embedding in valid_target_embeddings:
        protein_embd = torch.tensor(valid_target_embedding).sum(dim=0).mean(dim=0)
        protein_embd = protein_embd.detach().cpu().numpy()
        protein_emb_valid.append(protein_embd)

# Save protein embeddings valid to a file
PATH = save_dir + "protein_emb_valid_" + model_name + "_" + dataset_name + ".npy"
np.save(PATH, protein_emb_valid)

test_batches = [test_targets[i:i+BATCH_SIZE] for i in range(0, len(test_targets), BATCH_SIZE)]

protein_emb_test = []
for test_batch in tqdm(test_batches):
    test_target_embeddings = embedder.embed_sentences(test_batch)
    for test_target_embedding in test_target_embeddings:
        protein_embd = torch.tensor(test_target_embedding).sum(dim=0).mean(dim=0)
        protein_embd = protein_embd.detach().cpu().numpy()
        protein_emb_test.append(protein_embd)

# Save protein embeddings test to a file
PATH = save_dir + "protein_emb_test_" + model_name + "_" + dataset_name + ".npy"
np.save(PATH, protein_emb_test)