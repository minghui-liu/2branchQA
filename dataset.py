import torch
from torch.utils.data import Dataset
import json
import numpy as np
import pickle

# class QantaDataset(Dataset):
#     def __init__(self, data_file, question_file, max_seq_length):
#         self.data_file = data_file
#         self.max_seq_length = max_seq_length
#         self.questions = {}

#         self.graphs = self.load_graphs()
    
#     def load_question(self):
#         with open(self.data_file, 'r') as f:
#             for line in f:
#                 data = json.loads(line)
#                 self.questions[data['id']] = data['question']

#     def load_graphs(self):
#         graphs = {}
#         with open(self.data_file, 'r') as f:
#             for line in f:
#                 data = json.loads(line)
#                 graphs[data['id']] = data
#         return graphs

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         data = self.data[idx]
#         question = torch.tensor(data['question'])
#         candidate = torch.tensor(data['candidate'])
#         evidence = torch.tensor(data['evidence'])
#         label = torch.tensor(data['label'])
#         return question, candidate, evidence, label
    

class QBLinkDataset(Dataset):
    def __init__(self, data_file, embedding_file):
        self.data_file = data_file
        self.embedding_file = embedding_file
        self.embeddings = self.load_embedding()
        self.data = self.load_data()

    def load_embedding(self):
        # load embedding from npy file
        print(f"[QBLinkDataset] Loading embeddings from {self.embedding_file}")
        embeddings = np.load(self.embedding_file, allow_pickle=True)
        return embeddings

    def load_data(self):
        # load data from pickle file
        print(f"[QBLinkDataset] Loading data from {self.data_file}")
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        q_indices = np.array(data[0])
        e_indices = np.array(data[1])
        c_indices = np.array(data[2])
        label = np.int(data[3])

        q_embeddings = self.embeddings[q_indices]
        e_embeddings = self.embeddings[e_indices]
        c_embeddings = self.embeddings[c_indices]

        q_vector = np.mean(q_embeddings, axis=0).astype(np.float32)
        e_vector = np.mean(e_embeddings, axis=0).astype(np.float32)
        c_vector = np.mean(c_embeddings, axis=0).astype(np.float32)

        q_vector, e_vector, c_vector = torch.from_numpy(q_vector).float(), torch.from_numpy(e_vector).float(), torch.from_numpy(c_vector).float()
        return q_vector, e_vector, c_vector, label