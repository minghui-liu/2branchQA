import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

class TwoBranchNet(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # self.question_path = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.Linear(embedding_dim, embedding_dim),
        # )
        # self.evidence_path = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.Linear(embedding_dim, embedding_dim),
        # )
        # self.candidate_path = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.Linear(embedding_dim, embedding_dim),
        # )
        self.question_fc = nn.Linear(embedding_dim, embedding_dim)
        self.evidence_fc = nn.Linear(embedding_dim, embedding_dim)
        self.candidate_fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, question, candidate, evidence):
        question = self.question_fc(question)
        candidate = self.candidate_fc(candidate)
        evidence = self.evidence_fc(evidence)
        # question = self.question_path(question)
        # candidate = self.candidate_path(candidate)
        # evidence = self.evidence_path(evidence)

        question_cond = question * candidate
        evidence_cond = evidence * candidate

        return question_cond, evidence_cond
