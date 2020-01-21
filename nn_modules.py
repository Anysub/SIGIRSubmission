#!/usr/bin/env python

"""
    nn_modules.py
"""
import os
import psutil
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import gc

import numpy as np
from scipy import sparse
from helpers import to_numpy

import inspect

#from gpu_mem_track import  MemTracker

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
    
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)

def weight_init(m): 
	if isinstance(m, nn.Linear):
		nn.init.kaiming_uniform_(m.weight.data)

# --
# Samplers

class UniformNeighborSampler(object):
    """
        Samples from a "dense 2D edgelist", which looks like
        
            [
                [0, 1, 2, ..., 4],
                [0, 0, 5, ..., 10],
                ...
            ]
        
        stored as torch.LongTensor.

        Adj[a1,a2]=id, where the edge embedding of edge (a1,a2) is stored at emb[id]

        If a node does not have a neighbor, sample itself n_sample times and
        return emb[0]. * emb[0] is the padding zero embedding.

        We didn't apply the optimization from GraphSage
    """

    def __init__(self):
        pass

    def __call__(self, adj, ids, n_samples=16):
        cuda = adj.is_cuda

        neigh = []
        mask = []
        for v in ids:
            nonz = torch.nonzero(adj[v]).view(-1)
            #if (len(nonz) == 0):
                # no neighbor, only sample from itself
                # for edge embedding... PADDING with all-zero embedding at edge_emb[0]
                #if cuda:
                    #neigh.append(torch.cuda.LongTensor([v]).repeat(n_samples))
                    #mask.append(torch.cuda.LongTensor([1]).repeat(n_samples))
                #else:
                    #neigh.append(torch.LongTensor([v]).repeat(n_samples))
                    #mask.append(torch.LongTensor([1]).repeat(n_samples))
            #else:
            idx = np.random.choice(nonz.shape[0], n_samples)
            neigh.append(nonz[idx])
        mask = torch.zeros((ids.shape[0],n_samples)).to(ids.device)
        neigh = torch.stack(neigh).long().view(-1)
        edges = adj[
            ids.view(-1, 1).repeat(1, n_samples).view(-1),
            neigh]
        return neigh, edges, mask


class SpUniformNeighborSampler(object):
    """
        Samples from a "sparse 2D edgelist", which looks like

            [
                [0, 1, 2, ..., 4],
                [0, 0, 5, ..., 10],
                ...
            ]

        stored as torch.LongTensor.

        Adj[a1,a2]=id, where the edge embedding of edge (a1,a2) is stored at emb[id]

        If a node does not have a neighbor, sample itself n_sample times and
        return emb[0]. * emb[0] is the padding zero embedding.

        We didn't apply the optimization from GraphSage
    """

    def __init__(self):
        pass

    def __call__(self, adj, ids, n_samples=16):

        cuda = adj.is_cuda

        nonz = adj._indices()
        values = adj._values()

        mask = []
        neigh = []
        edges = []
        for v in ids:
            n = torch.nonzero(nonz[0, :] == v).view(-1)
            #if (len(n) == 0):
            #    # no neighbor, only sample from itself
            #    # for edge embedding... PADDING with all-zero embedding at edge_emb[0]
            #    if cuda:
            #        neigh.append(torch.cuda.LongTensor([v]).repeat(n_samples))
            #        edges.append(torch.cuda.LongTensor([0]).repeat(n_samples))
            #        mask.append(torch.cuda.LongTensor([1]).repeat(n_samples))
            #    else:
            #        neigh.append(torch.LongTensor([v]).repeat(n_samples))
            #        edges.append(torch.LongTensor([0]).repeat(n_samples))
            #        mask.append(torch.LongTensor([1]).repeat(n_samples))
            #else:
                # np.random.choice(nonz.shape[0], n_samples)
            if True:
#n.shape[0] >= n_samples:
                    idx = torch.randint(0, n.shape[0], (n_samples,))

                    neigh.append(nonz[1, n[idx]])
                    edges.append(values[n[idx]])
                    if cuda:
                        mask.append(torch.cuda.LongTensor([0]).repeat(n_samples))
                    else:
                        mask.append(torch.LongTensor([0]).repeat(n_samples))

            else:

                    if cuda:
                        neigh.append(torch.cat([nonz[1, n], torch.cuda.LongTensor([v]).repeat(n_samples - n.shape[0])]))
                        edges.append(torch.cat([values[n], torch.cuda.LongTensor([0]).repeat(n_samples - n.shape[0])]))
                        mask.append(torch.cat([torch.cuda.LongTensor([0]).repeat(n.shape[0]),
                                               torch.cuda.LongTensor([1]).repeat(n_samples - n.shape[0])]))
                    else:
                        neigh.append(torch.cat([nonz[1, n], torch.LongTensor([v]).repeat(n_samples - n.shape[0])]))
                        edges.append(torch.cat([values[n], torch.LongTensor([0]).repeat(n_samples - n.shape[0])]))
                        mask.append(torch.cat([torch.LongTensor([0]).repeat(n.shape[0]),
                                               torch.LongTensor([1]).repeat(n_samples - n.shape[0])]))

        neigh = torch.stack(neigh).long().view(-1)
        edges = torch.stack(edges).long().view(-1)
        mask = torch.stack(mask).float()

        return neigh, edges, mask

# --
# Preprocessers

class IdentityPrep(nn.Module):
    def __init__(self, input_dim, n_nodes=None, embedding_dim=64):
        """ Example of preprocessor -- doesn't do anything """
        super(IdentityPrep, self).__init__()
        self.input_dim = input_dim

    @property
    def output_dim(self):
        return self.input_dim

    def forward(self, ids, feats, layer_idx=0):
        return feats


class NodeEmbeddingPrep(nn.Module):
    def __init__(self, input_dim, n_nodes, pre_trained=None, embedding_dim=64):
        """ adds node embedding """
        super(NodeEmbeddingPrep, self).__init__()

        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=n_nodes + 1, embedding_dim=embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)  # Affine transform, for changing scale + location

        if pre_trained is not None:
            self.embedding.from_pretrained(pre_trained, padding_idx=0)

    @property
    def output_dim(self):
        if self.input_dim:
            return self.input_dim + self.embedding_dim
        else:
            return self.embedding_dim

    def forward(self, ids, feats, layer_idx=0):
        if layer_idx > 0:
            embs = self.embedding(ids)
        else:
            # Don't look at node's own embedding for prediction, or you'll probably overfit a lot
            embs = self.embedding(Variable(ids.clone().data.zero_() + self.n_nodes))

        embs = self.fc(embs)
        if self.input_dim and feats is not None:
            return torch.cat([feats, embs], dim=1)
        else:
            return embs


class LinearPrep(nn.Module):
    def __init__(self, input_dim, n_nodes, output_dim=32, embedding_dim=64):
        """ adds node embedding """
        super(LinearPrep, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.output_dim = output_dim

    def forward(self, ids, feats, layer_idx=0):
        return self.fc(feats)


# --
# Aggregators


class AttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation, hidden_dim=32,
                 dropout=0.5, alpha=0.8,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(AttentionAggregator, self).__init__()

        self.att = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm
        self.output_dim = output_dim
        self.activation = activation

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, x, neibs, edge_emb, mask):
        # Compute attention weights
        neib_att = self.att(neibs)
        x_att = self.att(x)
        neib_att = neib_att.view(x.size(0), -1, neib_att.size(1))
        x_att = x_att.view(x_att.size(0), x_att.size(1), 1)
        # ws = F.softmax(torch.bmm(neib_att, x_att).squeeze())

        ws = torch.bmm(neib_att, x_att).squeeze()
        ws += -9999999 * mask
        ws = F.softmax(ws, dim=1)

        # Weighted average of neighbors
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = torch.sum(agg_neib * ws.unsqueeze(-1), dim=1)

        out = self.fc_x(x) + self.fc_neib(agg_neib)

        if self.batchnorm:
            out = self.bn(out)

        out = self.dropout(out)

        if self.activation:
            out = self.activation(out)

        return out

class AttentionAggregator2(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation, hidden_dim=512,
                 dropout=0.5,attn_dropout=0,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(AttentionAggregator2, self).__init__()
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.att = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])

        self.att_neigh = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])

        self.att_edge = nn.Sequential(*[
            nn.Linear(edge_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])

        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_neib = nn.Linear(input_dim, output_dim)
        self.fc_edge = nn.Linear(edge_dim, output_dim)

        self.concat_node = concat_node
        self.concat_edge = concat_edge

        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        if concat_node:
            self.output_dim = output_dim * 2
        else:
            self.output_dim = output_dim
        if concat_edge:
            self.output_dim += output_dim
        else:
            pass
        self.activation = activation

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

        #self.apply(weight_init)

    def forward(self, x, neibs, edge_emb, mask):
        # Compute attention weights
        # neibs = torch.cat([neibs, edge_emb], dim=1)

        neib_att = self.att_neigh(neibs)
        edge_att = self.att_edge(edge_emb)
        x_att = self.att(x)

        neib_att = neib_att.view(x.size(0), -1, neib_att.size(1))
        edge_att = edge_att.view(x.size(0), -1, edge_att.size(1))
        x_att = x_att.view(x_att.size(0), x_att.size(1), 1)

        ws = torch.bmm(neib_att, x_att).squeeze()
       
        import math
        ws /= math.sqrt(512)
        #ws += -9999999 * mask
        #ws = F.leaky_relu(ws)
        ws = F.softmax(ws, dim=1)

        #dropout for attention coefficient
        ws = self.attn_dropout(ws)
        #ws = F.normalize(ws,p=1,dim=1)

        # Weighted average of neighbors
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = torch.bmm(ws.view(x.size(0),1,-1),agg_neib).squeeze()

        if self.concat_node:
            out = torch.cat([self.fc_x(x), self.fc_neib(agg_neib)], dim=1)
        else:
            out = self.fc_x(x) + self.fc_neib(agg_neib)

        ws = torch.bmm(edge_att, x_att).squeeze()
        ws += -9999999 * mask
        ws = F.softmax(ws, dim=1)

        # Weighted average of neighbors
        agg_edge = edge_emb.view(x.size(0), -1, edge_emb.size(1))
        agg_edge = torch.sum(agg_edge * ws.unsqueeze(-1), dim=1)

        if self.concat_edge:
            out = torch.cat([out,self.fc_edge(agg_edge)],dim=1)
        else:
            out = self.fc_edge(agg_edge)+out

        if self.batchnorm:
            out = self.bn(out)

        out = self.dropout(out)

        if self.activation:
            out = self.activation(out)

        return out

class IdEdgeAggregator(nn.Module):
    def __init__(self, input_dim, edge_dim, activation, dropout=0.5, batchnorm=False):
        super(IdEdgeAggregator, self).__init__()

        self.input_dim = input_dim
        self.activation = activation
        self.edge_dim = edge_dim
        self.batchnorm = batchnorm
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, neibs, edge_emb, mask):
        # identical mapping
        # e = sigma(w1*x+W2*neibs+b) @ e
        return edge_emb

class MetapathAttentionLayer(nn.Module):
    """
    metapath attention layer.
    """

    def __init__(self, in_features,n_head=4, alpha=0.8, dropout=0.5, hidden_dim=64, batchnorm=False):
        super(MetapathAttentionLayer, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        self.att = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])

        self.mlp = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features),
        ])

        a = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        nn.init.xavier_uniform_(a.data, gain=1.414)
        self.register_parameter('a', a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        n_meta = input.shape[0]
        input = input.transpose(0, 1)  # tensor(N,nmeta,in_features)
        N = input.size()[0]
        input_dim = input.shape[2]

        input = input.contiguous()
        a_input = self.att(input.view(-1, input_dim)) \
            .view(N, n_meta, -1)

        # a_input = torch.cat([input.repeat(1,1,n_meta).view(N, n_meta*n_meta, -1),
        #                      input.repeat(1,n_meta, 1)], dim=2).view(N, -1, 2 * input_dim)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))   #e: tensor(N,nmeta)
        e = F.softmax(e, dim=1).view(N, 1, n_meta)

        output = torch.bmm(e, input).squeeze()

        output = self.dropout(output)
        output = self.mlp(output)

        if self.batchnorm:
            output = self.bn(output)

        weight = torch.sum(e.view(N, n_meta), dim=0) / N

        return F.relu(output), weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'

class MetapathGateLayer(nn.Module):
    """
    metapath gated attention layer.
    """

    def __init__(self, in_features,n_head=4, alpha=0.8, dropout=0.5, hidden_dim=64, batchnorm=False):
        super(MetapathGateLayer, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        self.att = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, in_features, bias=False),
        ])

        self.mlp = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features),
        ])

        #a = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        #nn.init.xavier_uniform_(a.data, gain=1.414)
        #self.register_parameter('a', a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.gate=nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, in_features, bias=False),
        ])
        self.update=nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, in_features, bias=False),
        ])


        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, input):
        """
        :param input: tensor(nmeta,N,in_features)
        :return:
        """
        n_meta = input.shape[0]
        input = input.transpose(0, 1)  # tensor(N,nmeta,in_features)
        N = input.size()[0]
        input_dim = input.shape[2]

        input = input.contiguous()
        gate_input = F.sigmoid(self.gate(input))
        update_input = F.tanh(self.update(input))
        output = gate_input*update_input

        output = torch.sum(output,dim=1).squeeze()
        
        #a_input = self.att(input)
        #e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))   #e: tensor(N,nmeta)
        #e = F.softmax(e, dim=1).view(N, 1, n_meta)

        #output = torch.bmm(e, input).squeeze()

        output = self.dropout(output)
        output = self.mlp(output)

        if self.batchnorm:
            output = self.bn(output)

        #weight = torch.sum(e.view(N, n_meta), dim=0) / N
        weight=None

        return F.relu(output), weight

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


sampler_lookup = {
    "uniform_neighbor_sampler": UniformNeighborSampler,
    "sparse_uniform_neighbor_sampler": SpUniformNeighborSampler,
}

prep_lookup = {
    "identity": IdentityPrep,
    "node_embedding": NodeEmbeddingPrep,
    "linear": LinearPrep,
}

aggregator_lookup = {
    "attention": AttentionAggregator,
    "attention2": AttentionAggregator2,
}

metapath_aggregator_lookup = {
    "attention": MetapathAttentionLayer,
    "gate":MetapathGateLayer,
}

edge_aggregator_lookup = {
    "identity": IdEdgeAggregator,
}

