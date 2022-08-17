import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot
from torch.nn import ModuleList
import global_var


class RoSA(torch.nn.Module):
    def __init__(
        self,
        encoder='gcn',
        input_dim=1433,

        # model configuration
        layer_num=2, # layers of encoder
        hidden=128, # encoder hidden size
        proj_shape=(128, 128), # hidden size for predictor
        
        T = 0.4, # temperature
        is_rectified=True, # use rectified cost matrix
        topo_t = 2 # temperature for calculating re-scale ratios with topology dist
    ):
        super(RoSA, self).__init__()

        # load components
        if encoder == 'gcn':
            Encoder = GCN
        elif encoder == 'sage-gcn':
            Encoder = GraphSAGE_GCN
        else:
            raise NotImplementedError(f'{encoder} is not  implemented!')
        self.encoder = Encoder(input_dim=input_dim, layer_num=layer_num, hidden=hidden)
        self.T = T
        self.rectified = is_rectified
        self.topo_t = topo_t
        self.hidden = hidden

        # adapative size for mlp's input dim
        fake_x = torch.rand((2, input_dim))
        fake_edge_index = torch.LongTensor([[0], [0]])
        fake_g = Data(x=fake_x, edge_index=fake_edge_index)
        # fake_graph = Data(x=fake_x, edge_index=fake_edge_index, batch=torch.LongTensor([0]))
        with torch.no_grad():
            rep = self.encoder(fake_g)
            hid = rep.shape[-1]
            self.projector = Projector([hid, *proj_shape])
    
    def gen_rep(self, data):
        h = self.encoder(data)
        z = self.projector(h)
        return h, z

    def sim(self, reps1, reps2):
        reps1_unit = F.normalize(reps1, dim=-1)
        reps2_unit = F.normalize(reps2, dim=-1)
        if len(reps1.shape) == 2:
            sim_mat = torch.einsum("ik,jk->ij", [reps1_unit, reps2_unit])
        elif len(reps1.shape) == 3:
            sim_mat = torch.einsum('bik,bjk->bij', [reps1_unit, reps2_unit])
        else:
            print(f"{len(reps1.shape)} dimension tensor is not supported for this function!")
        return sim_mat

    def topology_dist(self, node_idx1, node_idx2):
        full_topology_dist = global_var.get_value('topology_dist').cuda()
        
        batch_size = node_idx1.shape[0]
        batch_subpology_dist = [full_topology_dist.index_select(dim=0, index=node_idx1[i]).index_select(dim=1, index=node_idx2[i]) for i in range(batch_size)]
        batch_subpology_dist = torch.stack(batch_subpology_dist)
        return batch_subpology_dist

    def _batched_semi_emd_loss(self, out1, avg_out1, out2, avg_out2, lamb=20, rescale_ratio=None):
        assert out1.shape[0] == out2.shape[0] and avg_out1.shape == avg_out2.shape
        
        cost_matrix = 1-self.sim(out1, out2)
        if rescale_ratio is not None:
            cost_matrix = cost_matrix * rescale_ratio

        # Sinkhorn iteration
        iter_times = 5
        with torch.no_grad():
            r = torch.bmm(out1, avg_out2.transpose(1,2))
            r[r<=0] = 1e-8
            r = r / r.sum(dim=1, keepdim=True)
            c = torch.bmm(out2, avg_out1.transpose(1,2))
            c[c<=0] = 1e-8
            c = c / c.sum(dim=1, keepdim=True)
            P = torch.exp(-1*lamb*cost_matrix)
            u = (torch.ones_like(c)/c.shape[1])
            for i in range(iter_times):
                v = torch.div(r, torch.bmm(P, u))
                u = torch.div(c, torch.bmm(P.transpose(1,2), v))
            u = u.squeeze(dim=-1)
            v = v.squeeze(dim=-1)
            transport_matrix = torch.bmm(torch.bmm(matrix_diag(v), P), matrix_diag(u))
        assert cost_matrix.shape == transport_matrix.shape

        # S = torch.mul(transport_matrix, 1-cost_matrix).sum(dim=1).sum(dim=1, keepdim=True)
        emd = torch.mul(transport_matrix, cost_matrix).sum(dim=1).sum(dim=1, keepdim=True)
        S = 2-2*emd
        return S

    def batched_semi_emd_loss(self, reps1, reps2, batch1, batch2, original_idx1=None, original_idx2=None):
        batch_size1 = batch1.max().cpu().item()
        batch_size2 = batch2.max().cpu().item()
        assert batch_size1 == batch_size2
        batch_size = batch_size1+1
        # avg nodes rep
        y_online1_pooling = global_mean_pool(reps1, batch1)
        y_online2_pooling = global_mean_pool(reps2, batch2)
        avg_out1 = y_online1_pooling[:, None, :] # (B,1,D)
        avg_out2 = y_online2_pooling[:, None, :] # (B,1,D)

        # x reps from sparse to dense
        out1, mask1 = to_dense_batch(reps1, batch1, fill_value=1e-8) # (B,N,D), B means batch size, N means the number of nodes, D means hidden size
        out2, mask2 = to_dense_batch(reps2, batch2, fill_value=1e-8) # (B,M,D)

        if original_idx1 != None and original_idx2 != None:
            dense_original_idx1, idx_mask1 = to_dense_batch(original_idx1, batch=batch1, fill_value=0)
            dense_original_idx2, idx_mask2 = to_dense_batch(original_idx2, batch=batch2, fill_value=0)
            topology_dist = self.topology_dist(dense_original_idx1, dense_original_idx2)
            rescale_ratio = torch.sigmoid(topology_dist/self.topo_t)
            topo_mask = torch.bmm(idx_mask1[:,:,None].float(), idx_mask2[:,None,:].float()).bool()
            loss_pos = self._batched_semi_emd_loss(out1, avg_out1, out2, avg_out2, rescale_ratio=rescale_ratio) * 2
        else:
            loss_pos = self._batched_semi_emd_loss(out1, avg_out1, out2, avg_out2)

        T = self.T # temperature
        f = lambda x: torch.exp(x/T)

        total_neg_loss = 0
        # completely create negative samples
        neg_index = list(range(batch_size))
        for i in range((batch_size-1)):
            neg_index.insert(0, neg_index.pop(-1))
            out1_perm = out1[neg_index].clone()
            out2_perm = out2[neg_index].clone()
            avg_out1_perm = avg_out1[neg_index].clone()
            avg_out2_perm = avg_out2[neg_index].clone()
            total_neg_loss += f(self._batched_semi_emd_loss(out1, avg_out1, out1_perm, avg_out1_perm)) + f(self._batched_semi_emd_loss(out1, avg_out1, out2_perm, avg_out2_perm))

        loss = -torch.log(f(loss_pos) / (total_neg_loss))

        return loss



    def gen_loss(self, reps1, reps2, batch1=None, batch2=None, original_idx1=None, original_idx2=None):
        # data with batch
        loss1 = self.batched_semi_emd_loss(reps1, reps2, batch1, batch2, original_idx1=original_idx1, original_idx2=original_idx2)
        loss2 = self.batched_semi_emd_loss(reps2, reps1, batch2, batch1, original_idx1=original_idx2, original_idx2=original_idx1)
        loss = (loss1*0.5 + loss2*0.5).mean()
        return loss

    def forward(self, view1, view2, batch1=None, batch2=None):
        h1, z1 = self.gen_rep(view1)
        h2, z2 = self.gen_rep(view2)

        if hasattr(view1, 'original_idx') and self.rectified:
            original_idx1, original_idx2 = view1.original_idx+1, view2.original_idx+1 # shift one, zero-index is stored for padding nodes
            loss = self.gen_loss(z1, z2, batch1, batch2, original_idx1, original_idx2) 
        else:
            loss = self.gen_loss(z1, z2, batch1, batch2) 

        return loss

    def embed(self, data):
        h = self.encoder(data)
        return h.detach()


class GCN(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden
        self.input_dim = input_dim

        self.convs = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GCNConv(input_dim, hidden*2)) 
            for i in range(layer_num-2):
                self.convs.append(GCNConv(hidden*2, hidden*2))
                glorot(self.convs[i].weight)
            self.convs.append(GCNConv(hidden*2, hidden))
            glorot(self.convs[-1].weight)

        else: # one layer gcn
            self.convs.append(GCNConv(input_dim, hidden)) 
            glorot(self.convs[-1].weight)

    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            x = F.relu(self.convs[i](x, edge_index))
        return x


class GraphSAGE_GCN(torch.nn.Module):
    def __init__(self, input_dim, layer_num=3, hidden=512):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.layer = layer_num
        self.acts = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for i in range(self.layer):
            if i == 0:
                self.convs.append(SAGEConv(input_dim, hidden, root_weight=True))
            else:
                self.convs.append(SAGEConv(hidden, hidden, root_weight=True))
            # self.acts.append(torch.nn.PReLU(hidden))
            self.acts.append(torch.nn.ELU())
            self.norms.append(torch.nn.BatchNorm1d(hidden))
            
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.layer):
            x = self.acts[i](self.norms[i](self.convs[i](x, edge_index)))
        return x


class Projector(torch.nn.Module):
    def __init__(self, shape=()):
        super(Projector, self).__init__()
        if len(shape) < 3:
            raise Exception("Wrong shape for Projector")

        self.main = torch.nn.Sequential(
            torch.nn.Linear(shape[0], shape[1]),
            torch.nn.BatchNorm1d(shape[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(shape[1], shape[2])
        )

    def forward(self, x):
        return self.main(x)


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result

class LogReg(torch.nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = torch.nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
