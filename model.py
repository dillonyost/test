from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, nprop, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc_1 = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus_1 = nn.Softplus()
        if n_h > 1:
            self.fcs_1 = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                        for _ in range(n_h-1)])
            self.softpluses_1 = nn.ModuleList([nn.Softplus()
                                               for _ in range(n_h-1)])
        if self.classification:
            self.fc_out_1 = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out_1 = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax_1 = nn.LogSoftmax(dim=1)
            self.dropout_1 = nn.Dropout()
        self.conv_to_fc_2 = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus_2 = nn.Softplus()
        if n_h > 1:
            self.fcs_2 = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                        for _ in range(n_h-1)])
            self.softpluses_2 = nn.ModuleList([nn.Softplus()
                                               for _ in range(n_h-1)])
        if self.classification:
            self.fc_out_2 = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out_2 = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax_2 = nn.LogSoftmax(dim=1)
            self.dropout_2 = nn.Dropout()

    def forward(self, nprop, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        [ndata, _] = crys_fea.size()
#        print (ndata)
        out = torch.Tensor([]).cuda()

        prop_fea_1 = self.conv_to_fc_1(self.conv_to_fc_softplus_1(crys_fea))
        prop_fea_1 = self.conv_to_fc_softplus_1(prop_fea_1)
        if self.classification:
            prop_fea_1 = self.dropout(prop_fea_1)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses_1'):
            for fc, softplus in zip(self.fcs, self.softpluses_1):
                prop_fea_1 = softplus(fc(prop_fea_1))
        out_1 = self.fc_out_1(prop_fea_1)

        prop_fea_2 = self.conv_to_fc_2(self.conv_to_fc_softplus_2(crys_fea))
        prop_fea_2 = self.conv_to_fc_softplus_2(prop_fea_2)
        if self.classification:
            prop_fea_2 = self.dropout(prop_fea_2)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses_2'):
            for fc, softplus in zip(self.fcs, self.softpluses_2):
                prop_fea_2 = softplus(fc(prop_fea_2))
        out_2 = self.fc_out_2(prop_fea_2)
        
        out = torch.cat((out_1, out_2), dim = 0)
        out = torch.reshape(out, (nprop, ndata))
#        print (out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
