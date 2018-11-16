import torch.nn as nn
import torch
from src.deeplab import Deeplab
import random

import utils
import numpy as np
from scipy.sparse import csc_matrix

class RWN(nn.Module):
    def __init__(self, num_classes=21,n=32,R=3):
        super(RWN, self).__init__()
        self.deeplab = Deeplab(num_classes=num_classes)
        for param in self.deeplab.parameters():
            param.requires_grad = False
        self.n=n
        self.R=R
        self.k=k=3+64+64

        self.upsample = nn.Upsample(size=(n,n), mode='bilinear', align_corners=True)

        self.conv=nn.Conv2d(k,1,(1,1),bias=False)
        self.pdist = nn.PairwiseDistance(p=1)

    def forward(self, x):
        batch_size=x.size()[0]
        input_size = x.size()[2:4]
        n=self.n
        n2=n**2
        assert input_size[0]==input_size[1]==n

        x2 = self.deeplab(x)[3]
        conv1,conv2,_=self.deeplab.low_level_feature

        x2=self.upsample(x2).view(batch_size,21,n2)
        conv1=self.upsample(conv1)
        conv2=self.upsample(conv2)

        intergrated=torch.cat((x,conv1,conv2),1)

        intergrated=self.conv(intergrated).view(batch_size,self.n,self.n)

        y=[]
        for batch in range(batch_size):
            intergrated_sampled = self._sampling(intergrated[batch].detach())
            W=self._L1_distance_within_R(intergrated_sampled)
            x_batch=x2[batch].transpose(0,1)
            y.append(torch.matmul(W,x_batch).view(1,n2,21))

        return torch.cat(y)


    def _L1_distance_within_R(self,x):
        n2=self.n**2
        indices_x=[]
        data=[]
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i,j]==0:
                    break
                new_x=utils.calc_D_between_scalar_matrix_within_R(x,i,j,self.R,self.n)
                new_x.data=np.exp(new_x.data)
                new_x.data=new_x.data/new_x.data.sum()
                indices_x+=[(i * self.n + j)]*new_x.data.shape[0]
                data.append(new_x)
        indices_y,value=utils.get_flatten_indices(data,self.n)
        return torch.sparse.FloatTensor(torch.LongTensor([indices_x,indices_y]),torch.FloatTensor(value),torch.Size([n2,n2]))


    def _sampling(self,x,fractions=0.3):
        """ converts dense tensor x to sparse format """

        indices = torch.nonzero(x)
        indices.requires_grad=False
        if len(indices.shape) == 0:  # if all elements are zeros
            return None
        indices_len=indices.shape[0]
        selected_indices=random.sample(range(indices_len),int(fractions*indices_len))
        indices=indices[selected_indices]
        indices = indices.t()
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return csc_matrix((values,(indices[0],indices[1])),shape=(self.n,self.n))


