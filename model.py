import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelLayer(nn.Module):

    def __init__(self,
                 n_in,
                 n_hid,
                 n_emb,
                 lambda_sparse=None,
                 lambda_l2=None,
                 activation='Sigmoid'):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_in, n_hid))
        self.u = nn.Parameter(torch.empty(n_in, 1, n_emb))
        self.v = nn.Parameter(torch.empty(1, n_hid, n_emb))
        self.b = nn.Parameter(torch.empty(n_hid))
        if activation == 'Identity':
            self.activation = nn.Identity()
            activation = 'Linear'
        else:
            self.activation = getattr(nn, activation)()
        activation = activation.lower()
        nn.init.xavier_uniform_(self.W,
                                gain=torch.nn.init.calculate_gain(activation))
        nn.init.xavier_uniform_(self.u,
                                gain=torch.nn.init.calculate_gain(activation))
        nn.init.xavier_uniform_(self.v,
                                gain=torch.nn.init.calculate_gain(activation))
        nn.init.zeros_(self.b)

        self.lambda_sparse = lambda_sparse
        self.lambda_l2 = lambda_l2

    def local_kernel(self, u, v):
        return torch.clamp(1 - torch.norm(u - v, p=2, dim=2), min=0)

    def forward(self, x):
        w_hat = self.local_kernel(self.u, self.v)
        # Local kernelised weight matrix
        W_eff = self.W * w_hat
        y = torch.matmul(x, W_eff) + self.b
        y = self.activation(y)
        if self.training and self.lambda_sparse is not None and self.lambda_l2 is not None:
            # Sparse regularisation
            reg_term = self.lambda_sparse * (w_hat**2).mean()
            reg_term += self.lambda_l2 * (self.W**2).mean()
            return y, reg_term
        else:
            return y


class KernelNet(nn.Module):

    def __init__(self,
                 n_u,
                 n_hid,
                 n_emb,
                 n_layers,
                 lambda_sparse=None,
                 lambda_l2=None,
                 dropout=0.33):
        super().__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(
                    KernelLayer(n_u, n_hid, n_emb, lambda_sparse, lambda_l2))
            else:
                layers.append(
                    KernelLayer(n_hid, n_hid, n_emb, lambda_sparse, lambda_l2))
        layers.append(
            KernelLayer(n_hid,
                        n_u,
                        n_emb,
                        lambda_sparse,
                        lambda_l2,
                        activation='ReLU'))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        total_reg = None
        if self.training:
            for i, layer in enumerate(self.layers):
                x, reg = layer(x)
                if i < len(self.layers) - 1:
                    x = self.dropout(x)
                if total_reg is None:
                    total_reg = reg
                else:
                    total_reg += reg
            return x, total_reg
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x)
            return x


class GLocalNet(nn.Module):

    def __init__(self, kernel_net, n_m, gk_size, dot_scale):
        super().__init__()
        self.gk_size = gk_size
        self.dot_scale = dot_scale
        self.local_kernel_net = kernel_net
        self.conv_kernel = torch.nn.Parameter(torch.empty(n_m, gk_size**2))
        nn.init.xavier_uniform_(self.conv_kernel,
                                gain=torch.nn.init.calculate_gain("leaky_relu"))
        self.stride = 1
        self.padding = (gk_size-1)//2

    def forward(self, x, x_local):
        gk = self.global_kernel(x_local, self.gk_size, self.dot_scale)
        x = self.global_conv(x, gk)
        if self.training:
            x, reg = self.local_kernel_net(x)
            return x, reg
        else:
            x = self.local_kernel_net(x)
            return x

    def global_kernel(self, x, gk_size, dot_scale):
        pooled_x = torch.mean(x, dim=1).view(1, -1)
        gk = torch.matmul(pooled_x,
                          self.conv_kernel) * dot_scale  # Scaled dot product
        gk = gk.view(1, 1, gk_size, gk_size)

        return gk

    def global_conv(self, input, W):
        input = input.unsqueeze(0).unsqueeze(0)
        conv2d = nn.LeakyReLU()(F.conv2d(input, W, stride=self.stride, padding=self.padding))
        return conv2d.squeeze(0).squeeze(0)