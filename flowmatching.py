import torch
from torch import nn

class FlowMatching:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

    @staticmethod
    def get_train_tuple(z1, z0=None, repeat=1, y=None):
        """
        Return (z_t, t, target, y_out)
        If y is provided (shape (bs,)), it will be expanded to match repeat and returned as y_out
        so downstream training can condition on labels.
        """
        bs, *other_dims = z1.shape
        t = torch.rand((bs, repeat, 1), device=z1.device)
        z1 = z1.reshape(bs, 1, -1).expand(-1, repeat, -1)
        if z0 is None:
            z0 = torch.randn_like(z1)
        else:
            z0 = z0.reshape(bs, 1, -1).expand(-1, repeat, -1) # expand z0 if provided, generally not used
        z_t = t * z1 + (1.0 - t) * z0 # linear flow map from z0 to z1
        target = z1 - z0
        t = t.reshape(-1)
        z_t = z_t.reshape(-1, *other_dims)
        target = target.reshape(-1, *other_dims)
        y_out = None
        if y is not None:
            # expand labels to match repeat and flatten
            y = y.reshape(bs, 1).expand(-1, repeat).reshape(-1)
            y_out = y
        return z_t, t, target, y_out

    @torch.no_grad()
    def sample_ode(self, z0, N=100, y=None):
        dt = 1.0 / N
        traj = []
        z = z0.detach().clone()
        batchsize = z.shape[0]
        traj.append(z.detach().clone()) # save the initial point and then N steps
        for i in range(N):
            # make t shape (batchsize,) so time_emb produces (B, hidden_size)
            # instead of (B,1,hidden_size), which would break downstream chunking
            t = torch.ones(batchsize, device=self.device) * (i * dt)
            if y is None:
                pred = self.model(z, t)
            else:
                pred = self.model(z, (t, y))
            z = z.detach().clone() + pred * dt # the Euler step: z_{t+1} = z_t + f(z_t, t) * dt
            traj.append(z.detach().clone()) # N steps   
        return traj
