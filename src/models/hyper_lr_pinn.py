import torch
import torch.nn as nn

class LR_PINN(nn.Module):
    def __init__(self, hidden_dim, phase='phase1', start_w=None, start_b=None, end_w=None, end_b=None,
                 col_0=None, col_1=None, col_2=None, row_0=None, row_1=None, row_2=None,
                 alpha_0=None, alpha_1=None, alpha_2=None):
        super(LR_PINN, self).__init__()

        # Common layers
        self.start_layer = nn.Linear(2, hidden_dim)
        self.end_layer = nn.Linear(hidden_dim, 1)

        self.hidden_dim = hidden_dim
        self.scale = 1 / hidden_dim

        # Phase1-specific parameters
        if phase == 'phase1':
            self.col_basis_0 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))
            self.col_basis_1 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))
            self.col_basis_2 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))

            self.row_basis_0 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))
            self.row_basis_1 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))
            self.row_basis_2 = nn.Parameter(self.scale * torch.rand(self.hidden_dim, self.hidden_dim))

            self.meta_layer_1 = nn.Linear(1, self.hidden_dim)  # Single parameter instead of three
            self.meta_layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.meta_layer_3 = nn.Linear(self.hidden_dim, self.hidden_dim)

            self.meta_alpha_0 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.meta_alpha_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.meta_alpha_2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Phase2-specific parameters
        if phase == 'phase2':
            self.start_layer.weight = nn.Parameter(start_w)
            self.start_layer.bias = nn.Parameter(start_b)
            self.end_layer.weight = nn.Parameter(end_w)
            self.end_layer.bias = nn.Parameter(end_b)

            self.col_basis_0 = nn.Parameter(col_0, requires_grad=False)
            self.col_basis_1 = nn.Parameter(col_1, requires_grad=False)
            self.col_basis_2 = nn.Parameter(col_2, requires_grad=False)

            self.row_basis_0 = nn.Parameter(row_0, requires_grad=False)
            self.row_basis_1 = nn.Parameter(row_1, requires_grad=False)
            self.row_basis_2 = nn.Parameter(row_2, requires_grad=False)

            self.alpha_0 = nn.Parameter(alpha_0)
            self.alpha_1 = nn.Parameter(alpha_1)
            self.alpha_2 = nn.Parameter(alpha_2)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.phase = phase
        self.meta_param = None

    def set_task(self, parameter, device):
        self.meta_param = parameter.to(device)

    def forward_phase1(self, x, t):
        # Meta-learning part (Phase1)
        meta_output = self.meta_layer_1(self.meta_param)
        meta_output = self.tanh(meta_output)
        meta_output = self.meta_layer_2(meta_output)
        meta_output = self.tanh(meta_output)
        meta_output = self.meta_layer_3(meta_output)
        meta_output = self.tanh(meta_output)

        meta_alpha_0_output = self.relu(self.meta_alpha_0(meta_output))
        meta_alpha_1_output = self.relu(self.meta_alpha_1(meta_output))
        meta_alpha_2_output = self.relu(self.meta_alpha_2(meta_output))

        alpha_0 = torch.diag_embed(meta_alpha_0_output)
        alpha_1 = torch.diag_embed(meta_alpha_1_output)
        alpha_2 = torch.diag_embed(meta_alpha_2_output)

        # Main neural network (Phase1)
        inputs = torch.cat([x, t], axis=1)
        weight_0 = torch.matmul(torch.matmul(self.col_basis_0, alpha_0), self.row_basis_0)
        weight_1 = torch.matmul(torch.matmul(self.col_basis_1, alpha_1), self.row_basis_1)
        weight_2 = torch.matmul(torch.matmul(self.col_basis_2, alpha_2), self.row_basis_2)

        emb_out = self.start_layer(inputs)
        emb_out = self.tanh(emb_out)
        emb_out = emb_out.unsqueeze(dim=1)

        emb_out = torch.bmm(emb_out, weight_0)
        emb_out = self.tanh(emb_out)

        emb_out = torch.bmm(emb_out, weight_1)
        emb_out = self.tanh(emb_out)

        emb_out = torch.bmm(emb_out, weight_2)
        emb_out = self.tanh(emb_out)

        emb_out = self.end_layer(emb_out)
        emb_out = emb_out.squeeze(dim=1)

        return emb_out

    def forward_phase2(self, x, t):
        # Main neural network (Phase2)
        weight_0 = torch.matmul(torch.matmul(self.col_basis_0, torch.diag(self.alpha_0)), self.row_basis_0)
        weight_1 = torch.matmul(torch.matmul(self.col_basis_1, torch.diag(self.alpha_1)), self.row_basis_1)
        weight_2 = torch.matmul(torch.matmul(self.col_basis_2, torch.diag(self.alpha_2)), self.row_basis_2)

        inputs = torch.cat([x, t], axis=1)
        emb_out = self.start_layer(inputs)
        emb_out = self.tanh(emb_out)

        emb_out = torch.matmul(emb_out, weight_0)
        emb_out = self.tanh(emb_out)

        emb_out = torch.matmul(emb_out, weight_1)
        emb_out = self.tanh(emb_out)

        emb_out = torch.matmul(emb_out, weight_2)
        emb_out = self.tanh(emb_out)

        emb_out = self.end_layer(emb_out)
        return emb_out

    def forward(self, x, t):
        if self.phase == 'phase1':
            return self.forward_phase1(x, t)
        elif self.phase == 'phase2':
            return self.forward_phase2(x, t)

    def get_phase2_weights(self, meta_param):
        """Extracts weights from Phase1 to initialize Phase2"""
        # Meta-learning to get the alphas (same as in Phase1 forward)
        meta_output = self.meta_layer_1(meta_param.unsqueeze(-1))
        meta_output = self.tanh(meta_output)
        meta_output = self.meta_layer_2(meta_output)
        meta_output = self.tanh(meta_output)
        meta_output = self.meta_layer_3(meta_output)
        meta_output = self.tanh(meta_output)

        meta_alpha_0_output = self.relu(self.meta_alpha_0(meta_output))
        meta_alpha_1_output = self.relu(self.meta_alpha_1(meta_output))
        meta_alpha_2_output = self.relu(self.meta_alpha_2(meta_output))

        alpha_0 = meta_alpha_0_output.squeeze()
        alpha_1 = meta_alpha_1_output.squeeze()
        alpha_2 = meta_alpha_2_output.squeeze()

        # Collect weights for Phase2 initialization
        return {
            'start_w': self.start_layer.weight.data.clone(),
            'start_b': self.start_layer.bias.data.clone(),
            'end_w': self.end_layer.weight.data.clone(),
            'end_b': self.end_layer.bias.data.clone(),
            'col_0': self.col_basis_0.data.clone(),
            'col_1': self.col_basis_1.data.clone(),
            'col_2': self.col_basis_2.data.clone(),
            'row_0': self.row_basis_0.data.clone(),
            'row_1': self.row_basis_1.data.clone(),
            'row_2': self.row_basis_2.data.clone(),
            'alpha_0': alpha_0,
            'alpha_1': alpha_1,
            'alpha_2': alpha_2
        }

def create_phase2_model(phase1_model, meta_param, hidden_dim):
    """
    Creates and returns an LR_PINN model for phase 2 using weights from the phase 1 model.

    Args:
    - phase1_model (nn.Module): The trained Phase 1 model.
    - meta_param (torch.Tensor): The meta-parameters for phase 2.
    - hidden_dim (int): The dimension of the hidden layer.

    Returns:
    - LR_PINN: An instance of the LR_PINN model for phase 2.
    """
    # Get phase 2 weights from the phase 1 model using meta parameters
    phase2_weights = phase1_model.get_phase2_weights(meta_param)

    # Initialize the Phase 2 model with the retrieved weights
    phase2_model = LR_PINN(
        hidden_dim=hidden_dim,
        phase='phase2',
        start_w=phase2_weights['start_w'],
        start_b=phase2_weights['start_b'],
        end_w=phase2_weights['end_w'],
        end_b=phase2_weights['end_b'],
        col_0=phase2_weights['col_0'],
        col_1=phase2_weights['col_1'],
        col_2=phase2_weights['col_2'],
        row_0=phase2_weights['row_0'],
        row_1=phase2_weights['row_1'],
        row_2=phase2_weights['row_2'],
        alpha_0=phase2_weights['alpha_0'],
        alpha_1=phase2_weights['alpha_1'],
        alpha_2=phase2_weights['alpha_2']
    )

    return phase2_model
