import torch
from torch.nn import functional as F

def loss_fn(model, inputs, f, parameter, solution_shape):
    x, y = inputs
    x.requires_grad = True
    y.requires_grad = True

    # Forward pass
    u = model(x, y)
    u_reshaped = u.reshape(solution_shape)

    # Compute first derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    # Compute second-order derivatives (Laplacian)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    laplacian_u = u_xx + u_yy

    # Compute residual (PDE loss)
    residual = F.mse_loss(laplacian_u, f(x, y, parameter))

    # Extract boundary values from u_reshaped
    u_boundary_top = u_reshaped[0, :]        # Top row
    u_boundary_bottom = u_reshaped[-1, :]    # Bottom row
    u_boundary_left = u_reshaped[:, 0]       # Left column
    u_boundary_right = u_reshaped[:, -1]     # Right column

    # Combine boundary values
    u_boundary_values = torch.cat([
        u_boundary_top,
        u_boundary_bottom,
        u_boundary_left,
        u_boundary_right
    ], dim=0)

    # Compute boundary loss
    boundary_loss = F.mse_loss(u_boundary_values, torch.zeros_like(u_boundary_values)
    )

    # Combine the losses
    total_loss = residual + boundary_loss

    return total_loss

# Get source function based on a key
def get_source_function(key):
    match key:
        case 'bischof':
            return lambda x, y, c: torch.ones_like(x) / c
        case 'desai':
            return lambda x, y, c: torch.sin(c * torch.pi * x) * torch.sin(c * torch.pi * y)
        case 'desai_song':
            return lambda x, y, c: 2 * torch.pi ** 2 * torch.sin(c * torch.pi * x) * torch.sin(c * torch.pi * y)
        case _:
            raise ValueError(f"Unknown key: {key}")
