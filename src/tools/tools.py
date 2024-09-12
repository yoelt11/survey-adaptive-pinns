import numpy as np
import torch
from matplotlib import pyplot as plt
import os

def print_trainable_parameters(model):
    """
    Prints the name, shape, and number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
    """
    total_params = 0
    print(f"{'Parameter Name':<40} {'Shape':<30} {'# of Parameters':<20}")
    print("-" * 90)

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_shape = tuple(param.shape)
            param_count = param.numel()
            total_params += param_count
            print(f"{name:<40} {str(param_shape):<30} {param_count:<20}")

    print("-" * 90)
    print(f"Total Trainable Parameters: {total_params:,}")
    return total_params

def save_comparison_plot(model, X_test, Y_test, u_test, device, save_dir, filename, show_fig=False):
    inputs = [X_test.flatten(0).unsqueeze(-1).to(device), Y_test.flatten(0).unsqueeze(-1).to(device)]
    u_pred = model(inputs[0], inputs[1]).reshape(X_test.shape)
    u_gt = u_test
    with torch.no_grad():
        fig, axs = plt.subplots(1, 2, figsize=(10,4))
        _, contour1 = plot2D_ax(u_pred.cpu().numpy(), X_test.cpu().numpy(), Y_test.cpu().numpy(), axs[0], title='[Vanilla PINN] - u predicted')
        _, contour2 = plot2D_ax(u_gt.cpu().numpy(), X_test.cpu().numpy(), Y_test.cpu().numpy(), axs[1], title='[Solver] - u ground truth')
        fig.colorbar(contour1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label='Value')
        file_path = os.path.join(save_dir, filename)
        if show_fig:
            plt.show()
        fig.savefig(file_path)
        plt.close()

def save_error_plot(model, X_test, Y_test, u_test, device, save_dir, filename, show_fig=False):
    inputs = [X_test.flatten(0).unsqueeze(-1).to(device), Y_test.flatten(0).unsqueeze(-1).to(device)]
    u_pred = model(inputs[0], inputs[1]).reshape(X_test.shape)
    u_gt = u_test

    with torch.no_grad():
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        error = np.abs((u_pred.to(device) - u_gt.to(device)).cpu().numpy())
        ax, contour = plot2D_ax(error, X_test.cpu().numpy(), Y_test.cpu().numpy(), ax, title="[Error Plot]  u_pred - u_gt", cmap='inferno')
        plt.colorbar(contour, ax=ax)
        file_path = os.path.join(save_dir, filename)
        if show_fig:
            plt.show()
        fig.savefig(file_path)
        plt.close()

def evaluate_model(model, x, y, u_gt, type="rmse"):
    model.eval()
    inputs = [x.flatten(0).unsqueeze(-1), y.flatten(0).unsqueeze(-1)]
    u_pred = model(inputs[0], inputs[1]).reshape(u_gt.shape)
    match type:
        case "rmse":
            error = calculate_rmse(u_pred, u_gt)
        case "rl2":
            error = calculate_relative_l2_error(u_pred, u_gt)
        case "l2_error":
            error = calculate_l2_error(u_pred, u_gt)
        case _:
            error = None
    return

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_l2_error(u_pred, u_gt):
    error = torch.norm(u_pred - u_gt, p=2)  # Compute the L2 norm of the error
    l2_error = error.item()  # Convert the tensor to a Python float
    return l2_error

def calculate_relative_l2_error(u_pred, u_gt):
    error = torch.norm(u_gt - u_pred, p=2)  # Compute the L2 norm of the error
    true_norm = torch.norm(u_gt, p=2)
    relative_l2_error = error.item()/ true_norm.item()  # Convert the tensor to a Python float
    return relative_l2_error

def calculate_rmse(u_pred, u_gt):
    return torch.sqrt(torch.mean((u_pred - u_gt) ** 2)).item()

def plot2D_ax(u, XX, YY, ax, resolution=100, labels=['x', 'y'], title='u domain', cmap='rainbow'):
    # -- plot figure
    contour = ax.contourf(XX, YY, u, levels=50, cmap=cmap)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    # -- return the figure
    return ax, contour

def sample_parameters_from_folder(dir, N, idx_start=0, type="train"):
    try:
        # List all items in the directory that start with the specified type
        items = sorted([e for e in os.listdir(dir) if e.startswith(type)])
        # Count the number of items
        num_items = len(items)
        step = max(1, num_items // N)
        selected_files = items[idx_start::step]

        # Extract parameters from filenames
        params_list = []
        for file in selected_files:
            # Remove the type prefix and file extension
            stripped_name = file[len(type)+1:].rsplit('.', 1)[0]
            # Split the remaining part into components
            components = stripped_name.split('_')
            # Extract the numerical values
            params = []
            for i in range(1, len(components), 2):
                try:
                    params.append(float(components[i]))
                except ValueError:
                    pass  # Handle cases where the component is not a float
            params_list.append(params)

        return params_list, selected_files

    except FileNotFoundError:
        return "Directory not found."
    except PermissionError:
        return "You don't have permission to access this directory."

def plot_parameter_points(mu_train, mu_test, save_dir, filename, show_fig=False):
    # Plotting
    fig = plt.figure(figsize=(6, 2))
    plt.plot(mu_train, np.zeros_like(mu_train), 'o', color='orange', label='Training Points')  # red dots for training points
    plt.plot(mu_test, np.zeros_like(mu_test), '*', color='violet', label='Testing Points')  # blue stars for testing points
    # Add a horizontal line at y=0
    plt.axhline(0, color='gray', linewidth=0.5)
    # Add labels and legend
    plt.xlabel('Parameter Value')
    plt.title('[Parameters] Training and Testing Points')
    plt.legend()
    file_path = os.path.join(save_dir, filename)
    if show_fig:
        plt.show()
    fig.savefig(file_path)
    plt.close()
