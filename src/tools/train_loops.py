from torch import optim
import time
from .tools import print_trainable_parameters, save_comparison_plot, save_error_plot, evaluate_model
import sys
sys.path.append("../")
from models.hyper_lr_pinn import create_phase2_model
import numpy as np

def pre_train(model, inputs, f, parameters, solution_shape, loss_fn, inner_epochs, outer_epochs, lr=0.001, print_interval=50):
    model.train()
    device = inputs[0].device
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for oe in range(outer_epochs):
        print(f"[INFO] Outer Epoch: {oe + 1}/{outer_epochs}")
        for param_idx, param in enumerate(parameters):
            #print(f"[INFO] Training parameter {param_idx + 1}/{len(parameters)}: {param}")
            #model.set_task(param.repeat(inputs[0].size(0), 1), device)
            model.set_task(param.reshape(-1, 1), device)
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            for ie in range(inner_epochs):
                optimizer.zero_grad()
                # Compute loss
                loss = loss_fn(model, inputs, f, param, solution_shape)
                # Backpropagation
                loss.backward()
                optimizer.step()
                # Print loss at specified intervals
                if (ie + 1) % print_interval == 0:
                    print(f'[Inner Epoch] [{ie + 1}/{inner_epochs}], Loss: {loss.item():.4f}')

        if (oe + 1) % print_interval == 0:
            print(f'[Outer Epoch] [{oe + 1}/{outer_epochs}] Completed')

    print("[INFO] Training completed.")
    return model

def fine_tune(model, inputs, f, loss_fn, parameters, epochs, solutions, solution_shape, lr, output_dir, print_interval=200):
    model.train()
    device = inputs[0].device

    X_test = inputs[0].reshape(solution_shape)
    Y_test = inputs[1].reshape(solution_shape)

    rmse = []
    rl2 = []
    losses = []
    n_collocation = inputs[0].size(0)

    # -- benchmark the time it takes to reach a target loss
    epochs_2_target = []
    time_2_target = []
    target_loss = 0.05
    
    time_per_param = []

    for idx, parameter in enumerate(parameters):
        model.reinitialize_weights()
        print("model-reinit")
        _ = print_trainable_parameters(model)
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        loss_per_parameter = []
        target_recorded = False
        
        
        time_start = time.time()
        for e  in range(epochs):
            optimizer.zero_grad()
            # Compute loss
            loss = loss_fn(model, inputs, f, parameter, solution_shape)
            # Backpropagation
            loss.backward()
            optimizer.step()
            # Print loss at specified intervals
            if (e + 1) % print_interval == 0:
                print(f'[Inner Epoch] [{e + 1}/{epochs}], Loss: {loss.item():.4f}')
            loss_per_parameter.append(loss.item())

            if target_loss > loss.item() and not target_recorded:
                target_recorded = True
                time_2_target.append(time.time() - time_start)
                epochs_2_target.append(e)
                print(f"[INFO] target reached, epochs: {e}, time: {time_2_target[-1]}")

            if not target_recorded and e == (epochs - 1):
                print("[INFO] target not reached")
                time_2_target.append(time.time() - time_start)
                epochs_2_target.append(e)

        time_per_param.append(time.time() - time_start)
        print("[INFO] time per parameter: ", time.time()-time_start)
        
        # save plots
        save_comparison_plot(model, X_test, Y_test, solutions[idx].to(device), device, output_dir, f"u_pred_vs_u_gt_{parameter:3f}.png")
        save_error_plot(model, X_test, Y_test, solutions[idx].to(device), device, output_dir, f"error_plot_{parameter:.3f}.png")
        rl2.append(evaluate_model(model, inputs[0], inputs[1], solutions[idx], error_type="rl2"))
        rmse.append(evaluate_model(model, inputs[0], inputs[1], solutions[idx], error_type="rmse"))
        losses.append(loss_per_parameter)
        print("[INFO] rmse: ", rmse[-1])
        print("[INFO] rl2: ", rl2[-1])

    average_time  = np.array(time_per_param).mean() 
    model_parameters = print_trainable_parameters(model)
    metrics = {
        "parameters": parameters,
        #"solver_time": solver_time,
        "loss": losses,
        "rmse_error": rmse,
        "rl2_error": rl2,
        "time": average_time,
        "model": model_parameters,
        "points": n_collocation,
        "epochs": epochs,
        "time2target": time_2_target,
        "epochs2target": epochs_2_target
        }

    return model, metrics

def fine_tune_meta(model, inputs, f, loss_fn, parameters, epochs, solutions, solution_shape, lr, output_dir, print_interval=200):
    base_model = model
    device = inputs[0].device

    X_test = inputs[0].reshape(solution_shape)
    Y_test = inputs[1].reshape(solution_shape)
    
    inputs[0] = X_test[::2].reshape(-1, 1)
    inputs[1] = Y_test[::2].reshape(-1, 1)
    reduced_solution_shape = X_test[::2].shape

    rmse = []
    rl2 = []
    losses = []
    n_collocation = inputs[0].size(0)
    # -- benchmark the time it takes to reach a target loss
    epochs_2_target = []
    time_2_target = []
    target_loss = 0.05

    time_per_param = []

    for idx, parameter in enumerate(parameters):
        # -- initialize model
        model = create_phase2_model(base_model, parameter.to(device), base_model.hidden_dim, base_model.rank)
        _ = print_trainable_parameters(model)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        loss_per_parameter = []
        target_recorded = False
        
        time_start = time.time()
        for e  in range(epochs):
            optimizer.zero_grad()
            # Compute loss
            loss = loss_fn(model, inputs, f, parameter, reduced_solution_shape)
            # Backpropagation
            loss.backward()
            optimizer.step()
            # Print loss at specified intervals
            if (e + 1) % print_interval == 0:
                print(f'[Inner Epoch] [{e + 1}/{epochs}], Loss: {loss.item():.4f}')
            loss_per_parameter.append(loss.item())
            
            if target_loss > loss.item() and not target_recorded:
                target_recorded = True
                time_2_target.append(time.time() - time_start)
                epochs_2_target.append(e)
                print(f"[INFO] target reached, epochs: {e}, time: {time_2_target[-1]}")

            if not target_recorded and e == (epochs - 1):
                print("[INFO] target not reached")
                time_2_target.append(time.time() - time_start)
                epochs_2_target.append(e)

        time_per_param.append(time.time() - time_start)
        print("[INFO] time per parameter: ", time.time()-time_start)

        # save plots
        save_comparison_plot(model, X_test, Y_test, solutions[idx].to(device), device, output_dir, f"u_pred_vs_u_gt_{parameter:3f}.png")
        save_error_plot(model, X_test, Y_test, solutions[idx].to(device), device, output_dir, f"error_plot_{parameter:.3f}.png")
        rl2.append(evaluate_model(model, X_test.reshape(-1, 1), Y_test.reshape(-1, 1), solutions[idx], error_type="rl2"))
        rmse.append(evaluate_model(model, X_test.reshape(-1, 1), Y_test.reshape(-1, 1), solutions[idx], error_type="rmse"))
        losses.append(loss_per_parameter)
        print("[INFO] rmse: ", rmse[-1])
        print("[INFO] rl2: ", rl2[-1])

    average_time  = np.array(time_per_param).mean() 
    model_parameters = print_trainable_parameters(model)
    metrics = {
        "parameters": parameters,
        #"solver_time": solver_time,
        "loss": losses,
        "rmse_error": rmse,
        "rl2_error": rl2,
        "time": average_time,
        "model": model_parameters,
        "points": n_collocation,
        "epochs": epochs,
        "time2target": time_2_target,
        "epochs2target": epochs_2_target
        }

    return model, metrics
