# -*- coding: utf-8 -*-
# Author: Quanmin Wei
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn

def set_requires_grad(model:nn.Module, layers_to_finetune:list, print_info:bool=True):
    """
    This function sets the requires_grad attribute of the parameters in a PyTorch model.
    It is commonly used for fine-tuning a pre-trained model by freezing some layers and training others.

    Parameters:
    model (nn.Module): The PyTorch model to set the requires_grad attribute for.
    layers_to_finetune (list of str): A list of layer names in the model that should have their requires_grad attribute set to True.
    print_info (bool, optional): If True, the function will print the statistics of trainable parameters. Defaults to True.

    Returns:
    nn.Module: The same model with the requires_grad attribute of the specified layers set to True.
    """
    for name, param in model.named_parameters():
        param.requires_grad = False  # default is False
        for layer in layers_to_finetune:
            if layer in name:
                param.requires_grad = True  # set True in layers_to_finetune
    
    if print_info:
        get_trainable_parameters(model, print_info)
        get_parameters_statistic(model, print_info)
    
    return model

def get_trainable_parameters(model:nn.Module, print_info:bool=True) -> list:
    """
    This function retrieves a list of names of trainable parameters in a PyTorch model.

    Parameters:
    model (nn.Module): The PyTorch model to retrieve trainable parameters from.
    print_info (bool, optional): If True, the function will print the list of trainable parameters. Defaults to True.

    Returns:
    list: A list of names of trainable parameters in the model.
    """

    # Using list comprehension to get the names of trainable parameters
    trainable_parameters = [(name) for name, para in model.named_parameters() if para.requires_grad is True]
    
    # Printing the list of trainable parameters if print_info is True
    if print_info:
        print("All trainable parameters: ", trainable_parameters)

    # Returning the list of trainable parameters
    return trainable_parameters

def get_parameters_statistic(model:nn.Module, print_info:bool=True) -> tuple:
    """
    This function calculates and optionally prints the total number of parameters, 
    the total number of trainable parameters, and the ratio of trainable parameters 
    to the total parameters in a PyTorch model.

    Parameters:
    model (nn.Module): The PyTorch model to calculate the statistics for.
    print_info (bool, optional): If True, the function will print the statistics. Defaults to True.

    Returns:
    tuple: A tuple containing the total number of parameters, the total number of trainable parameters, 
    and the ratio of trainable parameters to the total parameters.
    """

    total_params = sum(para.numel() for para in model.parameters())
    total_trainable_params = sum(para.numel() for para in model.parameters() if para.requires_grad)
    trainable_ratio = total_trainable_params / total_params

    if print_info is True:
        print(f"Total parameters: {total_params} || Total trainable parameters: {total_trainable_params} || Trainable parameters ratio: {trainable_ratio:.4%}")
    
    return total_params, total_trainable_params, trainable_ratio


def get_param_groups(model:nn.Module, layers_to_lr_times:list, base_lr:float, lr_times:int, print_info:bool=True):
    """
    This function creates a list of parameter groups for a PyTorch model, 
    which can be used for different learning rates during training.

    Parameters:
    model (nn.Module): The PyTorch model to create parameter groups for.
    layers_to_lr_times (list): A list of layer names for which the learning rate should be multiplied by 'lr_times'.
    base_lr (float): The base learning rate for all parameters.
    lr_times (int): The multiplier for the learning rate of the layers specified in 'layers_to_lr_times'.

    Returns:
    list: A list of dictionaries, where each dictionary contains a 'params' key (a list of parameters) and an 'lr' key (the learning rate for those parameters).
    """

    param_groups = []
    finetune_params_times = []
    finetune_params_no_times = []

    param_groups_name = []
    finetune_params_times_name = []
    finetune_params_no_times_name = []

    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_lr_times):
            finetune_params_times_name.append(name)
            finetune_params_times.append(param)
        else:
            finetune_params_no_times_name.append(name)
            finetune_params_no_times.append(param)

    param_groups.append({'params': finetune_params_no_times, 'lr': base_lr})
    param_groups.append({'params': finetune_params_times, 'lr': base_lr * lr_times})

    param_groups_name.append({'params': finetune_params_no_times_name, 'lr': base_lr})
    param_groups_name.append({'params': finetune_params_times_name, 'lr': base_lr * lr_times})

    if print_info:
        print("The param_groups information:")
        for i in range(len(param_groups_name)):
            print(f"The param_groups of lr {param_groups_name[i]['lr']} is: ", param_groups_name[i])

    return param_groups

if __name__ == "__main__":
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.layer1 = nn.Linear(10, 50)
            self.layer2 = nn.Linear(50, 20)
            self.layer3 = nn.Linear(20, 10)
            self.layer4 = nn.Linear(10, 2)
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = torch.relu(self.layer3(x))
            x = self.layer4(x)
            return x
    model = MyModel()

    layers_to_finetune = ['layer2', 'layer4']
    # layers_to_finetune = ['']

    model = set_requires_grad(model, layers_to_finetune=layers_to_finetune)
    get_trainable_parameters(model)
    model.train()
    get_trainable_parameters(model)
    model.eval()
    get_trainable_parameters(model)
    # param_groups = get_param_groups(model, ['layer1', 'layer23'], base_lr=0.05, lr_times=10)

    # print(param_groups)
