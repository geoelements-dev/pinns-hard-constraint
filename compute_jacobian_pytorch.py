# -*- coding: utf-8 -*-
"""
PyTorch implementation of Jacobian computation for PINNs
Converted from TensorFlow implementation
"""

import torch
import torch.nn.functional as F
from typing import List, Union


def jacobian(outputs: torch.Tensor, inputs: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """
    Computes jacobian of outputs w.r.t. inputs using PyTorch autograd.
    
    Args:
        outputs: A tensor of shape [batch_size, output_dim]
        inputs: A tensor of shape [batch_size, input_dim]  
        create_graph: Whether to create graph for higher-order derivatives
        
    Returns:
        Jacobian tensor of shape [batch_size, output_dim, input_dim]
    """
    batch_size, output_dim = outputs.shape
    input_dim = inputs.shape[1]
    
    jacobian_matrix = torch.zeros(batch_size, output_dim, input_dim, 
                                 dtype=outputs.dtype, device=outputs.device)
    
    for i in range(output_dim):
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[:, i] = 1.0
        
        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        jacobian_matrix[:, i, :] = gradients
    
    return jacobian_matrix


def jacobian_parameters(outputs: torch.Tensor, parameters: List[torch.Tensor], 
                       create_graph: bool = False) -> List[torch.Tensor]:
    """
    Computes jacobian of outputs w.r.t. model parameters.
    
    Args:
        outputs: Network outputs tensor
        parameters: List of model parameters (weights and biases)
        create_graph: Whether to create graph for higher-order derivatives
        
    Returns:
        List of jacobian tensors for each parameter
    """
    jacobians = []
    
    for param in parameters:
        if param.requires_grad:
            grad = torch.autograd.grad(
                outputs=outputs,
                inputs=param,
                grad_outputs=torch.ones_like(outputs),
                create_graph=create_graph,
                retain_graph=True,
                only_inputs=True
            )[0]
            jacobians.append(grad)
        else:
            jacobians.append(torch.zeros_like(param))
    
    return jacobians