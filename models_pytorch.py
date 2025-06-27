# -*- coding: utf-8 -*-
"""
PyTorch implementation of PINN models
Converted from TensorFlow implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from compute_jacobian_pytorch import jacobian, jacobian_parameters
from typing import List, Tuple, Optional


class Sampler:
    """Data sampler for boundary conditions and residual points"""
    
    def __init__(self, dim: int, coords: np.ndarray, func, name: Optional[str] = None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample N points from the domain"""
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y


class NN(nn.Module):
    """Vanilla MLP for PINN"""
    
    def __init__(self, layers: List[int], bcs_samplers: List[Sampler], 
                 res_samplers: Sampler, u, a: float, b: float, sigma: float, 
                 device: str = 'cpu'):
        super(NN, self).__init__()
        
        self.device = torch.device(device)
        self.layers = layers
        self.bcs_samplers = bcs_samplers
        self.res_samplers = res_samplers
        self.a = a
        self.b = b
        
        # Initialize normalization parameters
        X, _ = res_samplers.sample(int(1e5))
        self.mu_X = torch.tensor(X.mean(0), dtype=torch.float32, device=self.device)
        self.sigma_X = torch.tensor(X.std(0), dtype=torch.float32, device=self.device)
        self.mu_x = self.mu_X[0]
        self.sigma_x = self.sigma_X[0]
        
        # Build network layers
        self.network = self._build_network(layers)
        self.to(self.device)
        
        # Test data
        N_test = 1000
        self.X_star = np.linspace(0, 1, N_test)[:, None]
        self.u_star = u(self.X_star, a, b)
        
        # Logging
        self.loss_bcs_log = []
        self.loss_res_log = []
        self.l2_error_log = []
        
    def _build_network(self, layers: List[int]) -> nn.Sequential:
        """Build the neural network architecture"""
        network_layers = []
        
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            # Xavier initialization
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            network_layers.append(linear)
            
            if i < len(layers) - 2:  # No activation on output layer
                network_layers.append(nn.Tanh())
                
        return nn.Sequential(*network_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)
    
    def net_u(self, x: torch.Tensor) -> torch.Tensor:
        """Network prediction for u"""
        x.requires_grad_(True)
        u = self.forward(x)
        return u
    
    def net_r(self, x: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual"""
        x.requires_grad_(True)
        u = self.net_u(x)
        
        # Compute first derivative
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                 create_graph=True, retain_graph=True)[0] / self.sigma_x
        
        # Compute second derivative  
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                                  create_graph=True, retain_graph=True)[0] / self.sigma_x
        
        return u_xx
    
    def fetch_minibatch(self, sampler: Sampler, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetch a minibatch from the sampler"""
        X, Y = sampler.sample(N)
        X = (X - self.mu_X.cpu().numpy()) / self.sigma_X.cpu().numpy()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.float32, device=self.device)
        return X, Y
    
    def compute_loss(self, X_bc1: torch.Tensor, u_bc1: torch.Tensor,
                    X_bc2: torch.Tensor, u_bc2: torch.Tensor,
                    X_res: torch.Tensor, r_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss"""
        # Boundary losses
        u_bc1_pred = self.net_u(X_bc1)
        u_bc2_pred = self.net_u(X_bc2)
        
        loss_bc1 = F.mse_loss(u_bc1_pred, u_bc1)
        loss_bc2 = F.mse_loss(u_bc2_pred, u_bc2)
        loss_bcs = loss_bc1 + loss_bc2
        
        # Residual loss
        r_pred = self.net_r(X_res)
        loss_res = F.mse_loss(r_pred, r_target)
        
        # Total loss
        total_loss = loss_res + loss_bcs
        
        return total_loss, loss_bcs, loss_res
    
    def train_model(self, nIter: int = 10000, batch_size: int = 128, 
                   learning_rate: float = 1e-3, log_interval: int = 1000):
        """Train the PINN model"""
        # Setup optimizer with learning rate decay
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        start_time = time.time()
        
        for it in range(nIter):
            optimizer.zero_grad()
            
            # Fetch minibatches
            X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_samplers[0], batch_size)
            X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_samplers[1], batch_size)
            X_res_batch, f_batch = self.fetch_minibatch(self.res_samplers, batch_size)
            
            # Compute loss
            total_loss, loss_bcs, loss_res = self.compute_loss(
                X_bc1_batch, u_bc1_batch, X_bc2_batch, u_bc2_batch, 
                X_res_batch, f_batch
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Update learning rate
            if it % 1000 == 0 and it > 0:
                scheduler.step()
            
            # Logging
            if it % log_interval == 0:
                elapsed = time.time() - start_time
                
                # Compute test error
                u_pred = self.predict_u(self.X_star)
                error_u = np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)
                
                self.loss_bcs_log.append(loss_bcs.item())
                self.loss_res_log.append(loss_res.item())
                self.l2_error_log.append(error_u)
                
                print(f'It: {it}, Loss: {total_loss.item():.3e}, '
                      f'Loss_bcs: {loss_bcs.item():.3e}, '
                      f'Loss_res: {loss_res.item():.3e}, Time: {elapsed:.2f}')
                
                start_time = time.time()
    
    def predict_u(self, X_star: np.ndarray) -> np.ndarray:
        """Predict u at test points"""
        X_star_norm = (X_star - self.mu_X.cpu().numpy()) / self.sigma_X.cpu().numpy()
        X_star_tensor = torch.tensor(X_star_norm, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            u_pred = self.net_u(X_star_tensor)
        
        return u_pred.cpu().numpy()
    
    def predict_r(self, X_star: np.ndarray) -> np.ndarray:
        """Predict residual at test points"""
        X_star_norm = (X_star - self.mu_X.cpu().numpy()) / self.sigma_X.cpu().numpy()
        X_star_tensor = torch.tensor(X_star_norm, dtype=torch.float32, device=self.device)
        
        r_pred = self.net_r(X_star_tensor)
        return r_pred.detach().cpu().numpy()


class NN_FF(NN):
    """Neural Network with Fourier Features"""
    
    def __init__(self, layers: List[int], bcs_samplers: List[Sampler], 
                 res_samplers: Sampler, u, a: float, b: float, sigma: float, 
                 device: str = 'cpu'):
        
        # Call parent init but don't build network yet
        super().__init__(layers, bcs_samplers, res_samplers, u, a, b, sigma, device)
        
        # Initialize Fourier features (fixed, not trainable)
        self.W = torch.randn(1, layers[0] // 2, device=self.device) * sigma
        self.W.requires_grad = False
        
        # Rebuild network for Fourier features
        self.network = self._build_ff_network(layers)
        self.to(self.device)
    
    def _build_ff_network(self, layers: List[int]) -> nn.Sequential:
        """Build network with Fourier feature input"""
        network_layers = []
        
        # First layer takes Fourier features (input_dim -> layers[0])
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            network_layers.append(linear)
            
            if i < len(layers) - 2:
                network_layers.append(nn.Tanh())
                
        return nn.Sequential(*network_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Fourier feature encoding"""
        # Apply Fourier feature transformation
        ff = torch.cat([torch.sin(torch.matmul(x, self.W)), 
                       torch.cos(torch.matmul(x, self.W))], dim=1)
        
        return self.network(ff)


class NN_mFF(NN):
    """Neural Network with Multi-scale Fourier Features"""
    
    def __init__(self, layers: List[int], bcs_samplers: List[Sampler], 
                 res_samplers: Sampler, u, a: float, b: float, sigma: float, 
                 device: str = 'cpu'):
        
        super().__init__(layers, bcs_samplers, res_samplers, u, a, b, sigma, device)
        
        # Initialize two sets of Fourier features
        self.W1 = torch.randn(1, layers[0] // 2, device=self.device) * 1.0
        self.W2 = torch.randn(1, layers[0] // 2, device=self.device) * sigma
        self.W1.requires_grad = False
        self.W2.requires_grad = False
        
        # Rebuild network for multi-scale Fourier features
        self.network1, self.network2, self.final_layer = self._build_mff_network(layers)
        self.to(self.device)
    
    def _build_mff_network(self, layers: List[int]) -> Tuple[nn.Sequential, nn.Sequential, nn.Linear]:
        """Build network with multi-scale Fourier features"""
        # Two parallel networks
        network1_layers = []
        network2_layers = []
        
        # Build parallel networks (excluding final layer)
        for i in range(len(layers) - 2):
            # Network 1
            linear1 = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_uniform_(linear1.weight)
            nn.init.zeros_(linear1.bias)
            network1_layers.extend([linear1, nn.Tanh()])
            
            # Network 2  
            linear2 = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_uniform_(linear2.weight)
            nn.init.zeros_(linear2.bias)
            network2_layers.extend([linear2, nn.Tanh()])
        
        # Final layer that combines both networks
        final_layer = nn.Linear(2 * layers[-2], layers[-1])
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        
        return (nn.Sequential(*network1_layers), 
                nn.Sequential(*network2_layers), 
                final_layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale Fourier features"""
        # Apply two different Fourier feature transformations
        ff1 = torch.cat([torch.sin(torch.matmul(x, self.W1)), 
                        torch.cos(torch.matmul(x, self.W1))], dim=1)
        ff2 = torch.cat([torch.sin(torch.matmul(x, self.W2)), 
                        torch.cos(torch.matmul(x, self.W2))], dim=1)
        
        # Pass through parallel networks
        h1 = self.network1(ff1)
        h2 = self.network2(ff2)
        
        # Concatenate and pass through final layer
        h_combined = torch.cat([h1, h2], dim=1)
        return self.final_layer(h_combined)