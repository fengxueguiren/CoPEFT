import torch
import torch.nn as nn
from opencood.models.fuse_modules.fusion_in_one import regroup


class CoPrompt(nn.Module):

    def __init__(self, hidden_size, down_ratio=4, bias = True, dropout_p = 0.1) -> None:
        """
        Initializes the CoPrompt (CoPEFT) module.

        Parameters:
        - hidden_size (int): The number of features in the input tensor.
        - down_ratio (int, optional): The downsampling ratio for the convolutional layers. Default is 4.
        - bias (bool, optional): Whether to use bias in the convolutional layers. Default is True.
        - dropout_p (float, optional): The dropout probability. Default is 0.1.

        Returns:
        - None
        """
        
        super().__init__()
        
        self.prompt_num = 1
        self.hidden_size = hidden_size 
        self.down_ratio = down_ratio
        self.bias = bias
        self.dropout_p = dropout_p

        # sst for c. prompt
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.shift = nn.Parameter(torch.zeros(hidden_size))

        # c. adapter and a. prompt
        self.spacial_attention = nn.Conv2d(hidden_size, 1, 1, bias=self.bias)
        self.down_layer = nn.Conv2d(hidden_size, hidden_size // self.down_ratio, 1, bias=self.bias)
        self.activation = nn.ReLU()
        self.up_layer = nn.Conv2d(hidden_size // self.down_ratio, hidden_size, 1, bias=self.bias)
        self.pro_prompt = nn.Linear(hidden_size, hidden_size, bias=self.bias)

        self.reset_parameters_coprompt()

    def reset_parameters_coprompt(self) -> None:

        nn.init.kaiming_normal_(
            self.down_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(
            self.up_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(
            self.spacial_attention.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.xavier_normal_(self.pro_prompt.weight)

        if self.bias:
            nn.init.zeros_(self.down_layer.bias)
            nn.init.zeros_(self.up_layer.bias)
            nn.init.zeros_(self.spacial_attention.bias)
            # nn.init.zeros_(self.pro_prompt.bias)

    def forward(self, x, record_len=None, only_adapter=False):
        """
        Performs the forward pass of the CoPrompt module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (N, C, W, H).
        - record_len (torch.Tensor, optional): A tensor containing the lengths of each record. Default is None.
        - only_adapter (bool, optional): A flag indicating whether to only perform the adapter operation. Default is False.

        Returns:
        - new_x (torch.Tensor): The output tensor after applying the CoPrompt module.
        - con_prompts (list of torch.Tensor): A list of contextual prompt tensors.
        - self.prompt_num (int): The number of prompts used.
        """

        N, C, W, H = x.shape # [N, 64, 200, 504]
  
        x_temp = x.clone()
        if only_adapter is False:
            x_local_regroup = regroup(x, record_len)
            co_x_local = torch.cat([x.max(dim=0, keepdim=True)[0].repeat(record_len[i].item(), 1, 1, 1) for i, x in enumerate(x_local_regroup)], dim=0)
            spacial_attention = self.spacial_attention(co_x_local)
        else:
            spacial_attention = self.spacial_attention(x)

        x_down = self.activation(self.down_layer(x))
        x_down = nn.functional.dropout(x_down, p=self.dropout_p, training=self.training)
        x_result = self.up_layer(x_down)

        new_x = spacial_attention * x_result + x_temp

        if only_adapter is False:
            con_prompt = self.scale.view(1, -1, 1, 1) * new_x + self.shift.view(1, -1, 1, 1)
            con_prompt_regroup = regroup(con_prompt, record_len)
            con_prompts = [x.max(dim=0, keepdim=True)[0] for x in con_prompt_regroup]

            con_prompts = [self.pro_prompt(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) for x in con_prompts]
            
            return new_x, con_prompts, self.prompt_num
        else:
            return new_x

if __name__ == "__main__":

    test_input = torch.rand(size=(5, 64, 128, 128))
    N, C, W, H = test_input.shape

    test_prompt = CoPrompt(hidden_size=C)
    test_output = test_prompt(test_input, torch.tensor([3,2]))

    print(test_output[1][0].shape)
