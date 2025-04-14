import os 
from os.path import join

import torch
from torchvision import transforms

def feature_visualization(tensor_feature, saved_path='/data/code/CoPEFT/opencood/logs/images/', max_munber=10, mode='max'):
    """
    Visualize the features extracted from a neural network.

    Parameters:
    tensor_feature (torch.Tensor): The tensor containing the features to visualize. It should have a shape of (B, C, H, W), where B is the batch size, C is the number of channels, H is the height, and W is the width.
    saved_path (str, optional): The directory path where the visualized images will be saved. Defaults to None.
    max_munber (int, optional): The maximum number of images to visualize. Defaults to 10.
    mode (str, optional): The mode of visualization. It can be either 'max' or 'mean'. If 'max', the maximum value along the channel dimension will be used for visualization. If 'mean', the mean value along the channel dimension will be used for visualization. Defaults to 'max'.

    Returns:
    None
    """     
    B, C, H, W = tensor_feature.shape

    if saved_path is not None:
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

    saver = transforms.ToPILImage()

    if mode == 'max':
        feature_vis = tensor_feature.max(dim=1, keepdim=True)[0].cpu()
        feature_vis = torch.cat([feature_vis, feature_vis,feature_vis], dim=1)
    elif mode == 'mean':
        feature_vis = tensor_feature.mean(dim=1, keepdim=True).cpu()
        feature_vis = torch.cat([feature_vis, feature_vis,feature_vis], dim=1)
    elif mode == 'all':
        feature_vis2 = tensor_feature.mean(dim=1, keepdim=True).cpu()
        feature_vis3 = tensor_feature.sum(dim=1, keepdim=True).cpu()
        feature_vis1 = tensor_feature.max(dim=1, keepdim=True)[0].cpu()
        feature_vis = torch.cat([feature_vis1, feature_vis2,feature_vis3], dim=1)
    elif mode == 'channel':
        feature_vis = tensor_feature.cpu()
        
    else:
        raise NotImplementedError('Unsupported mode')

    for i in range(B):
        if mode == 'channel':
            for j in range(C):
                if i < max_munber:
                    file_name = f'{i}_{j}.jpg'
                    if saved_path is None:
                        pass
                    else:
                        file_name = join(saved_path, file_name)
                    saver(feature_vis[i][j:j+1]).save(file_name)
        else:
            if i < max_munber:
                file_name = f'{i}.jpg'
                if saved_path is None:
                    pass
                else:
                    file_name = join(saved_path, file_name)
                saver(feature_vis[i]).save(file_name)
            else:
                break


if __name__ == '__main__':
    feature_visualization(torch.rand(size=(3, 3, 64, 100)), mode='channel')