# -*- coding: utf-8 -*-
# Author: Quanmin Wei, Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics
import tqdm
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import fine_tuning_utils
import glob

def train_parser():
    parser = argparse.ArgumentParser(description="CoPEFT")
    # The YAML configuration file of CoPEFT.
    parser.add_argument("--fine_tuning_hypes_yaml", "-fty", type=str, required=False,
                        default='/data/code/CoPEFT/opencood/hypes_yaml/dairv2x/fine_tuning/copeft/pointpillar_coalign_final.yaml',
                        help='CoPEFT yaml file needed ')
    # Deployment data availability ratio.
    parser.add_argument('--few_shot_rate', default=0.1, type=float,
                        help='few_shot_rate, -1 is the default means not to use this.')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()

    ###################CoPEFT fine tuning####################
    # load fine tuning hypes
    fine_tuning_hypes = yaml_utils.load_yaml(opt.fine_tuning_hypes_yaml)
    fine_tuning_flag = fine_tuning_hypes['fine_tuning_setting']['fine_tuning_flag']

    if fine_tuning_flag is True:
        opt.model_dir = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['pretrained_model_dir']
        # load model hyperparams
        hypes = yaml_utils.load_yaml(file=None, opt=opt)

        fine_tuning_mode = fine_tuning_hypes['fine_tuning_setting']['fine_tuning_mode']

        target_dataset_name = fine_tuning_hypes['fine_tuning_setting']['args']['dataset_args']['dataset_name']


        hypes['optimizer']['lr'] = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['lr']

        hypes['lr_scheduler'] = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['lr_scheduler']

        if target_dataset_name == 'dairv2x':
            hypes['fusion']['dataset'] = target_dataset_name
            hypes['data_dir'] = fine_tuning_hypes['fine_tuning_setting']['args']['dataset_args']['data_dir']
            hypes['root_dir'] = fine_tuning_hypes['fine_tuning_setting']['args']['dataset_args']['root_dir']
            hypes['validate_dir'] = fine_tuning_hypes['fine_tuning_setting']['args']['dataset_args']['validate_dir']
            hypes['test_dir'] = fine_tuning_hypes['fine_tuning_setting']['args']['dataset_args']['test_dir']
            if opt.few_shot_rate != -1:
                hypes['few_shot_rate'] = opt.few_shot_rate
        elif target_dataset_name == 'opv2v':
            hypes['fusion']['dataset'] = target_dataset_name

            hypes['root_dir'] = fine_tuning_hypes['fine_tuning_setting']['args']['dataset_args']['root_dir']
            hypes['validate_dir'] = fine_tuning_hypes['fine_tuning_setting']['args']['dataset_args']['validate_dir']
            hypes['test_dir'] = fine_tuning_hypes['fine_tuning_setting']['args']['dataset_args']['test_dir']
            if opt.few_shot_rate != -1:
                hypes['few_shot_rate'] = opt.few_shot_rate

        else:
            raise NotImplementedError("dataset not supported: {}".format(target_dataset_name))

    else:
        hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    layers_to_finetune = None
    print("#### fine_tuning mode is {}".format(fine_tuning_mode))
    if fine_tuning_mode == 'full':
        pass
    elif fine_tuning_mode == 'only_head':
        layers_to_finetune = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['layers_to_finetune']

    elif fine_tuning_mode == 'naive_adapter' or fine_tuning_mode == 'ssf' or fine_tuning_mode == 'ssf_adapter' or fine_tuning_mode == 'copeft' or fine_tuning_mode == 'naive_prompt':
        hypes['model']['core_method'] = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['ft_core_method']
        layers_to_finetune = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['layers_to_finetune']
    
    else:
        raise NotImplementedError("fine_tuning mode not supported: {}".format(fine_tuning_mode))
    
    lr_times_rate = None
    if 'lr_times' in fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']:
        lr_times_rate = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['lr_times']['lr_times_rate']
        layers_to_lr_times = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['lr_times']['layers_to_lr_times']

        base_lr = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['lr']
        
    ###################End ####################

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)
    
    # optimizer setup
    if lr_times_rate is None:
        optimizer = train_utils.setup_optimizer(hypes, model)
    else: 
        pass
    
    # if we want to train from last checkpoint.
    if opt.model_dir:
        init_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
        if fine_tuning_flag:
            few_shot_rate = hypes.get('few_shot_rate', None)
            if few_shot_rate is None:
                saved_path = train_utils.make_saved_path(dir_name=os.path.join(fine_tuning_mode, fine_tuning_hypes['name']), saved_path=opt.model_dir, time_mark=True, hypes=hypes, fthypes=fine_tuning_hypes)
            else:
                del hypes['few_shot_rate']
                saved_path = train_utils.make_saved_path(dir_name=os.path.join(fine_tuning_mode, "rate_"+str(few_shot_rate), fine_tuning_hypes['name']), saved_path=opt.model_dir, time_mark=True, hypes=hypes, fthypes=fine_tuning_hypes)

            init_epoch = 0
            print(f"fine tuning from {init_epoch} to {fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['train_params']['epoches']} epoch.")
            if lr_times_rate is not None:
                param_groups = fine_tuning_utils.get_param_groups(model, layers_to_lr_times, base_lr, lr_times_rate, print_info=True)
                optimizer = train_utils.setup_optimizer(hypes, model, param_groups)
        else:
            saved_path = opt.model_dir
            print(f"resume from {init_epoch} epoch.")
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    # else:
    #     init_epoch = 0
    #     # if we train the model from scratch, we need to create a folder
    #     # to save the model,
    #     saved_path = train_utils.setup_train(hypes)
    #     scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    
    ###################CoPEFT fine tuning####################
    if layers_to_finetune is not None:
        model = fine_tuning_utils.set_requires_grad(model, layers_to_finetune, print_info=True)
    ###################END ###################

    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    if fine_tuning_flag is True:
        eval_freq = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['train_params']['eval_freq']
        save_freq = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['train_params']['save_freq']
        epoches = fine_tuning_hypes['fine_tuning_setting']['args']['fine_tuning_args']['train_params']['epoches']
    else:
        eval_freq = hypes['train_params']['eval_freq']
        save_freq = hypes['train_params']['save_freq']
        epoches = hypes['train_params']['epoches']
    
    # start_time = time.time()
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        pbar = tqdm.tqdm(total=len(train_loader), leave=True)
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            pbar.update(1)
            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar)

            # back-propagation
            final_loss.backward()
            optimizer.step()

            # torch.cuda.empty_cache()
        if epoch % eval_freq == 0:
            valid_ave_loss = []
            pbar_val = tqdm.tqdm(total=len(val_loader), leave=True)

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
                    pbar_val.update(1)
                    criterion.logging(epoch, i, len(val_loader), writer=None, suffix="_eval", pbar=pbar_val)


            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if epoch % save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        scheduler.step(epoch)

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    # ddp training may leave multiple bestval
    bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
    
    if len(bestval_model_list) > 1:
        import numpy as np
        bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
        ascending_idx = np.argsort(bestval_model_epoch_list)
        for idx in ascending_idx:
            if idx != (len(bestval_model_list) - 1):
                os.remove(bestval_model_list[idx])
    
    # end_time = time.time()
    # print(f"Training time: {end_time - start_time}")

if __name__ == '__main__':
    main()
