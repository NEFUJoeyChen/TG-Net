import os
import random
import time
import shutil
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from trainer import Trainer
from dataset import Dataset
from utils.tools import get_config, random_bbox, mask_image
from utils.logger import get_logger

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, default=1234, help='manual seed')


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    # Configure checkpoint path
    checkpoint_path = os.path.join('checkpoints',
                                   config['dataset_name'],
                                   config['mask_type'] + '_' + config['expname'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    writer = SummaryWriter(checkpoint_path)
    logger = get_logger(checkpoint_path)    # get logger and configure it at the first call

    # logger.info("Arguments: {}".format(args))
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # Log the configuration
    # logger.info("Configuration: {}".format(config))

     # for unexpected error logging
    # Load the dataset
    # logger.info("Training on dataset: {}".format(config['dataset_name']))
    train_tf = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])
    train_dataset = Dataset(transform=train_tf)
    # val_dataset = Dataset(data_path=config['val_data_path'],
    #                       with_subfolder=config['data_with_subfolder'],
    #                       image_size=config['image_size'],
    #                       random_crop=config['random_crop'])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True
                                               )
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                           batch_size=config['batch_size'],
    #                                           shuffle=False,
    #                                           num_workers=config['num_workers'])

    # Define the trainer
    trainer = Trainer(config)
    # logger.info("\n{}".format(trainer.netG))
    # logger.info("\n{}".format(trainer.localD))
    # logger.info("\n{}".format(trainer.globalD))

    # if cuda:
    #     trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
    #     trainer_module = trainer.module
    # else:
    trainer_module = trainer

    # Get the resume iteration to restart training
    start_epoch = trainer_module.resume(config['resume']) if config['resume'] else 0

    iterable_train_loader = iter(train_loader)
    # ground_truth, labels = iterable_train_loader.next()
    time_count = time.time()
    total_steps = 0
    for epoch in tqdm(range(start_epoch, 20)):
        epoch_start_time = time.time()

        for i, data in enumerate(tqdm(train_loader)):
            ground_truth, labels = data
            # ground_truth, labels = iterable_train_loader.next()
            # except StopIteration:
            #     iterable_train_loader = iter(train_loader)
            #     ground_truth = next(iterable_train_loader)
            total_steps += config['batch_size']
            # Prepare the inputs
            bboxes = random_bbox(config, batch_size=ground_truth.size(0))
            x, mask = mask_image(ground_truth, bboxes, config)
            if cuda:
                x = x.cuda()
                mask = mask.cuda()
                ground_truth = ground_truth.cuda()

            ###### Forward pass ######
            compute_g_loss = (i + 1) % config['n_critic'] == 0
            losses, inpainted_result, offset_flow = trainer(x, bboxes, mask, ground_truth, compute_g_loss)
            # Scalars from different devices are gathered into vectors
            for k in losses.keys():
                if not losses[k].dim() == 0:
                    losses[k] = torch.mean(losses[k])

            ###### Backward pass ######
            # Update D
            trainer_module.optimizer_d.zero_grad()
            losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
            losses['d'].backward()
            trainer_module.optimizer_d.step()

            # Update G
            if compute_g_loss:
                trainer_module.optimizer_g.zero_grad()
                losses['g'] = losses['l1'] * config['l1_loss_alpha'] \
                              + losses['ae'] * config['ae_loss_alpha'] \
                              + losses['wgan_g'] * config['gan_loss_alpha']
                losses['g'].backward()
                trainer_module.optimizer_g.step()

            # Log and visualization
            log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
            if (i + 1) % config['print_iter'] == 0:
                time_count = time.time() - time_count
                speed = config['print_iter'] / time_count
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()

                message = 'Iter: [%d/%d] ' % (i+1, 5169)
                for k in log_losses:
                    v = losses.get(k, 0.)
                    writer.add_scalar(k, v, i+1)
                    message += '%s: %.6f ' % (k, v)
                message += speed_msg
                logger.info(message)

            if (i + 1) % (config['viz_iter']) == 0:
                viz_max_out = config['viz_max_out']
                if x.size(0) > viz_max_out:
                    viz_images = torch.stack([x[:viz_max_out], inpainted_result[:viz_max_out],
                                              offset_flow[:viz_max_out]], dim=1)
                else:
                    viz_images = torch.stack([x, inpainted_result, offset_flow], dim=1)
                viz_images = viz_images.view(-1, *list(x.size())[1:])
                vutils.save_image(viz_images,
                                  '%s/niter_%03d.png' % (checkpoint_path, i+1),
                                  nrow=3 * 4,
                                  normalize=True)

        # Save the model
        if (epoch + 1) % config['snapshot_save_iter'] == 0:
            trainer_module.save_model(checkpoint_path, epoch+1)
            
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, 20, time.time() - epoch_start_time))


if __name__ == '__main__':
    main()
