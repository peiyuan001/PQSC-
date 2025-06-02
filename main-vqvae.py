import torch
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
import time
from pathlib import Path
import shutil
from trainer import VQVAETrainer
from datasets import get_dataset
from hps import HPS_VQVAE as HPS
from helper import get_device, get_parameter_count
from torch.optim import lr_scheduler



if __name__ == '__main__':

    """
        Parameter Setting:
            args: Global settings
            cfg:  specific settings for datasets, also can be globally controlled
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1, choices=[0,1,2,9]) #9:cpu, 0,1,2:gpu
    parser.add_argument('--lwe_mode', type=str, default='Authorized', choices=['Authorized', 'Eaves'])
    parser.add_argument('--sionna_channel', default=True)
    parser.add_argument('--channel_type', type=str, default='exp', choices=['exp', 'sim'])
    parser.add_argument('--channel_selection', type=str, default='Rayleigh', choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--task', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--no-tqdm', action='store_true')
    parser.add_argument('--no-save', default=False)
    parser.add_argument('--save_mode', type=str, default='task', choices=['task', 'given', 'none'])
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--evaluate', default=False, action='store_true')
    parser.add_argument('--save-jpg', action='store_true')
    parser.add_argument('--set_cfg', default=True)
    parser.add_argument('--save_noise', default=False)
    args = parser.parse_args()

    cfg = HPS[args.task]
    if args.set_cfg:
        print(f"> -------args set cfg-----------")
        cfg.batch_size = 4
        cfg.max_epochs = 50
        cfg.learning_rate = 2e-3
        cfg.noise_db = 20
        cfg.nb_levels = 1
        cfg.scaling_rates = [2]
    device = get_device(args.device)
    print(f"from main-vqvae.py: device:{device}")
    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    cfg_id = f"scale{cfg.scaling_rates}_{cfg.noise_db}dB_Transfer{args.transfer_learning}_channel={args.channel_type}+{args.channel_selection}_lwe={args.lwe_mode}"

    """
        Initialization:
            Evaluating lr at the first iter
            Dynamic learning rate
    """
    print(f"> Initialising VQ-VAE-2 model")
    print(f"> Running on {args.device}")
    trainer = VQVAETrainer(cfg, args)
    optimizer = trainer.opt
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=64, min_lr=5e-4, verbose=True
    )
    print(f"> Number of parameters: {get_parameter_count(trainer.net)}")

    train_loader, test_loader, show_loader = get_dataset(args.task, cfg)
    print("show loader len=", len(show_loader))
    print("> Settings:", cfg)

    # if not args.no_save:
    runs_dir = Path(f"runs")
    root_dir = runs_dir / f"{args.task}_{cfg_id}_{save_id}" #_{save_id}
    chk_dir = root_dir / "checkpoints"
    img_dir = root_dir / "images"
    log_dir = root_dir / "logs"
    runs_dir.mkdir(exist_ok=True)
    root_dir.mkdir(exist_ok=True)
    chk_dir.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir)
    shutil.copy('./hps.py', log_dir) # save hyper parameter settings

    print(f"> Loading {cfg.display_name} dataset")
    if args.evaluate:
        cfg.max_epochs = 2
        print('--evaluate mode--')

    print('f>--evaluating learning rate...')
    pb = tqdm(train_loader, disable=args.no_tqdm)
    for i, (x, _) in enumerate(pb):
        loss, loss_mse, loss_quanti, latent, psnr, ssim = trainer.train(x)
        if psnr <= 15:
            cfg.learning_rate = 1e-2
        elif psnr > 15 and psnr < 18:
            cfg.learning_rate = 3e-3
        else:
            cfg.learning_rate = 1e-3
        print(f"from main-vqvae.py: lr starting with {cfg.learning_rate}")
        break

    """
        Training, Evaluation, and Result Saving
    """
    for eid in range(cfg.max_epochs):
        print(f"> Epoch {eid + 1}/{cfg.max_epochs}:")
        epoch_start_time = time.time()

        "-------------Training phase---------------"
        if not args.evaluate:
            epoch_loss, epoch_mse_loss, epoch_quanti_loss = 0.0, 0.0, 0.0
            pb = tqdm(train_loader, disable=args.no_tqdm)
            psnr_sum = 0
            ssim_sum = 0
            for i, (x, _) in enumerate(pb):

                # running model
                start_time = time.time()
                loss, loss_mse, loss_quanti, latent, psnr, ssim = trainer.train(x)
                end_time = time.time()

                # calculating metrics
                psnr_sum += psnr
                psnr_avg = psnr_sum/(i+1)
                ssim_sum += ssim
                ssim_avg = ssim_sum/(i+1)
                epoch_loss += loss
                epoch_mse_loss += loss_mse
                epoch_quanti_loss += loss_quanti

                # updating leaning rate
                iter = i+eid*len(pb)
                if not args.fine_tune:
                    scheduler.step(psnr_avg, iter)
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Current Learning Rate: {current_lr:5f}')

                # saving results
                """-------TensorBoard saving---------"""
                writer.add_scalar('MSE Loss/train', loss_mse, iter)
                writer.add_scalar('PSNR Loss/train', psnr, iter)
                writer.add_scalar('Quantization Loss/train', loss_quanti, iter)
                writer.add_scalar('lr/train', current_lr, iter)
                writer.add_scalar('ssim/train', ssim, iter)
                """-------Log savlog_fileing---------"""

                with open(os.path.join(log_dir, 'training_log.txt'), 'a') as log_file:
                    log_file.write(f"Epoch {eid + 1}, iter{i + 1}, Global iter{iter}: MSE Loss = {loss_mse:.4f},"
                                   f" PSNR LOSS = {psnr:.4f}, psnr_avg: {psnr_avg:.4f}, Quantization Loss = {loss_quanti:.4f}"
                                   f"SSIM = {ssim:.4f}, SSIM_avg = {ssim_avg:.4f}\n")
                pb.set_description(f"training_loss: {epoch_loss / (i+1):.4f} [mse_loss: {epoch_mse_loss/ (i+1):.4f}, quanti_loss: {epoch_quanti_loss / (i+1):.4f}],"
                                   f"PSNR LOSS = {psnr:.4f}, psnr_avg: {psnr_avg:.4f}"
                                   f"SSIM = {ssim:.4f}, SSIM_avg = {ssim_avg:.4f}")
            print(f"> Training loss: {epoch_loss / len(train_loader)} [mse_loss: {epoch_mse_loss / len(train_loader)}, quanti_loss: {epoch_quanti_loss / len(train_loader)}]"
                  f"psnr: {psnr:.4f}, psnr_avg: {psnr_avg:.4f}\n")

        "-------------Evaluation phase---------------"
        epoch_loss, epoch_r_loss, epoch_l_loss = 0.0, 0.0, 0.0
        pb = tqdm(test_loader, disable=args.no_tqdm)
        psnr_sum = 0
        ssim_sum = 0
        for i, (x, xxxxxxx) in enumerate(pb):
            loss, loss_mse, loss_quanti, _, latent, psnr, ssim = trainer.eval(x)
            psnr_sum += psnr
            psnr_avg = psnr_sum/(i+1)
            ssim_sum += ssim
            ssim_avg = ssim_sum / (i + 1)
            epoch_loss += loss
            epoch_mse_loss += loss_mse
            epoch_quanti_loss += loss_quanti
            iter = i + eid * len(pb)
            writer.add_scalar('MSE Loss/test', loss_mse, iter)
            writer.add_scalar('PSNR Loss/test', psnr, iter)
            writer.add_scalar('Quantization Loss/test', loss_quanti, iter)
            writer.add_scalar('ssim/test', ssim, iter)
            with open(os.path.join(log_dir, 'testing_log.txt'), 'a') as log_file:
                log_file.write(f"Epoch {eid + 1}, iter{i + 1}, Global iter{iter}: MSE Loss = {loss_mse:.4f},"
                               f" PSNR LOSS = {psnr:.4f}, psnr_avg: {psnr_avg:.4f}, Quantization Loss = {loss_quanti:.4f}"
                               f"SSIM = {ssim:.4f}, SSIM_avg = {ssim_avg:.4f}\n")
            pb.set_description(f"evaluation: {epoch_loss / (i+1):.4f} [mse_loss: {epoch_mse_loss/ (i+1):.4f}, quanti_loss: {epoch_quanti_loss / (i+1):.4f}],"
                               f"PSNR LOSS = {psnr:.4f}, psnr_avg: {psnr_avg:.4f}"
                               f"SSIM = {ssim:.4f}, SSIM_avg = {ssim_avg:.4f}")
            print(f"> Evaluation loss: {epoch_loss / len(test_loader)} [mse_loss: {epoch_mse_loss / len(test_loader)},"
                  f" quanti_loss: {epoch_quanti_loss / len(test_loader)}], psnr: {psnr:.4f}, psnr_avg: {psnr_avg:.4f}\n")

        "-----------------------------save img----------------------------------"
        if eid % 2 == 0 and not args.no_save and args.save_mode == 'task':
            show_num = 2
            pb = tqdm(show_loader, disable=args.no_tqdm)
            print(f"show loader length={len(pb)}")
            for j, (x, xxxxxxx) in enumerate(pb):
                psnr_sum = 0
                loss, loss_mse, loss_quanti, y, latent, psnr, ssim = trainer.eval(x)
                for j in range(min(show_num, len(x))):
                    if eid == 2:
                        save_image(x[j],
                                   img_dir / f"origin-ep{str(eid)}-no{str(j)}.{'jpg' if args.save_jpg else 'png'}",
                                   normalize=True, value_range=(-1, 1))
                for j in range(min(show_num, len(x))):
                    save_image(y[j],
                               img_dir / f"recon-ep{str(eid)}-no{str(j)}-psnr{psnr:.2f}-ssim{ssim:.2f}.{'jpg' if args.save_jpg else 'png'}",
                               normalize=True, value_range=(-1, 1))
                    print(f"img saved at {img_dir}")
        if eid % cfg.checkpoint_frequency == 0 and not args.no_save:
            trainer.save_checkpoint(chk_dir / f"{args.task}-state-dict-{str(eid).zfill(4)}.pt")
        print(f"> Epoch time taken: {time.time() - epoch_start_time:.2f} seconds.\n")
        writer.add_scalar('time/time', time.time() - epoch_start_time, eid)

