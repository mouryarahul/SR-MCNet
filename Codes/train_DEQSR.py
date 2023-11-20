import os
import argparse
import copy
import csv
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from torch.utils.data.dataloader import DataLoader
from datasets import TrainDataset, EvalDataset

from RDN.rdn import RDN
import RCAN
from SRCNN.srcnn import SRCNN

from realSN_models import DnCNN, FixedPointFuncCase2, DEQNet
from solvers import anderson_3

from interpolation import torch_resize

from utils import AverageMeter, rgb_to_y
from tqdm import tqdm

from skimage.metrics import structural_similarity as calc_ssim
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-num', type=int, default=2)
    parser.add_argument('--config-num', type=int, default=1)
    
    parser.add_argument('--interp-scale', type=int, default=2)
    parser.add_argument('--interp-method', type=str, default='TORCH',
                        help='choose among PIL, TORCH')

    parser.add_argument('--fixed-dnn', type=str, default='RCAN',
                        help='choose among SRCNN, RDN, RCAN')
    
    args = parser.parse_known_args()[0]

    if args.fixed_dnn == 'RCAN':
        # RCAN Specification
        parser.add_argument('--model', default='RCAN',help='model name')
        parser.add_argument('--n_resblocks', type=int, default=20,
                            help='number of residual blocks')
        parser.add_argument('--n_feats', type=int, default=64,
                            help='number of feature maps')
        parser.add_argument('--res_scale', type=float, default=1,
                            help='residual scaling')
        parser.add_argument('--shift_mean', default=True,
                            help='subtract pixel mean from the input')                  
        # options for residual group and feature channel reduction
        parser.add_argument('--n_resgroups', type=int, default=10,
                            help='number of residual groups')
        parser.add_argument('--reduction', type=int, default=16,
                            help='number of feature maps reduction')
        parser.add_argument('--rgb_range', type=int, default=255,
                            help='maximum value of RGB') 
        parser.add_argument('--n_colors', type=int, default=3,
                            help='number of color channels to use')
        parser.add_argument('--print_model', action='store_true',
                            help='print model')                        
        parser.add_argument('--save_models', action='store_true',
                            help='save all intermediate models')
        parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')                        
        parser.add_argument('--self_ensemble', action='store_true',
                            help='use self-ensemble method for test')
        parser.add_argument('--pre_train', type=str, default='../Learned_RCAN_Models/',
                            help='pre-trained model directory')
    elif args.fixed_dnn == 'RDN':
        # Specification for RDN Network
        parser.add_argument('--rdn-num-features', type=int, default=64)
        parser.add_argument('--rdn-growth-rate', type=int, default=64)
        parser.add_argument('--rdn-num-blocks', type=int, default=16)
        parser.add_argument('--rdn-num-layers', type=int, default=8)
        parser.add_argument('--rdn-weight-file', type=str, default='../Learned_RDN_Models/')
    else:
        # Weight files for Fixed SRCNN
        parser.add_argument('--srcnn-weight-file', type=str, default='../Learned_SRCNN_Models/T91/Interp_Order_3/')

    # Parameters for Fixed-Point Solver and Deep Equilibrium Layer
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--rho', type=float, default=0.1)
    parser.add_argument('--tol', type=float, default=1E-5)
    parser.add_argument('--maxiter-forward', type=int, default=200)
    parser.add_argument('--miniter-forward', type=int, default=5)
    parser.add_argument('--maxiter-backward', type=int, default=100)
    parser.add_argument('--miniter-backward', type=int, default=5)
    parser.add_argument('--beta', type=float, default=2.5)
    parser.add_argument('--hierarchical', action='store_true')

    # Jacobian regularization related [Bai et al. 2021]
    parser.add_argument('--jac_regul', action='store_true')
    parser.add_argument('--jac_loss_weight', type=float, default=1.0,
                        help='jacobian regularization loss weight (default to 0)')
    parser.add_argument('--jac_loss_freq', type=float, default=0.35,
                        help='the frequency of applying the jacobian regularization (default to 0)')
    parser.add_argument('--jac_incremental', type=int, default=3500,
                        help='if positive, increase jac_loss_weight by 0.1 after this many steps')
    parser.add_argument('--spectral_radius_mode', action='store_true',
                        help='compute spectral radius at validation time')
    
    # Specification for DNCNNs
    parser.add_argument('--dncnn-num-layers', type=int, default=6)
    parser.add_argument('--dncnn-lip', type=float, default=0.9)
    parser.add_argument('--dncnn-bias', action='store_true')
    parser.add_argument('--dncnn-bn', action='store_true')
    parser.add_argument('--dncnn-act', type=str, default='CELU', choices=['RELU', 'CELU'])
    parser.add_argument('--dncnn-skip', type=bool, default=False)
    parser.add_argument('--dncnn-weight-file', type=str, default='../Learned_DNCNN_Models/T91/RealSN_DnCNN')
    parser.add_argument('--dncnn-lr', type=float, default=0.001)
    parser.add_argument('--sigmaL', type=float, default=0.01250)
    parser.add_argument('--sigmaH', type=float, default=0.02500)

    # Optimizer Specification
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1E-3)  # Initial Learning Rate
    parser.add_argument('--alpha-lr', type=float, default=0.05)
    parser.add_argument('--decrease-lr', type=bool, default=True)
    parser.add_argument('--lr-step', type=int, default=30)
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--clip-grad', action='store_true')
    parser.add_argument('--clip-norm', type=float, default=0.01, 
                        help='gradient clipping')

    # Parameters for DataLoader
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=80)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--datasets-dir', type=str, default='../Data/')
    parser.add_argument('--train-data', type=str, default='T91')
    parser.add_argument('--eval-data', type=str, default='Set5')

    parser.add_argument('--last-checkpoint', type=str, default='')
    parser.add_argument('--best-weight-file', type=str, default='')
    parser.add_argument('--output-dir', type=str, default='../Learned_DEQSR_Models')

    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--use-gpu', action='store_true') #type=bool, default=True) #
    parser.add_argument('--verbose', action='store_true') #type=bool, default=True) #
    parser.add_argument('--verbose2', action='store_true') #type=bool, default=True) #

    args = parser.parse_args()

    # set seed for random generator
    torch.manual_seed(args.seed)

    # Define the Training and Evaluation data file paths and Set the Directory path to save Learned Models
    train_file = os.path.join(args.datasets_dir, 'datasets_{}'.format(args.interp_method), args.train_data)
    eval_file = os.path.join(args.datasets_dir, 'datasets_{}'.format(args.interp_method), args.eval_data)
    train_file = train_file + "_bicubic_x{}".format(args.interp_scale) + ".h5"
    eval_file = eval_file + "_bicubic_x{}".format(args.interp_scale) + ".h5"

    # Define the Output Path
    output_dir = os.path.join(args.output_dir, args.train_data,
                                '{}'.format(args.fixed_dnn),
                                'x{}'.format(args.interp_scale),
                                'Case_{}'.format(args.case_num), 'Config_{}'.format(args.config_num))
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make Dataset Iterable
    train_dataset = TrainDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # Select Device
    if args.use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cuda.matmul.allow_tf32 = False
            if torch.backends.cudnn.is_available():
                import torch.backends.cudnn as cudnn
                cudnn.enabled = True
                cudnn.allow_tf32 = True
                cudnn.benchmark = True
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print("Device=", device)

    # Instantiate the Fixed Networks in evaluation mode
    if args.fixed_dnn == 'SRCNN':
        fixed_dnn_weight_file = os.path.join(args.srcnn_weight_file, 'x{}'.format(args.interp_scale), 'PIL', 'best.pth')
        fixed_dnn = SRCNN()
        fixed_dnn.load_state_dict(torch.load(fixed_dnn_weight_file, map_location='cpu'))
    elif args.fixed_dnn == 'RCAN':
        args.pre_train = os.path.join(args.pre_train, 'RCAN_BIX{}.pt'.format(args.interp_scale))
        fixed_dnn = RCAN.Model(args, device)
    elif args.fixed_dnn == 'RDN':
        fixed_dnn_weight_file = os.path.join(args.rdn_weight_file, 'PIL', 'x{}'.format(args.interp_scale), 'best.pth')
        fixed_dnn = RDN(scale_factor=args.interp_scale,
                        num_channels=3,
                        num_features=args.rdn_num_features,
                        growth_rate=args.rdn_growth_rate,
                        num_blocks=args.rdn_num_blocks,
                        num_layers=args.rdn_num_layers)
        fixed_dnn.load_state_dict(torch.load(fixed_dnn_weight_file, map_location='cpu'))
    else:
        raise Exception("Unknown Fixed DNN!")
    fixed_dnn = fixed_dnn.to(device)
    fixed_dnn.eval()

    
    # Define the DEQ Model
    dncnn = DnCNN(channels=1, num_of_layers=args.dncnn_num_layers, lip=args.dncnn_lip, bias=args.dncnn_bias, bn=args.dncnn_bn, act=args.dncnn_act, skip=args.dncnn_skip).to(device)
    if args.dncnn_weight_file:
        dncnn_weight_file = os.path.join(args.dncnn_weight_file, 'Layers_{}'.format(args.dncnn_num_layers), 'BN_{}'.format(args.dncnn_bn), '{}'.format(args.dncnn_act), 'sigma_{:0.5f}_{:0.5f}'.format(args.sigmaL, args.sigmaH), 'best.pth')
        dncnn.load_state_dict(torch.load(dncnn_weight_file, map_location='cpu'))

    if args.case_num == 2:
        print("Choosing Case 2")
        f = FixedPointFuncCase2(dncnn, args.interp_scale, (args.sigmaL+args.sigmaH)/2, args.epsilon, alpha=args.alpha, rho=args.rho)
    else:
        raise ValueError("Unknown case!")

    # DEQ Model
    model = DEQNet(f, anderson_3, args.case_num, tol=args.tol,
                    min_iter_forward=args.miniter_forward,
                    max_iter_forward=args.maxiter_forward,
                    min_iter_backward=args.miniter_backward,
                    max_iter_backward=args.maxiter_backward, 
                    m=5, beta=args.beta, verbose=args.verbose2)
    model = model.to(device) # Bring model parameters to Device

    # Define the Loss Function
    loss_func = torch.nn.MSELoss().to(device)

    # Select the Optimizer and parameters to be learned
    if args.optimizer == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer = optim.Adam(
                    [
                        {'params': model.f.alpha, 'lr': args.alpha_lr},
                        {'params': model.f.dncnn.parameters(), 'lr': args.dncnn_lr}
                    ], 
                    lr=args.lr)
    elif args.optimizer == 'sgd':
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        optimizer = optim.SGD(
                    [
                        {'params': model.f.alpha, 'lr': args.alpha_lr},
                        {'params': model.f.dncnn.parameters(), 'lr': args.dncnn_lr}
                    ],
                    lr=args.lr,
                    momentum=0.9)
    else:
        raise ValueError("Unsupported Optimizer! select either 'adam' or 'sgd'.")

    if args.decrease_lr:
        scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        # scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)

    # Load last checkpoint from earlier training session
    train_step = 0
    new_epoch = 0
    best_epoch = 0
    best_psnr = 0.0
    log_write_mode = 'w'
    if args.last_checkpoint:
        print("Loading Last Checkpoint File...")
        log_write_mode = 'a'
        checkpoint = torch.load(args.last_checkpoint, map_location=lambda storage, loc: storage.cuda())
        new_epoch = checkpoint['epoch'] + 1
        best_epoch = checkpoint['best_epoch']
        best_psnr = checkpoint['best_psnr']
        model.load_state_dict(checkpoint['last_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    elif args.best_weight_file:
        print("Loading Best Model File...")
        log_write_mode = 'w'
        model.load_state_dict(torch.load(args.best_weight_file, map_location=lambda storage, loc: storage.cuda()))
    else:
        pass


    # Create a Log file to record results
    log_file_name = os.path.join(output_dir, 'training_log.txt')
    with open(log_file_name, mode=log_write_mode) as log_file:
        print("Parameters:\n", args, file=log_file)
        log_writer = csv.writer(log_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        # Log items 
        log_writer.writerow(['{:8s}'.format("Epoch No"), '{:14s}'.format("LR"),
                            '{:14s}'.format("Loss"), '{:14s}'.format("Val_Loss"),
                            '{:14s}'.format("DF"),
                            '{:10s}'.format("PSNR"), '{:10s}'.format("SSIM"),
                            '{:10s}'.format("Beta")
                            ])
        log_file.flush()
        # ===================================== Training Phase =======================================#
        for epoch in range(new_epoch, args.num_epochs):
            model.train()
            epoch_losses = AverageMeter()
 
            with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

                # Learning Phase
                for data in train_dataloader:
                    lr_imgs, true_hr_imgs, lr_up_imgs = data
                    lr_up_imgs = lr_up_imgs.to(device)
                    true_hr_imgs = true_hr_imgs.to(device)
                    lr_imgs = lr_imgs.to(device)

                    # For DEQ Jacobian Regularization:
                    compute_jac_loss = args.jac_regul and (np.random.uniform(0,1) < args.jac_loss_freq)
                    # compute_jac_loss = True

                    # Zero out the previous gradients
                    model.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Convert images into Y channel
                    true_hr_y = rgb_to_y(true_hr_imgs)
                    true_hr_y = true_hr_y/255.0
                    if args.fixed_dnn == 'SRCNN':
                        lr_y = rgb_to_y(lr_imgs)
                        lr_y = lr_y / 255.0
                        lr_up_y = rgb_to_y(lr_up_imgs)
                        lr_up_y = lr_up_y/255.0
                    elif args.fixed_dnn == 'RDN':
                        lr_y = rgb_to_y(lr_imgs)
                        lr_y = lr_y / 255.0
                        lr_imgs = lr_imgs/255.0
                    elif args.fixed_dnn == 'RCAN':
                        lr_y = rgb_to_y(lr_imgs)
                        lr_y = lr_y / 225.0
                    else:
                        raise Exception("Unknown Fixed DNN!")

                    # Predict output from the Fixed DNN
                    if device.type == 'cuda':
                        # with torch.cuda.amp.autocast():
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            with torch.no_grad():
                                if args.fixed_dnn == 'SRCNN':
                                    sr = fixed_dnn(lr_up_y)
                                elif args.fixed_dnn == 'RDN':
                                    sr = fixed_dnn(lr_imgs)
                                elif args.fixed_dnn == 'RCAN':
                                    sr = fixed_dnn(lr_imgs)
                                else:
                                    raise Exception("Unknown Fixed DNN!")
                    elif device.type == 'cpu':
                        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                            with torch.no_grad():
                                if args.fixed_dnn == 'SRCNN':
                                    sr = fixed_dnn(lr_up_y)
                                elif args.fixed_dnn == 'RDN':
                                    sr = fixed_dnn(lr_imgs)
                                elif args.fixed_dnn == 'RCAN':
                                    sr = fixed_dnn(lr_imgs)
                                else:
                                    raise Exception("Unknown Fixed DNN!")
                    else:
                        with torch.no_grad():
                            if args.fixed_dnn == 'SRCNN':
                                sr = fixed_dnn(lr_up_y)
                            elif args.fixed_dnn == 'RDN':
                                sr = fixed_dnn(lr_imgs)
                            elif args.fixed_dnn == 'RCAN':
                                sr = fixed_dnn(lr_imgs)
                            else:
                                raise Exception("Unknown Fixed DNN!")
                    
                    # Convert images into Y channel
                    sr = sr.float()
                    if args.fixed_dnn == 'SRCNN':
                        sr_y = sr
                    elif args.fixed_dnn == 'RDN':
                        sr_y = rgb_to_y(255.0*sr) / 255.0
                    elif args.fixed_dnn == 'RCAN':
                        sr_y = rgb_to_y(sr) / 255.0
                    else:
                        raise Exception("Unknown Fixed DNN!")

                    # Predict output from the Model
                    pred_hr_y, jac_loss, _ = model(sr_y, lr_y, compute_jac_loss)
                    
                    # for index in range(0, 2):
                    #     plt.figure()
                    #     plt.imshow(true_hr_y[index].detach().cpu().numpy().squeeze())
                    #     plt.title('True HR image')
                    #     plt.colorbar()
                    #     plt.figure()
                    #     plt.imshow(lr_y[index].detach().cpu().numpy().squeeze())
                    #     plt.title('LR input image')
                    #     plt.colorbar()
                    #     plt.figure()
                    #     plt.imshow(w[index].detach().cpu().numpy().squeeze())
                    #     plt.title('CNN HR image')
                    #     plt.colorbar()
                    #     plt.figure()
                    #     plt.imshow(pred_hr_y[index].detach().cpu().numpy().squeeze())
                    #     plt.title('Final HR image')
                    #     plt.colorbar()
                    #     plt.show()

                    # Calculate the Current Loss
                    loss = loss_func(pred_hr_y, true_hr_y)
                    jac_loss = jac_loss.type_as(loss)

                    # Update the metric recorder
                    epoch_losses.update(loss.detach().item(), len(pred_hr_y))

                    # Calculate Gradients
                    if compute_jac_loss:
                        (loss + jac_loss * args.jac_loss_weight).backward()
                        if args.verbose2:
                            print("Jacobian Loss = ", jac_loss.item())
                    else:
                        loss.backward()

                    # Perform Gradient Clipping
                    if args.clip_grad:
                        total_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, error_if_nonfinite=True)
                        if args.verbose2:
                            print("Gradient Norm={:0.8e}".format(total_grad.item()))

                    # Update the Learnable Parameters
                    optimizer.step()

                    train_step += 1

                    # For display Purpose
                    t.set_postfix(loss='{:.12f}'.format(epoch_losses.avg))
                    t.update(len(pred_hr_y))

                    if compute_jac_loss and args.jac_loss_weight > 0 and args.jac_loss_freq > 0 and \
                        args.jac_incremental > 0 and train_step % args.jac_incremental == 0:
                            if args.verbose2:
                                print(f"Adding 0.1 to jac. regularization weight after {train_step} steps")
                            args.jac_loss_weight += 0.1

            
            # Print Learning Rate and Avg Epoch Loss
            last_lr = scheduler.get_last_lr()

            if args.verbose:
                print('Epoch:', epoch)
                print('LR=                                            ', last_lr)
                print('Loss=                                          {:0.8e}'.format(epoch_losses.avg))

            # ============================================ Evaluation Phase =======================================#
            model.eval()
            
            epoch_val_losses = AverageMeter()
            epoch_psnr = AverageMeter()
            epoch_ssim = AverageMeter()
            epoch_data_error = AverageMeter()
            epoch_cnn_psnr = AverageMeter()
            epoch_cnn_ssim = AverageMeter()

            for data in eval_dataloader:
                lr_imgs, true_hr_imgs, lr_up_imgs = data
                    
                with torch.no_grad():
                    lr_up_imgs = lr_up_imgs.to(device)
                    true_hr_imgs = true_hr_imgs.to(device)
                    lr_imgs = lr_imgs.to(device)

                    # ===================== Evaluation on Full Image =========================#
                    # Convert images into Y channel
                    true_hr_y = rgb_to_y(true_hr_imgs)
                    true_hr_y = true_hr_y/255.0
                    if args.fixed_dnn == 'SRCNN':
                        lr_y = rgb_to_y(lr_imgs)
                        lr_y = lr_y / 255.0
                        lr_up_y = rgb_to_y(lr_up_imgs)
                        lr_up_y = lr_up_y/255.0
                    elif args.fixed_dnn == 'RDN':
                        lr_y = rgb_to_y(lr_imgs)
                        lr_y = lr_y / 255.0
                        lr_imgs = lr_imgs/255.0
                    elif args.fixed_dnn == 'RCAN':
                        lr_y = rgb_to_y(lr_imgs)
                        lr_y = lr_y / 255.0
                    else:
                        raise Exception("Unknown Fixed DNN!")


                    # Predict output from the Fixed DNN
                    if device.type == 'cuda':
                        with torch.autocast(device_type=device.type, dtype=torch.float16):
                            if args.fixed_dnn == 'SRCNN':
                                sr = fixed_dnn(lr_up_y)
                            elif args.fixed_dnn == 'RDN':
                                sr = fixed_dnn(lr_imgs)
                            elif args.fixed_dnn == 'RCAN':
                                sr = fixed_dnn(lr_imgs)
                            else:
                                raise Exception("Unknown Fixed DNN!")
                    elif device.type == 'cpu':
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            if args.fixed_dnn == 'SRCNN':
                                sr = fixed_dnn(lr_up_y)
                            elif args.fixed_dnn == 'RDN':
                                sr = fixed_dnn(lr_imgs)
                            elif args.fixed_dnn == 'RCAN':
                                sr = fixed_dnn(lr_imgs)
                            else:
                                raise Exception("Unknown Fixed DNN!")
                    else:
                        if args.fixed_dnn == 'SRCNN':
                            sr = fixed_dnn(lr_up_y)
                        elif args.fixed_dnn == 'RDN':
                            sr = fixed_dnn(lr_imgs)
                        elif args.fixed_dnn == 'RCAN':
                            sr = fixed_dnn(lr_imgs)
                        else:
                            raise Exception("Unknown Fixed DNN!")
                    
                    # Convert images into Y channel
                    sr = sr.to(torch.float32)
                    if args.fixed_dnn == 'SRCNN':
                        sr_y = sr
                    elif args.fixed_dnn == 'RDN':
                        sr_y = rgb_to_y(255.0*sr) / 255.0
                    elif args.fixed_dnn == 'RCAN':
                        sr_y = rgb_to_y(sr) / 255.0
                    else:
                        raise Exception("Unknown Fixed DNN!")

                    # Predict output from the Model
                    pred_hr_y, jac_loss, sradius = model(sr_y, lr_y, compute_jac_loss=False, spectral_radius_mode=args.spectral_radius_mode)
                    if args.spectral_radius_mode:
                        if args.verbose:
                            print("Spectral radius over validation set: ", sradius.item())

                    # Calculate the Validation Loss
                    val_loss = loss_func(pred_hr_y, true_hr_y)
                    epoch_val_losses.update(val_loss.detach().item(), len(true_hr_y))

                    # Convert ndarrays to Numpy
                    true_hr_y   = true_hr_y.detach().squeeze().cpu().numpy()
                    pred_hr_y   = pred_hr_y.detach().squeeze().cpu().numpy()
                    sr_y        = sr_y.detach().squeeze().cpu().numpy()
                    lr_y        = lr_y.detach().squeeze().cpu().numpy()

                    # plt.close('all')
                    # plt.figure(); plt.imshow(true_hr_y, cmap='gray'); plt.title("True HR Image"); plt.colorbar();
                    # plt.figure(); plt.imshow(lr_y, cmap='gray'); plt.title("True LR Image"); plt.colorbar();
                    # plt.figure(); plt.imshow(sr_y, cmap='gray'); plt.title("CNN HR Image"); plt.colorbar();
                    # plt.figure(); plt.imshow(pred_hr_y, cmap='gray'); plt.title("Final HR Image"); plt.colorbar();
                    # plt.show()

                    # epoch_data_error.update(torch.linalg.norm((F.interpolate(pred_hr_y[i].unsqueeze(dim=0), size=None, scale_factor=(1/args.interp_scale), align_corners=False, antialias=True, mode='bicubic') 
                    #                                             - lr_y[i].unsqueeze(dim=0)).flatten()).cpu().numpy())
                    epoch_data_error.update(np.linalg.norm((torch_resize(pred_hr_y, scale=(1/args.interp_scale), order=3) - lr_y).flatten()))
                    true_hr_y = true_hr_y[args.interp_scale:-args.interp_scale, args.interp_scale:-args.interp_scale]
                    pred_hr_y = pred_hr_y[args.interp_scale:-args.interp_scale, args.interp_scale:-args.interp_scale]
                    sr_y = sr_y[args.interp_scale:-args.interp_scale, args.interp_scale:-args.interp_scale]
                    epoch_psnr.update(calc_psnr(true_hr_y, pred_hr_y, data_range=1.0))
                    epoch_ssim.update(calc_ssim(true_hr_y, pred_hr_y, data_range=1.0))
                    
                    epoch_cnn_psnr.update(calc_psnr(true_hr_y, sr_y, data_range=1.0))
                    epoch_cnn_ssim.update(calc_ssim(true_hr_y, sr_y, data_range=1.0))
                       
            # Log information per epoch
            if args.case_num == 3 or args.case_num == 5 or args.case_num == 6:
                log_writer.writerow(['{:8d}'.format(epoch), '{:1.8e}'.format(last_lr[0]),
                                    '{:4.8e}'.format(epoch_losses.avg), '{:4.8e}'.format(epoch_val_losses.avg),
                                    '{:4.8e}'.format(epoch_data_error.avg),
                                    '{:4.8f}'.format(epoch_psnr.avg), '{:4.8f}'.format(epoch_ssim.avg),
                                    'NA'
                                    ])
            else:
                log_writer.writerow(['{:8d}'.format(epoch), '{:1.8e}'.format(last_lr[0]),
                                    '{:4.8e}'.format(epoch_losses.avg), '{:4.8e}'.format(epoch_val_losses.avg),
                                    '{:4.8e}'.format(epoch_data_error.avg),
                                    '{:4.8f}'.format(epoch_psnr.avg), '{:4.8f}'.format(epoch_ssim.avg),
                                    '{:4.8f}'.format(model.f.alpha.item())
                                    ])

            log_file.flush()

            if args.verbose:
                # Print info on screen
                print("Pred PSNR on Full Image:                     {:.4f} & SSIM:{:.4f}".format(epoch_psnr.avg, epoch_ssim.avg))
                print("CNN PSNR on Full Image:                      {:.4f} & SSIM:{:.4f}".format(epoch_cnn_psnr.avg, epoch_cnn_ssim.avg))
                print("Data-Fidelity Value:                         {:4.8e}".format(epoch_data_error.avg))
                print("\n")

            # Keep the last epoch weight
            last_epoch_model_weight = copy.deepcopy(model.state_dict())

            # Save model after every epoch
            # torch.save(last_epoch_model_weight, os.path.join(output_dir, '{}_epoch.pth'.format(epoch)))
            
            # Select the Best Epoch
            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_epoch_model_weight = last_epoch_model_weight
                # Save the Best Model Weights
                log_writer.writerow(["Best Epoch:{} with PSNR: {:0.4f}".format(best_epoch, best_psnr)])
                torch.save(best_epoch_model_weight, os.path.join(output_dir, 'best.pth'))
                log_file.flush()

            # Save the last Checkpoint: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
            torch.save({
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_psnr': best_psnr,
                'last_model_state_dict': last_epoch_model_weight,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(output_dir, 'last_checkpoint.pth'))

            if args.decrease_lr:
                scheduler.step()  # Change the Learning Rate
        
        log_writer.writerow(["Best Epoch:{} with PSNR: {:0.4f}".format(best_epoch, best_psnr)])
        log_file.close()
