import os
import argparse

import numpy as np
import torch

from skimage.io import imread, imsave
from PIL import Image

from RDN.rdn import RDN
import RCAN
from SRCNN.srcnn import SRCNN

from realSN_models import DnCNN, FixedPointFuncCase2, DEQNet
from solvers import anderson_3

from TVTV_Solver import Batch_TVTV_Solver_NUMPY, Batch_Smooth_TVTV_Solver_NUMPY

import torch.nn.functional as F

from interpolation import pil_resize, cv2_resize, torch_resize

from utils import AverageMeter, convert_rgb_to_ycbcr, convert_ycbcr_to_rgb

from skimage.metrics import structural_similarity as calc_ssim
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--case-num', type=int, default=2)
    parser.add_argument('--config-num', type=int, default=179)

    parser.add_argument('--input-dir', type=str, default='../Data/TestImages/Set14')

    parser.add_argument('--interp-scale', type=int, default=2)
    parser.add_argument('--interp-method', type=str, default='TORCH',
                        help='choose among PIL, TORCH')

    parser.add_argument('--post-process', action='store_true')

    parser.add_argument('--fixed-dnn', type=str, default='RCAN',
                        help="Choose between SRCNN or RDN or RCAN")
    
    args = parser.parse_known_args()[0]

    if args.fixed_dnn == 'RCAN':
        # RCAN Specification
        parser.add_argument('--model', default='RCAN',
                            help='model name')
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
        # RDN Specification
        parser.add_argument('--num-features', type=int, default=64)
        parser.add_argument('--growth-rate', type=int, default=64)
        parser.add_argument('--num-blocks', type=int, default=16)
        parser.add_argument('--num-layers', type=int, default=8)
        parser.add_argument('--rdn-weight-file', type=str, default='../Learned_RDN_Models/')
    else:
        parser.add_argument('--srcnn-weight-file', type=str, default='../Learned_SRCNN_Models/')

    if not args.post_process:
        # Specification for DNCNNs
        parser.add_argument('--dncnn-num-layers', type=int, default=6)
        parser.add_argument('--dncnn-bias', action='store_true')
        parser.add_argument('--dncnn-bn', action='store_true')
        parser.add_argument('--dncnn-skip', type=bool, default=False)
        parser.add_argument('--dncnn-weight-file', type=str, default='')
        parser.add_argument('--sigmaL', type=float, default=0.06250)
        parser.add_argument('--sigmaH', type=float, default=0.12500)

        # Parameters for Fixed-Point Solver and Deep Equilibrium Layer
        parser.add_argument('--alpha', type=float, default=5.0)
        parser.add_argument('--epsilon', type=float, default=1.0)
        parser.add_argument('--rho', type=float, default=1.0)
        parser.add_argument('--tol', type=float, default=1E-6)
        parser.add_argument('--maxiter-forward', type=int, default=200)
        parser.add_argument('--miniter-forward', type=int, default=5)
        parser.add_argument('--maxiter-backward', type=int, default=80)
        parser.add_argument('--miniter-backward', type=int, default=5)
        parser.add_argument('--beta', type=float, default=0.9)

        parser.add_argument('--learned-model-dir', type=str, default='../Learned_DEQSR_Models')
        parser.add_argument('--best-model', type=bool, default=True)

    
    parser.add_argument('--train-data', type=str, default='T91')
    parser.add_argument('--output-dir', type=str, default='../Results')

    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--use-gpu', type=bool, default=True)

    parser.add_argument('--save-image', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--verbose2', type=bool, default=True)
    
    args = parser.parse_args()

    print("Running Test Script...")

    # set seed for random generator
    torch.manual_seed(args.seed)

    if args.post_process:
        # Define output directory name
        output_dir = os.path.join(args.output_dir, os.path.split(args.input_dir)[1],
                                    "Interp_Order_{}".format(3),
                                    # "{}".format(args.train_data),
                                    "x{}".format(args.interp_scale),
                                    "{}".format(args.interp_method),
                                    '{}'.format(args.fixed_dnn))
        output_dir = os.path.join(output_dir, 'Post_Processing')
    else:
        # Define output directory name
        output_dir = os.path.join(args.output_dir, os.path.split(args.input_dir)[1],
                                    "Interp_Order_{}".format(3),
                                    # "{}".format(args.train_data),
                                    "x{}".format(args.interp_scale),
                                    "{}".format(args.interp_method),
                                    '{}'.format(args.fixed_dnn),
                                    'DEQNET',
                                    "Case_{}".format(args.case_num),
                                    'Config_{}'.format(args.config_num)
                                )
        
        # Define Learned model weight file
        learned_weight_file = os.path.join(args.learned_model_dir, 
                                        "{}".format(args.train_data), 
                                        '{}'.format(args.fixed_dnn),
                                        "x{}".format(args.interp_scale),
                                        "Case_{}".format(args.case_num),
                                        "Config_{}".format(args.config_num))
        if args.best_model:
            learned_weight_file = os.path.join(learned_weight_file, 'best.pth')
        else:
            learned_weight_file = os.path.join(learned_weight_file, 'last_checkpoint.pth')

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        fixed_dnn_weight_file = os.path.join(args.srcnn_weight_file, '{}'.format(args.train_data), 'Interp_Order_3', 'x{}'.format(args.interp_scale), 'PIL', 'best.pth')
        fixed_dnn = SRCNN().to(device)
        fixed_dnn.load_state_dict(torch.load(fixed_dnn_weight_file, map_location=lambda storage, loc: storage.cuda()))
    elif args.fixed_dnn == 'RCAN':
        args.pre_train = os.path.join(args.pre_train, 'RCAN_BIX{}.pt'.format(args.interp_scale))
        fixed_dnn = RCAN.Model(args, device)
    elif args.fixed_dnn == 'RDN':
        fixed_dnn_weight_file = os.path.join(args.rdn_weight_file, 'PIL', 'x{}'.format(args.interp_scale), 'best.pth')
        fixed_dnn = RDN(scale_factor=args.interp_scale,
                        num_channels=3,
                        num_features=args.num_features,
                        growth_rate=args.growth_rate,
                        num_blocks=args.num_blocks,
                        num_layers=args.num_layers).to(device)
        fixed_dnn.load_state_dict(torch.load(fixed_dnn_weight_file, map_location=lambda storage, loc: storage))
    else:
        raise Exception("Unknown Fixed DNN!")
    fixed_dnn.eval()

    if args.post_process:
        log_file_name = os.path.join(output_dir, 'testing_log.txt')
    else:   # Define the DEQ Models
        dncnn = DnCNN(channels=1, num_of_layers=args.dncnn_num_layers, bias=args.dncnn_bias, bn=args.dncnn_bn, skip=args.dncnn_skip).to(device)
        if args.dncnn_weight_file:
            dncnn.load_state_dict(torch.load(args.dncnn_weight_file, map_location=lambda storage, loc: storage.cuda()))
                
        if args.case_num == 2:
            print("Choosing Case 2")
            # f = FixedPointFuncCase2(dncnn, args.interp_scale, args.epsilon, alpha=args.alpha, rho=args.rho)
            f = FixedPointFuncCase2(dncnn, args.interp_scale, (args.sigmaL+args.sigmaH)/2, args.epsilon, alpha=args.alpha, rho=args.rho)
        else:
            raise Exception("Unknown case!")

        model = DEQNet(f, anderson_3, args.case_num, tol=args.tol,
                        min_iter_forward=args.miniter_forward,
                        max_iter_forward=args.maxiter_forward,
                        min_iter_backward=args.miniter_backward,
                        max_iter_backward=args.maxiter_backward, 
                        m=5, beta=args.beta, verbose=args.verbose2).to(device)

        # Load learned weights into the model
        if args.best_model:
            log_file_name = os.path.join(output_dir, 'testing_log.txt')
            model.load_state_dict(torch.load(learned_weight_file, map_location=lambda storage, loc: storage.cuda()))
        else:
            log_file_name = os.path.join(output_dir, 'checkpoint_testing_log.txt')
            checkpoint = torch.load(learned_weight_file, map_location=lambda storage, loc: storage.cuda())
            model.load_state_dict(checkpoint['last_model_state_dict'])

        # Set model in Evaluation Mode
        model.eval()

    # Metrics Recorders
    epoch_psnr_final = AverageMeter()
    epoch_ssim_final = AverageMeter()
    epoch_psnr_cnn = AverageMeter()
    epoch_ssim_cnn = AverageMeter()
    epoch_data_consistency_final = AverageMeter()
    epoch_data_consistency_cnn = AverageMeter()

    with open(log_file_name, 'w') as fp:
        input_dir = args.input_dir
        with os.scandir(input_dir) as entries:
            for i, entry in enumerate(entries):
                if entry.is_file():
                    # Read image file
                    input_file_name = os.path.join(input_dir, entry.name)
                    img_hr = imread(input_file_name, as_gray=False)

                    if img_hr.ndim != 3:
                        raise Exception("Only color images are supported!")

                    # Prepare it for Downsampling and Upsampling
                    img_height, img_width, img_channels = img_hr.shape
                    img_width = (img_width // args.interp_scale) * args.interp_scale
                    img_height = (img_height // args.interp_scale) * args.interp_scale
                    img_hr = np.asarray(Image.fromarray(img_hr).resize((img_width, img_height), resample=Image.Resampling.BICUBIC))

                    # Perform Up and then Down sampling
                    if args.interp_method == "PIL":
                        # print("Choosing PIL")
                        img_lr = pil_resize(img_hr, (1/args.interp_scale), 3)
                        img_lr_up = pil_resize(img_lr, args.interp_scale, 3)
                    elif args.interp_method == "TORCH":
                        # print("Choosing TORCH")
                        img_lr = torch_resize(img_hr.astype(np.float32), (1/args.interp_scale), 3)
                        img_lr_up = torch_resize(img_lr.astype(np.float32), args.interp_scale, 3)
                    elif args.interp_method == "CV":
                        # print("Choosing CV")
                        img_lr = cv2_resize(img_hr, (1/args.interp_scale), 3)
                        img_lr_up = cv2_resize(img_lr, args.interp_scale, 3)
                    else:
                        raise Exception("Unsupported Interpolation Method!")

                    # Clip the output to proper range because Bicubic Interpolation can produce out-of-range value
                    img_hr = img_hr.astype(np.uint8)
                    img_lr = np.clip(img_lr, a_min=0, a_max=255).astype(np.uint8)
                    img_lr_up = np.clip(img_lr_up, a_min=0, a_max=255).astype(np.uint8)

                    # Convert to YCbCr format
                    if args.fixed_dnn == 'SRCNN':
                        img_hr = img_hr.astype('float32')
                        img_lr = img_lr.astype('float32')
                        img_lr_up = img_lr_up.astype('float32')

                        # Get only Y channel from YCbCr format
                        ycbcr_hr = convert_rgb_to_ycbcr(img_hr)
                        y_hr = ycbcr_hr[:,:,0] / 255.0

                        ycbcr_lr = convert_rgb_to_ycbcr(img_lr)
                        y_lr = ycbcr_lr[:,:,0] / 255.0

                        ycbcr_lr_up = convert_rgb_to_ycbcr(img_lr_up)
                        y_lr_up = ycbcr_lr_up[:,:,0] / 255.0

                        # Convert to Torch Tensor
                        y_hr = torch.from_numpy(y_hr).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
                        y_lr = torch.from_numpy(y_lr).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
                        y_lr_up = torch.from_numpy(y_lr_up).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
                    elif args.fixed_dnn == 'RDN':                       
                        img_hr = img_hr.astype('float32')
                        img_lr = img_lr.astype('float32')

                        ycbcr_hr = convert_rgb_to_ycbcr(img_hr)
                        y_hr = ycbcr_hr[:,:,0] / 255.0
                        ycbcr_lr = convert_rgb_to_ycbcr(img_lr)
                        y_lr = ycbcr_lr[:,:,0] / 255.0
                        y_lr = torch.from_numpy(y_lr).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
                        y_hr = torch.from_numpy(y_hr).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
                        
                        img_lr = torch.from_numpy(img_lr.transpose(2,0,1))
                        img_lr = img_lr / 255.0
                        img_lr = img_lr.unsqueeze(dim=0).to(device)

                    elif args.fixed_dnn == 'RCAN':  # Fixed DNN is RCAN
                        img_hr = img_hr.astype('float32')
                        img_lr = img_lr.astype('float32')

                        ycbcr_hr = convert_rgb_to_ycbcr(img_hr)
                        y_hr = ycbcr_hr[:,:,0] / 255.0
                        y_hr = torch.from_numpy(y_hr).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
                        
                        ycbcr_lr = convert_rgb_to_ycbcr(img_lr)
                        y_lr = ycbcr_lr[:,:,0] / 255.0
                        y_lr = torch.from_numpy(y_lr).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
                                                
                        img_lr = torch.from_numpy(img_lr.transpose(2,0,1))
                        img_lr = img_lr.unsqueeze(dim=0).to(device)
                    else:
                        raise Exception("Unsupported Fixed DNN!")

                    # Predict output from the Fixed Networks
                    with torch.no_grad():
                        # Runs the forward pass under autocast.
                        if device.type == 'cuda':
                            with torch.autocast(device_type=device.type, dtype=torch.float16):
                                if args.fixed_dnn == 'SRCNN':
                                    y_lr_up = y_lr_up.to(device)
                                    y_sr = fixed_dnn(y_lr_up)
                                    y_sr = y_sr.float()
                                elif args.fixed_dnn == 'RDN':
                                    img_sr = fixed_dnn(img_lr)
                                    img_sr = 255.0 * img_sr.float()
                                    img_sr_ycbcr = convert_rgb_to_ycbcr(img_sr).float()
                                    y_sr = img_sr_ycbcr[:,:,0] / 255.0
                                    y_sr = y_sr.unsqueeze(0).unsqueeze(0)
                                    # y_lr = y_lr.unsqueeze(0).unsqueeze(0)
                                else: # Fixed DNN is'RCAN'
                                    img_sr = fixed_dnn(img_lr)
                                    img_sr = img_sr.float()
                                    img_sr_ycbcr = convert_rgb_to_ycbcr(img_sr).float()
                                    y_sr = img_sr_ycbcr[:,:,0] / 255.0
                                    y_sr = y_sr.unsqueeze(0).unsqueeze(0)
                                    # y_lr = y_lr.unsqueeze(0).unsqueeze(0)
                        elif device.type == 'cpu':
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                                if args.fixed_dnn == 'SRCNN':
                                    y_lr_up = y_lr_up.to(device)
                                    y_sr = fixed_dnn(y_lr_up)
                                    y_sr = y_sr.float()
                                elif args.fixed_dnn == 'RDN':
                                    img_sr = fixed_dnn(img_lr)
                                    img_sr = 255.0 * img_sr.float()
                                    img_sr_ycbcr = convert_rgb_to_ycbcr(img_sr).float()
                                    y_sr = img_sr_ycbcr[:,:,0] / 255.0
                                    y_sr = y_sr.unsqueeze(0).unsqueeze(0)
                                    # y_lr = y_lr.unsqueeze(0).unsqueeze(0)
                                else: # Fixed DNN is'RCAN'
                                    img_sr = fixed_dnn(img_lr)
                                    img_sr = img_sr.float()
                                    img_sr_ycbcr = convert_rgb_to_ycbcr(img_sr).float()
                                    y_sr = img_sr_ycbcr[:,:,0] / 255.0
                                    y_sr = y_sr.unsqueeze(0).unsqueeze(0)
                                    # y_lr = y_lr.unsqueeze(0).unsqueeze(0)
                        else:
                            if args.fixed_dnn == 'SRCNN':
                                y_lr_up = y_lr_up.to(device)
                                y_sr = fixed_dnn(y_lr_up)
                                y_sr = y_sr.float()
                            elif args.fixed_dnn == 'RDN':
                                img_sr = fixed_dnn(img_lr)
                                img_sr = 255.0 * img_sr.float()
                                img_sr_ycbcr = convert_rgb_to_ycbcr(img_sr).float()
                                y_sr = img_sr_ycbcr[:,:,0] / 255.0
                                y_sr = y_sr.unsqueeze(0).unsqueeze(0)
                                # y_lr = y_lr.unsqueeze(0).unsqueeze(0)
                            else: # Fixed DNN is'RCAN'
                                img_sr = fixed_dnn(img_lr)
                                img_sr = img_sr.float()
                                img_sr_ycbcr = convert_rgb_to_ycbcr(img_sr).float()
                                y_sr = img_sr_ycbcr[:,:,0] / 255.0
                                y_sr = y_sr.unsqueeze(0).unsqueeze(0)
                                # y_lr = y_lr.unsqueeze(0).unsqueeze(0)


                        if args.post_process:
                            pred_y_sr = Batch_TVTV_Solver_NUMPY(y_lr, y_sr, None, None, 1.0, args.interp_scale, 3, args.interp_method, False, False, 1000)
                            # pred_y_sr = Batch_Smooth_TVTV_Solver_NUMPY(y_lr, y_sr, None, None, 1.0, 0.01, args.interp_scale, 3, args.interp_method, False, False, 150)
                            pred_y_sr = pred_y_sr.to(y_sr)
                        else:
                            pred_y_sr, _, _ = model(y_sr, y_lr, compute_jac_loss=False, spectral_radius_mode=False)
                        
                        
                        # Convert predicted outputs to Numpy
                        y_hr        = y_hr.detach().cpu().numpy().squeeze()
                        pred_y_sr   = pred_y_sr.detach().cpu().numpy().squeeze()
                        y_lr        = y_lr.detach().cpu().numpy().squeeze()
                        y_sr        = y_sr.detach().cpu().numpy().squeeze()
                        
                        # Calculate Data Fidelity Error
                        if args.interp_method == 'PIL':
                            final_data_error = np.linalg.norm((pil_resize(pred_y_sr, scale=(1/args.interp_scale), order=3) - y_lr).flatten())
                            cnn_data_error = np.linalg.norm((pil_resize(y_sr, scale=(1/args.interp_scale), order=3) - y_lr).flatten())
                        else:
                            final_data_error = np.linalg.norm((torch_resize(pred_y_sr, scale=(1/args.interp_scale), order=3) - y_lr).flatten())
                            cnn_data_error = np.linalg.norm((torch_resize(y_sr, scale=(1/args.interp_scale), order=3) - y_lr).flatten())
                            
                        # Clip the output images to range [0, 1] for measuring Image Qualities Metrics
                        # pred_y_hr = pred_y_hr.clip(0.0, 1.0)
                        # w = w.clip(0.0, 1.0)
                        
                        
                        # Remove edges of the image before calculating Metrics
                        y_hr = y_hr[args.interp_scale:-args.interp_scale, args.interp_scale:-args.interp_scale]
                        pred_y_sr = pred_y_sr[args.interp_scale:-args.interp_scale, args.interp_scale:-args.interp_scale]
                        y_sr = y_sr[args.interp_scale:-args.interp_scale, args.interp_scale:-args.interp_scale]

                        # Calculate PSNR and SSIM
                        final_psnr = calc_psnr(y_hr, pred_y_sr, data_range=1.0)
                        cnn_psnr = calc_psnr(y_hr, y_sr, data_range=1.0)
                        final_ssim = calc_ssim(y_hr, pred_y_sr, data_range=1.0)
                        cnn_ssim = calc_ssim(y_hr, y_sr, data_range=1.0)

                        epoch_psnr_final.update(final_psnr)
                        epoch_ssim_final.update(final_ssim)
                        epoch_psnr_cnn.update(cnn_psnr)
                        epoch_ssim_cnn.update(cnn_ssim)
                        epoch_data_consistency_final.update(final_data_error)
                        epoch_data_consistency_cnn.update(cnn_data_error)

                        print("Image File Name:                         ", entry.name, file=fp)
                        print("Final Data Consistency:                   {:.8e}".format(final_data_error), file=fp)
                        print("CNN Data Consistency:                     {:.8e}".format(cnn_data_error), file=fp)
                        print("Final Output PSNR:                        {:.4f} & SSIM:{:.4f}".format(final_psnr, final_ssim), file=fp)
                        print("CNN Output PSNR:                          {:.4f} & SSIM:{:.4f}".format(cnn_psnr, cnn_ssim), file=fp)
                        print('Final Output Avg PSNR:                    {:.4f} & SSIM: {:.4f}'.format(epoch_psnr_final.avg, epoch_ssim_final.avg), file=fp)
                        print('CNN Output Avg PSNR:                      {:.4f} & SSIM: {:.4f}'.format(epoch_psnr_cnn.avg, epoch_ssim_cnn.avg), file=fp)
                        print("Final Avg Data Consistency:               {:8e}".format(epoch_data_consistency_final.avg), file=fp)
                        print("CNN Avg Data Consistency:                 {:8e}".format(epoch_data_consistency_cnn.avg), file=fp)
                        print("\n", file=fp)

                        if args.save_image:
                            # Convert Images back to range [0 255.]
                            pred_y_sr = 255.0 * pred_y_sr
                            y_sr = 255.0 * y_sr

                            hr_img = np.array([ycbcr_hr[..., 0], ycbcr_hr[..., 1], ycbcr_hr[..., 2]]).transpose([1, 2, 0])
                            hr_img = np.clip(convert_ycbcr_to_rgb(hr_img), 0.0, 255.0).astype(np.uint8)

                            pred_hr_img = np.array([pred_y_sr, ycbcr_hr[..., 1], ycbcr_hr[..., 2]]).transpose([1, 2, 0])
                            pred_hr_img = np.clip(convert_ycbcr_to_rgb(pred_hr_img), 0.0, 255.0).astype(np.uint8)

                            cnn_hr_img = np.array([y_sr, ycbcr_hr[..., 1], ycbcr_hr[..., 2]]).transpose([1, 2, 0])
                            cnn_hr_img = np.clip(convert_ycbcr_to_rgb(cnn_hr_img), 0.0, 255.0).astype(np.uint8)

                            # Save the images
                            tail = os.path.split(input_file_name)[1]

                            file_name = tail.replace('.', '_original.')
                            output_file_name = os.path.join(output_dir, file_name)
                            imsave(output_file_name, hr_img)

                            file_name = tail.replace('.', '_final_x{}.'.format(args.interp_scale))
                            output_file_name_final = os.path.join(output_dir, file_name)
                            imsave(output_file_name_final, pred_hr_img)

                            file_name = tail.replace('.', '_cnn_x{}.'.format(args.interp_scale))
                            output_file_name_cnn = os.path.join(output_dir, file_name)
                            imsave(output_file_name_cnn, cnn_hr_img)

        fp.close()
        print("End of Test Script.")