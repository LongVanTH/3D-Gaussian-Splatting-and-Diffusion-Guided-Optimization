import os
import torch
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils import visualize_renders
from data_utils_harder_scene import get_nerf_datasets, trivial_collate, compute_ssim, get_expon_lr_func

from pytorch3d.renderer import PerspectiveCameras
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    attrs = ["means", "pre_act_scales", "colours", "pre_act_opacities"]
    if not gaussians.is_isotropic:
        attrs += ["pre_act_quats"]

    for attr in attrs:
        param = getattr(gaussians, attr)
        if not getattr(param, "requires_grad", False):
            param.requires_grad = True

def setup_optimizer(gaussians):

    gaussians.check_if_trainable()

    ### YOUR CODE HERE ###
    # HINT: Modify the learning rates to reasonable values. We have intentionally
    # set very high learning rates for all parameters.
    # HINT: Consider reducing the learning rates for parameters that seem to vary too
    # fast with the default settings.
    # HINT: Consider setting different learning rates for different sets of parameters.
    # Gaussian Splatting original default learning rates can be found here:
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/arguments/__init__.py#L71
    means_lr_init = 0.00016
    means_lr_final = 0.0000016
    means_lr_delay_mult = 0.01
    max_steps = 30000
    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': 0.05, "name": "opacities"},
        {'params': [gaussians.pre_act_scales], 'lr': 0.005, "name": "scales"},
        {'params': [gaussians.colours], 'lr': 0.0025, "name": "colours"},
        {'params': [gaussians.means], 'lr': 0.00016, "name": "means"},
        {'params': [gaussians.pre_act_quats], 'lr': 0.001, "name": "quats"}, # Uncomment if isotropic is False
    ]
    optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)
    means_scheduler = get_expon_lr_func(
        lr_init=means_lr_init,
        lr_final=means_lr_final,
        lr_delay_mult=means_lr_delay_mult,
        max_steps=max_steps
    )

    return optimizer, means_scheduler

def update_lr(optimizer, scheduler, itr):
    for param_group in optimizer.param_groups:
        if param_group["name"] == "means":
            param_group["lr"] = scheduler(itr)

def ndc_to_screen_camera(camera, img_size = (128, 128)):

    min_size = min(img_size[0], img_size[1])

    screen_focal = camera.focal_length * min_size / 2.0
    screen_principal = torch.tensor([[img_size[0]/2, img_size[1]/2]]).to(torch.float32)

    return PerspectiveCameras(
        R=camera.R, T=camera.T, in_ndc=False,
        focal_length=screen_focal, principal_point=screen_principal,
        image_size=(img_size,),
    )

def run_training(args):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    debug_root = os.path.join(args.out_path, "q1_training")
    if not os.path.exists(debug_root):
        os.makedirs(debug_root, exist_ok=True)

    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name="materials", data_root=args.data_path,
        image_size=[128, 128],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    train_itr = iter(train_loader)

    # Preparing some code for visualization
    viz_gif_path_1 = os.path.join(args.out_path, "q1_harder_training_progress.gif")
    viz_gif_path_2 = os.path.join(args.out_path, "q1_harder_training_final_renders.gif")
    viz_idxs = np.linspace(0, len(train_dataset)-1, 5).astype(np.int32)[:4]

    gt_viz_imgs = [(train_dataset[i]["image"]*255.0).numpy().astype(np.uint8) for i in viz_idxs]
    gt_viz_imgs = [np.array(Image.fromarray(x).resize((256, 256))) for x in gt_viz_imgs]
    gt_viz_img = np.concatenate(gt_viz_imgs, axis=1)

    viz_cameras = [ndc_to_screen_camera(train_dataset[i]["camera"]).cuda() for i in viz_idxs]

    # Init gaussians and scene
    gaussians = Gaussians(
        num_points=10000, init_type="random",
        device=args.device, isotropic=False
    )
    scene = Scene(gaussians)

    # Making gaussians trainable and setting up optimizer
    make_trainable(gaussians)
    optimizer, means_scheduler = setup_optimizer(gaussians)

    # Training loop
    viz_frames = []
    for itr in range(args.num_itrs):

        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)

        gt_img = data[0]["image"].cuda()
        camera = ndc_to_screen_camera(data[0]["camera"]).cuda()

        # Rendering scene using gaussian splatting
        ### YOUR CODE HERE ###
        # HINT: Can any function from the Scene class help?
        # HINT: Set bg_colour to (0.0, 0.0, 0.0)
        # HINT: Set img_size to (128, 128)
        # HINT: Get per_splat from args.gaussians_per_splat
        # HINT: camera is available above
        pred_img = scene.render(
            camera = camera,
            per_splat = args.gaussians_per_splat,
            img_size = (128, 128),
            bg_colour = (0.0, 0.0, 0.0)
        )[0]

        # Compute loss
        ### YOUR CODE HERE ###
        update_lr(optimizer, means_scheduler, itr)
        l1_loss = torch.nn.functional.l1_loss(pred_img, gt_img)
        ssim_loss = 1.0 - compute_ssim(pred_img, gt_img)
        lambda_ssim = 0.2
        loss = (1.0 - lambda_ssim) * l1_loss + lambda_ssim * ssim_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"[*] Itr: {itr:07d} | Loss: {loss:0.3f}")

        if itr % args.viz_freq == 0:
            viz_frame = visualize_renders(
                scene, gt_viz_img,
                viz_cameras, (128, 128)
            )
            viz_frames.append(viz_frame)
            debug_path = os.path.join(debug_root, f"{itr:07d}.png")
            imageio.imwrite(debug_path, viz_frame)

            # Saving training progess GIF
            imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1/10.0)*1000)

            # Creating renderings of the training views after training is completed.
            frames = []
            viz_loader = DataLoader(
                train_dataset, batch_size=1, shuffle=False, num_workers=0,
                drop_last=True, collate_fn=trivial_collate
            )
            for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
                gt_img = viz_data[0]["image"].cuda()
                camera = ndc_to_screen_camera(viz_data[0]["camera"]).cuda()

                with torch.no_grad():

                    # Rendering scene using gaussian splatting
                    ### YOUR CODE HERE ###
                    # HINT: Can any function from the Scene class help?
                    # HINT: Set bg_colour to (0.0, 0.0, 0.0)
                    # HINT: Set img_size to (128, 128)
                    # HINT: Get per_splat from args.gaussians_per_splat
                    # HINT: camera is available above
                    pred_img = scene.render(
                        camera = camera,
                        per_splat = args.gaussians_per_splat,
                        img_size = (128, 128),
                        bg_colour = (0.0, 0.0, 0.0)
                    )[0]

                pred_npy = pred_img.detach().cpu().numpy()
                pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
                frames.append(pred_npy)

            # Saving renderings
            viz_gif_path_3 = os.path.join(debug_root, f"q1_training_renders_{itr:07d}.gif")
            imageio.mimwrite(viz_gif_path_3, frames, loop=0, duration=(1/10.0)*1000)

            # Running evaluation using the test dataset
            psnr_vals, ssim_vals = [], []
            for val_data in tqdm(val_loader, desc="Running Evaluation"):

                gt_img = val_data[0]["image"].cuda()
                camera = ndc_to_screen_camera(val_data[0]["camera"]).cuda()

                with torch.no_grad():

                    # Rendering scene using gaussian splatting
                    ### YOUR CODE HERE ###
                    # HINT: Can any function from the Scene class help?
                    # HINT: Set bg_colour to (0.0, 0.0, 0.0)
                    # HINT: Set img_size to (128, 128)
                    # HINT: Get per_splat from args.gaussians_per_splat
                    # HINT: camera is available above
                    pred_img = scene.render(
                        camera = camera,
                        per_splat = args.gaussians_per_splat,
                        img_size = (128, 128),
                        bg_colour = (0.0, 0.0, 0.0)
                    )[0]

                    gt_npy = gt_img.detach().cpu().numpy()
                    pred_npy = pred_img.detach().cpu().numpy()
                    psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
                    ssim = structural_similarity(gt_npy, pred_npy, channel_axis=-1, data_range=1.0)

                    psnr_vals.append(psnr)
                    ssim_vals.append(ssim)

            mean_psnr = np.mean(psnr_vals)
            mean_ssim = np.mean(ssim_vals)
            print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
            print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")

    print("[*] Training Completed.")

    # Saving training progess GIF
    imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1/10.0)*1000)

    # Creating renderings of the training views after training is completed.
    frames = []
    viz_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
        gt_img = viz_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(viz_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img = scene.render(
                camera = camera,
                per_splat = args.gaussians_per_splat,
                img_size = (128, 128),
                bg_colour = (0.0, 0.0, 0.0)
            )[0]

        pred_npy = pred_img.detach().cpu().numpy()
        pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(pred_npy)

    # Saving renderings
    imageio.mimwrite(viz_gif_path_2, frames, loop=0, duration=(1/10.0)*1000)

    # Running evaluation using the test dataset
    psnr_vals, ssim_vals = [], []
    for val_data in tqdm(val_loader, desc="Running Evaluation"):

        gt_img = val_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(val_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            # HINT: Can any function from the Scene class help?
            # HINT: Set bg_colour to (0.0, 0.0, 0.0)
            # HINT: Set img_size to (128, 128)
            # HINT: Get per_splat from args.gaussians_per_splat
            # HINT: camera is available above
            pred_img = scene.render(
                camera = camera,
                per_splat = args.gaussians_per_splat,
                img_size = (128, 128),
                bg_colour = (0.0, 0.0, 0.0)
            )[0]

            gt_npy = gt_img.detach().cpu().numpy()
            pred_npy = pred_img.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
            ssim = structural_similarity(gt_npy, pred_npy, channel_axis=-1, data_range=1.0)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
    print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/materials", type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )
    parser.add_argument(
        "--num_itrs", default=1000, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=20, type=int,
        help="Frequency with which visualization should be performed."
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)
