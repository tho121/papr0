import os, sys

sys.path.append(os.getcwd())

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from imageio import imwrite
from pydantic import validator
import kornia
import cv2
import math

from torchvision import transforms

from my.utils import (
    tqdm, EventStorage, HeartBeat, EarlyLoopBreak,
    get_event_storage, get_heartbeat, read_stats
)
from my.config import BaseConf, dispatch, optional_load_config, write_full_config
from my.utils.seed import seed_everything

from adapt import ScoreAdapter, karras_t_schedule
from run_img_sampling import SD, StableDiffusion
from misc import torch_samps_to_imgs
from pose import PoseConfig, camera_pose, sample_near_eye

from run_nerf import VoxConfig
from voxnerf.utils import every
from voxnerf.render import (
    as_torch_tsrs, rays_from_img, ray_box_intersect, render_ray_bundle
)
from voxnerf.vis import stitch_vis, bad_vis as nerf_vis, vis_img
from voxnerf.data import load_blender
from my3d import get_T, depth_smooth_loss

from run_zero123 import evaluate, tsr_stats, vis_routine, scene_box_filter

# # diet nerf
# import sys
# sys.path.append('./DietNeRF/dietnerf')
# from DietNeRF.dietnerf.run_nerf import get_embed_fn
# import torch.nn.functional as F

#######################

import yaml
import argparse
import torch
import os
import shutil
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import bisect
import time
import sys
import io
import imageio
from PIL import Image
from logger import *
from dataset import get_dataset, get_loader
from models import get_model, get_loss
from train import parse_args, DictAsMember, setup_seed
from models.model import PAPR

device_glb = torch.device("cuda")

class SJC_0(BaseConf):
    family:     str = "sd"
    sd:         SD = SD(
        variant="objaverse",
        scale=100.0
    )
    lr:         float = 0.05
    n_steps:    int = 10000
    vox:        VoxConfig = VoxConfig(
        model_type="V_SD", grid_size=100, density_shift=-1.0, c=3,
        blend_bg_texture=False, bg_texture_hw=4,
        bbox_len=1.0
    )
    pose:       PoseConfig = PoseConfig(rend_hw=32, FoV=49.1, R=2.0)

    emptiness_scale:    int = 10
    emptiness_weight:   int = 0
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    grad_accum: int = 1

    depth_smooth_weight: float = 1e5
    near_view_weight: float = 1e5

    depth_weight:       int = 0

    var_red:     bool = True

    train_view:         bool = True
    scene:              str = 'chair'
    index:              int = 2

    view_weight:        int = 10000
    prefix:             str = 'exp'
    nerf_path:          str = "data/nerf_wild"

    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def ready(self):
        cfgs = self.dict()

        family = cfgs.pop("family")
        model = getattr(self, family).make()

        cfgs.pop("vox")
        vox = self.vox.make()

        cfgs.pop("pose")
        poser = self.pose.make()

        return cfgs, model, vox, poser

        #sjc_3d(**cfgs, poser=poser, model=model, vox=vox, 
        #       papr_model=self.papr_model, papr_losses=self.papr_losses, papr_args=self.papr_args)

    #def set_papr_model(self, papr_model, papr_losses, papr_args):
    #    self.papr_model = papr_model
    #    self.papr_losses = papr_losses
    #    self.papr_args = papr_args

papr_batch_size = 12
papr_steps_per_it = 1000
papr_train_interval = 500

#1000 zero it = 2000 papr it
#5000 zeri it = 10000 papr it

def sjc_3d(poser, vox, model: ScoreAdapter, papr_model: PAPR, papr_losses, papr_args,
    lr, n_steps, emptiness_scale, emptiness_weight, emptiness_step, emptiness_multiplier,
    depth_weight, var_red, train_view, scene, index, view_weight, prefix, nerf_path, \
    depth_smooth_weight, near_view_weight, grad_accum, **kwargs):

    assert model.samps_centered()
    _, target_H, target_W = model.data_shape()
    bs = 1
    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)
    opt = torch.optim.Adamax(vox.opt_params(), lr=lr)

    H, W = poser.H, poser.W
    Ks, poses, prompt_prefixes = poser.sample_train(n_steps)

    ts = model.us[30:-10]
    fuse = EarlyLoopBreak(5)

    same_noise = torch.randn(1, 4, H, W, device=model.device).repeat(bs, 1, 1, 1)

    folder_name = prefix + '/scene-%s-index-%d_scale-%s_train-view-%s_view-weight-%s_depth-smooth-wt-%s_near-view-wt-%s' % \
                            (scene, index, model.scale, train_view, view_weight, depth_smooth_weight, near_view_weight)

    # load nerf view
    images_, _, poses_, mask_, fov_x = load_blender('train', scene=scene, path=nerf_path)
    # K_ = poser.get_K(H, W, fov_x * 180. / math.pi)
    K_ = poser.K
    input_image, input_K, input_pose, input_mask = images_[index], K_, poses_[index], mask_[index]
    input_pose[:3, -1] = input_pose[:3, -1] / np.linalg.norm(input_pose[:3, -1]) * poser.R
    background_mask, image_mask = input_mask == 0., input_mask != 0.
    input_image = cv2.resize(input_image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    image_mask = cv2.resize(image_mask.astype(np.float32), dsize=(256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
    background_mask = cv2.resize(background_mask.astype(np.float32), dsize=(H, W), interpolation=cv2.INTER_NEAREST).astype(bool)

    # to torch tensor
    input_image = torch.as_tensor(input_image, dtype=float, device=device_glb)
    input_image = input_image.permute(2, 0, 1)[None, :, :]
    input_image = input_image * 2. - 1.
    image_mask = torch.as_tensor(image_mask, dtype=bool, device=device_glb)
    image_mask = image_mask[None, None, :, :].repeat(1, 3, 1, 1)
    background_mask = torch.as_tensor(background_mask, dtype=bool, device=device_glb)

    print('==== loaded input view for training ====')

    opt.zero_grad()

    papr_loss_fn = get_loss(papr_args.training.losses)
    papr_loss_fn = papr_loss_fn.to(papr_model.device)

    papr_log_dir = os.path.join(papr_args.save_dir, papr_args.index)
    os.makedirs(os.path.join(papr_log_dir, "test"), exist_ok=True)
    papr_log_dir = os.path.join(papr_log_dir, "test")

    papr_y_batch = torch.empty((papr_batch_size, H, W, 3), dtype=torch.float16, device=papr_model.device)
    papr_ro_batch = torch.empty((papr_batch_size, 3), dtype=torch.float32, device=papr_model.device)
    papr_rd_batch = torch.empty((papr_batch_size, H, W, 3), dtype=torch.float32, device=papr_model.device)
    papr_c2w_batch = np.empty((papr_batch_size, 4, 4), dtype=np.float64)

    coord_scale = papr_args.dataset.coord_scale
    scaling_matrix = torch.tensor([[coord_scale, 0, 0, 0],
                                           [0, coord_scale, 0, 0],
                                           [0, 0, coord_scale, 0],
                                           [0, 0, 0, 1]], dtype=torch.float32, device=papr_model.device)
    np_scaling_matrix = scaling_matrix.cpu().numpy()
    
    papr_steps = 0

    with tqdm(total=n_steps) as pbar, \
        HeartBeat(pbar) as hbeat, \
            EventStorage(folder_name) as metric:
        
        with torch.no_grad():

            tforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((256, 256))
            ])

            #TODO: resize to target rayd size
            
            '''
            import torch.nn.functional as F

            # Assuming rd_ is a tensor with shape [32, 32, 3], we transpose it to [3, 32, 32]
            rd_ = rd_.permute(2, 0, 1).unsqueeze(0)

            # Now rd_ has a shape of [1, 3, 32, 32], which is [batch_size, channels, height, width]
            # Use interpolate to resize it to [1, 3, 256, 256]
            rd_expanded = F.interpolate(rd_, size=(256, 256), mode='bilinear', align_corners=False)

            # After interpolation, we remove the batch dimension and transpose back to [256, 256, 3]
            rd_expanded = rd_expanded.squeeze(0).permute(1, 2, 0)
            '''


            papr_tforms = transforms.Resize((H, W))

            input_im = tforms(input_image)

            # get input embedding
            model.clip_emb = model.model.get_learned_conditioning(input_im.float()).tile(1,1,1).detach()
            model.vae_emb = model.model.encode_first_stage(input_im.float()).mode().detach()

        for i in range(n_steps):
            if fuse.on_break():
                break
            
            if train_view:

                # supervise with input view
                # if i < 100 or i % 10 == 0:
                with torch.enable_grad():
                    y_, depth_, ro_, rd_, ws_ = render_one_view(vox, aabb, H, W, input_K, input_pose, return_w=True)
                    y_ = model.decode(y_)
                rgb_loss = ((y_ - input_image) ** 2).mean()

                # depth smoothness loss
                input_smooth_loss = depth_smooth_loss(depth_) * depth_smooth_weight * 0.1
                input_smooth_loss.backward(retain_graph=True)

                input_loss = rgb_loss * float(view_weight)
                input_loss.backward(retain_graph=True)
                if train_view and i % 100 == 0:
                    metric.put_artifact("input_view", ".png", lambda fn: imwrite(fn, torch_samps_to_imgs(y_)[0]))

            # y: [1, 4, 64, 64] depth: [64, 64]  ws: [n, 4096]
            y, depth, ro, rd, ws = render_one_view(vox, aabb, H, W, Ks[i], poses[i], return_w=True)

            # near-by view
            eye = poses[i][:3, -1]
            near_eye = sample_near_eye(eye)
            near_pose = camera_pose(near_eye, -near_eye, poser.up)
            y_near, depth_near, ro_near, rd_near, ws_near = render_one_view(vox, aabb, H, W, Ks[i], near_pose, return_w=True)
            near_loss = ((y_near - y).abs().mean() + (depth_near - depth).abs().mean()) * near_view_weight
            near_loss.backward(retain_graph=True)

            # get T from input view
            pose = poses[i]
            T_target = pose[:3, -1]
            T_cond = input_pose[:3, -1]
            T = get_T(T_target, T_cond).to(model.device)

            if isinstance(model, StableDiffusion):
                pass
            else:
                y = torch.nn.functional.interpolate(y, (target_H, target_W), mode='bilinear')

            with torch.no_grad():
                chosen_σs = np.random.choice(ts, bs, replace=False)
                chosen_σs = chosen_σs.reshape(-1, 1, 1, 1)
                chosen_σs = torch.as_tensor(chosen_σs, device=model.device, dtype=torch.float32)
                # chosen_σs = us[i]

                noise = torch.randn(bs, *y.shape[1:], device=model.device)

                zs = y + chosen_σs * noise

                score_conds = model.img_emb(input_im, conditioning_key='hybrid', T=T)

                Ds = model.denoise_objaverse(zs, chosen_σs, score_conds)

                if var_red:
                    grad = (Ds - y) / chosen_σs
                else:
                    grad = (Ds - zs) / chosen_σs

                grad = grad.mean(0, keepdim=True)

            y.backward(-grad, retain_graph=True)

            emptiness_loss = (torch.log(1 + emptiness_scale * ws) * (-1 / 2 * ws)).mean() # negative emptiness loss
            emptiness_loss = emptiness_weight * emptiness_loss
            # if emptiness_step * n_steps <= i:
            #     emptiness_loss *= emptiness_multiplier
            emptiness_loss = emptiness_loss * (1. + emptiness_multiplier * i / n_steps)
            emptiness_loss.backward(retain_graph=True)

            # depth smoothness loss
            smooth_loss = depth_smooth_loss(depth) * depth_smooth_weight

            if i >= emptiness_step * n_steps:
                smooth_loss.backward(retain_graph=True)

            depth_value = depth.clone()

            #batch data for PAPR, forward pass when batch filled
            if papr_train_interval - (i % papr_train_interval) < papr_batch_size:
                with torch.no_grad():
                    papr_y = model.decode(y)
                    papr_y = papr_tforms(papr_y).permute(0, 2, 3, 1)

                papr_batch_i = (i % papr_train_interval) % papr_batch_size
                papr_y_batch[papr_batch_i] = papr_y.detach()

                ro_homogeneous = torch.cat((ro[0].detach(), torch.ones_like(ro[0][..., :1])), dim=-1)
                ro_scaled = torch.matmul(scaling_matrix, ro_homogeneous.unsqueeze(-1)).squeeze(-1)

                papr_ro_batch[papr_batch_i] = ro_scaled[:3]
                papr_rd_batch[papr_batch_i] = rd.view(H, W, 3).detach()
                papr_c2w_batch[papr_batch_i] = np.dot(np_scaling_matrix, poses[i])

            if (i + 1) % papr_train_interval == 0:

                papr_y_batch[0] = papr_tforms(input_image).permute(0, 2, 3, 1).detach()

                ro_homogeneous_ = torch.cat((ro_[0].detach(), torch.ones_like(ro[0][..., :1])), dim=-1)
                ro_scaled_ = torch.matmul(scaling_matrix, ro_homogeneous_.unsqueeze(-1)).squeeze(-1)

                papr_ro_batch[0] = ro_scaled_[:3]
                papr_rd_batch[0] = rd_.view(H, W, 3).detach()
                papr_c2w_batch[0] = np.dot(np_scaling_matrix, input_pose)

                target_step = papr_steps + papr_steps_per_it
                papr_train_and_eval(papr_steps, target_step, papr_model, model.device, [[papr_y_batch, papr_ro_batch, papr_rd_batch, papr_c2w_batch]], papr_losses, papr_loss_fn, papr_log_dir, papr_args)
                papr_steps += papr_steps_per_it

            if i % grad_accum == (grad_accum-1):
                opt.step()
                opt.zero_grad()
            
            metric.put_scalars(**tsr_stats(y))

            if i % 1000 == 0 and i != 0:
                with EventStorage(model.im_path.replace('/', '-') + '_scale-' + str(model.scale) + "_test"):
                    evaluate(model, vox, poser)

            if every(pbar, percent=1):
                with torch.no_grad():
                    if isinstance(model, StableDiffusion):
                        y = model.decode(y)
                    vis_routine(metric, y, depth_value)

            metric.step()
            pbar.update()
            pbar.set_description(model.im_path)
            hbeat.beat()

        metric.put_artifact(
            "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
        )
        with EventStorage("test"):
            evaluate(model, vox, poser)

        metric.step()

        hbeat.done()

def render_one_view(vox, aabb, H, W, K, pose, return_w=False):
    N = H * W
    ro, rd = rays_from_img(H, W, K, pose)
    ro, rd, t_min, t_max = scene_box_filter(ro, rd, aabb)
    assert len(ro) == N, "for now all pixels must be in"
    ro, rd, t_min, t_max = as_torch_tsrs(vox.device, ro, rd, t_min, t_max)
    rgbs, depth, weights = render_ray_bundle(vox, ro, rd, t_min, t_max)

    rgbs = rearrange(rgbs, "(h w) c -> 1 c h w", h=H, w=W)
    depth = rearrange(depth, "(h w) 1 -> h w", h=H, w=W)
    weights = rearrange(weights, "N (h w) 1 -> N h w", h=H, w=W)
    if return_w:
        return rgbs, depth, ro, rd, weights
    else:
        return rgbs, depth, ro, rd

def create_papr(args, eval_args, resume):
    log_dir = os.path.join(args.save_dir, args.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args, device)
    #dataset = get_dataset(args.dataset, mode="train")
    #eval_dataset = get_dataset(eval_args.dataset, mode="test")
    model = model.to(device)

    start_step = 0
    losses = [[], [], []]
    if resume > 0:
        start_step = model.load(log_dir)

        train_losses = torch.load(os.path.join(log_dir, "train_losses.pth")).tolist()
        eval_losses = torch.load(os.path.join(log_dir, "eval_losses.pth")).tolist()
        eval_psnrs = torch.load(os.path.join(log_dir, "eval_psnrs.pth")).tolist()
        losses = [train_losses, eval_losses, eval_psnrs]

        print("!!!!! Resume from step %s" % start_step)
    elif args.load_path:
        try:
            resume_step = model.load(os.path.join(args.save_dir, args.load_path))
        except:
            model_state_dict = torch.load(os.path.join(args.save_dir, args.load_path, "model.pth"))
            for step, state_dict in model_state_dict.items():
                resume_step = step
                model.load_my_state_dict(state_dict)
        print("!!!!! Loaded model from %s at step %s" % (args.load_path, resume_step))

    #train_and_eval(start_step, model, device, dataset, eval_dataset, losses, args)
    #print(torch.cuda.memory_summary())
        
    return model, losses

def papr_train_and_eval(start_step, target_step, model, device, data, losses, loss_fn, log_dir, args):

    steps = []
    #train_losses, eval_losses, eval_psnrs = losses

    train_losses = []
    eval_losses = []
    eval_psnrs = []

    pt_lrs = []
    tx_lrs = []

    avg_train_loss = 0.
    step = start_step
    eval_step_cnt = start_step
    pruned = False

    print("Start step:", start_step, "Total steps:", target_step)
    start_time = time.time()
    while step < target_step:
        for _, batch in enumerate(data):
            if (args.training.prune_steps > 0) and (step < args.training.prune_stop) and (step >= args.training.prune_start):
                if len(args.training.prune_steps_list) > 0 and step % args.training.prune_steps == 0:
                    cur_prune_thresh = args.training.prune_thresh_list[bisect.bisect_left(args.training.prune_steps_list, step)]
                    model.clear_optimizer()
                    model.clear_scheduler()
                    num_pruned = model.prune_points(cur_prune_thresh)
                    model.init_optimizers(step)
                    pruned = True
                    print("Step %d: Pruned %d points, prune threshold %f" % (step, num_pruned, cur_prune_thresh))

                elif step % args.training.prune_steps == 0:
                    model.clear_optimizer()
                    model.clear_scheduler()
                    num_pruned = model.prune_points(args.training.prune_thresh)
                    model.init_optimizers(step)
                    pruned = True
                    print("Step %d: Pruned %d points" % (step, num_pruned))

            if pruned and len(args.training.add_steps_list) > 0:
                if step in args.training.add_steps_list:
                    cur_add_num = args.training.add_num_list[args.training.add_steps_list.index(step)]
                    if 'max_num_pts' in args and args.max_num_pts > 0:
                        cur_add_num = min(cur_add_num, args.max_num_pts - model.points.shape[0])
                    
                    if cur_add_num > 0:
                        model.clear_optimizer()
                        model.clear_scheduler()
                        num_added = model.add_points(cur_add_num)
                        model.init_optimizers(step)
                        model.added_points = True
                        print("Step %d: Added %d points" % (step, num_added))

            elif pruned and (args.training.add_steps > 0) and (step % args.training.add_steps == 0) and (step < args.training.add_stop) and (step >= args.training.add_start):
                cur_add_num = args.training.add_num
                if 'max_num_pts' in args and args.max_num_pts > 0:
                    cur_add_num = min(cur_add_num, args.max_num_pts - model.points.shape[0])
                
                if cur_add_num > 0:
                    model.clear_optimizer()
                    model.clear_scheduler()
                    num_added = model.add_points(args.training.add_num)
                    model.init_optimizers(step)
                    model.added_points = True
                    print("Step %d: Added %d points" % (step, num_added))

            loss, out, tgt = train_step(step, model, batch, loss_fn, args)
            avg_train_loss += loss
            step += 1
            eval_step_cnt += 1
            
            if step % 200 == 0:
                time_used = time.time() - start_time
                print("Train step:", step, "loss:", loss, "tx_lr:", model.tx_lr, "pts_lr:", model.pts_lr, "scale:", model.scaler.get_scale(), f"time: {time_used:.2f}s")
                start_time = time.time()

            if (step % args.eval.step == 0) or (step % 500 == 0 and step < 10000):
                train_losses.append(avg_train_loss / eval_step_cnt)
                pt_lrs.append(model.pts_lr)
                tx_lrs.append(model.tx_lr)
                steps.append(step)
                eval_step(steps, model, device, batch, loss_fn, out, tgt, args, train_losses, eval_losses, eval_psnrs, pt_lrs, tx_lrs)
                avg_train_loss = 0.
                eval_step_cnt = 0

            out = out.detach().cpu().numpy()
            tgt = tgt.detach().cpu().numpy()

            if ((step - 1) % 200 == 0) and args.eval.save_fig:
                coord_scale = args.dataset.coord_scale
                pt_plot_scale = 0.8 * coord_scale
                
                img_dir = os.path.join(log_dir, "images")
                os.makedirs(img_dir, exist_ok=True)
                out_path = os.path.join(img_dir, f"out_frame_step_{step}.jpg")

                #Save model output and target images
                img1 = Image.fromarray((out[0] * 255).astype('uint8'))
                img2 = Image.fromarray((tgt[0] * 255).astype('uint8'))

                width1, height1 = img1.size
                width2, height2 = img2.size
                total_width = width1 + width2
                max_height = max(height1, height2)

                combined_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
                combined_image.paste(img1, (0, 0))
                combined_image.paste(img2, (width1, 0))

                combined_image.save(out_path)

                #save images of point cloud

                pc_dir = os.path.join(log_dir, "point_clouds")
                os.makedirs(pc_dir, exist_ok=True)

                points_np = model.points.detach().cpu().numpy()
                points_influ_scores_np = None
                if model.points_influ_scores is not None:
                    points_influ_scores_np = model.points_influ_scores.squeeze().detach().cpu().numpy()
                pcd_plot = get_training_pcd_single_plot(step, points_np, pt_plot_scale, points_influ_scores_np)
  
                image_path = os.path.join(pc_dir, f"pc_frame_step_{step}.png")
                pcd_plot.save(image_path)
                

            if step >= target_step:
                break

    print("Training finished!")

def train_step(step, model, batch, loss_fn, args):
    tgt, rayo, rayd, c2w = batch

    tgt = tgt.to(torch.float32)

    model.clear_grad()
    out = model(rayo, rayd, c2w, step)
    out = model.last_act(out)
    loss = loss_fn(out, tgt)
    model.scaler.scale(loss).backward()
    model.step(step)
    if args.scaler_min_scale > 0 and model.scaler.get_scale() < args.scaler_min_scale:
        model.scaler.update(args.scaler_min_scale)
    else:
        model.scaler.update()

    return loss.item(), out, tgt

def eval_step(steps, model, device, batch, loss_fn, train_out, train_tgt, args, train_losses, eval_losses, eval_psnrs, pt_lrs, tx_lrs):
    step = steps[-1]
    tgt, rayo, rayd, c2w = batch
    
    with torch.no_grad():
        eval_loss = loss_fn(train_out, train_tgt)
        eval_psnr = -10. * np.log(((train_out - train_tgt)**2).mean().item()) / np.log(10.)
        
        eval_losses.append(eval_loss.item())
        eval_psnrs.append(eval_psnr.item())

    model.clear_grad()

    N, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape

    rand_idx = 0
    if N > 1:
        rand_idx = random.randint(1, N-1)

    topk = min([num_pts, model.select_k])

    selected_points = torch.zeros(1, H, W, topk, 3)

    with torch.no_grad():
        _, attn = model.evaluate(rayo[rand_idx], rayd[rand_idx].unsqueeze(0), c2w[rand_idx], step=step)
        selected_points = model.selected_points

        print("Eval step:", step, "train_loss:", train_losses[-1], "eval_loss:", eval_losses[-1], "eval_psnr:", eval_psnrs[-1])

        log_dir = os.path.join(args.save_dir, args.index)
        os.makedirs(log_dir, exist_ok=True)
        if args.eval.save_fig:
            os.makedirs(os.path.join(log_dir, "train_main_plots"), exist_ok=True)
            os.makedirs(os.path.join(log_dir, "train_pcd_plots"), exist_ok=True)

            coord_scale = args.dataset.coord_scale
            pt_plot_scale = 1.0 * coord_scale  

            # calculate depth, weighted sum the distances from top K points to image plane
            od = -rayo[rand_idx]
            D = torch.sum(od * rayo[rand_idx])
            dists = torch.abs(torch.sum(selected_points.to(od.device) * od, -1) - D) / torch.norm(od)
            if model.bkg_feats is not None:
                dists = torch.cat([dists, torch.ones(1, H, W, model.bkg_feats.shape[0]).to(dists.device) * 0], dim=-1)
            cur_depth = (torch.sum(attn.squeeze(-1).to(od.device) * dists, dim=-1)).detach().cpu()

            train_tgt_rgb = train_tgt[rand_idx].squeeze().cpu().numpy().astype(np.float32)
            train_pred_rgb = train_out[rand_idx].squeeze().cpu().numpy().astype(np.float32)
            test_tgt_rgb = train_tgt[0].squeeze().cpu().numpy().astype(np.float32)
            test_pred_rgb = train_out[0].squeeze().cpu().numpy().astype(np.float32)
            points_np = model.points.detach().cpu().numpy()
            depth = cur_depth.squeeze().numpy().astype(np.float32)
            points_influ_scores_np = None
            if model.points_influ_scores is not None:
                points_influ_scores_np = model.points_influ_scores.squeeze().detach().cpu().numpy()

            # main plot
            main_plot = get_training_main_plot(args.index, steps, train_tgt_rgb, train_pred_rgb, test_tgt_rgb, test_pred_rgb, train_losses, 
                                            eval_losses, points_np, pt_plot_scale, depth, pt_lrs, tx_lrs, eval_psnrs, points_influ_scores_np)
            save_name = os.path.join(log_dir, "train_main_plots", "%s_iter_%d.png" % (args.index, step))
            main_plot.save(save_name)

        # point cloud plot
        ro = rayo[rand_idx].squeeze().detach().cpu().numpy()
        rd = rayd[rand_idx].squeeze().detach().cpu().numpy()
        
        pcd_plot = get_training_pcd_plot(args.index, steps[-1], ro, rd, points_np, args.dataset.coord_scale, pt_plot_scale, points_influ_scores_np)
        save_name = os.path.join(log_dir, "train_pcd_plots", "%s_iter_%d.png" % (args.index, step))
        pcd_plot.save(save_name)

    model.save(step, log_dir)
    if step % 50000 == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, "model_%d.pth" % step))

    torch.save(torch.tensor(train_losses), os.path.join(log_dir, "train_losses.pth"))
    torch.save(torch.tensor(eval_losses), os.path.join(log_dir, "eval_losses.pth"))
    torch.save(torch.tensor(eval_psnrs), os.path.join(log_dir, "eval_psnrs.pth"))

    return 0


if __name__ == "__main__":

    args = parse_args()
    with open(args.opt, 'r') as f:
        config = yaml.safe_load(f)
    eval_config = copy.deepcopy(config)
    eval_config['dataset'].update(eval_config['eval']['dataset'])
    eval_config = DictAsMember(eval_config)
    config = DictAsMember(config)

    log_dir = os.path.join(config.save_dir, config.index)
    os.makedirs(log_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(log_dir, 'train.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(log_dir, 'train_error.log'), sys.stderr)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

    find_all_python_files_and_zip(".", os.path.join(log_dir, "code.zip"))

    setup_seed(config.seed)

    papr_model, papr_losses = create_papr(config, eval_config, args.resume)

    ###############################

    seed_everything(0)

    zero123_cfg = optional_load_config("chair_config.yml")
    zero123_cfg = SJC_0(**zero123_cfg).dict()

    #cfg = argparse_cfg_template(cfg)  # cmdline takes priority
    mod = SJC_0(**zero123_cfg)

    write_full_config(mod)

    zero123_cfg, zero123_model, vox, poser = mod.ready()

    sjc_3d(**zero123_cfg, poser=poser, model=zero123_model, vox=vox, 
               papr_model=papr_model, papr_losses=papr_losses, papr_args=config)



    
    