import os
import random
import lpips
import torch
from tqdm.autonotebook import tqdm

loss_fn = lpips.LPIPS(net='alex', version='0.1').to(torch.device('cuda:0'))


@torch.no_grad()
def calc_LPIPS(data_dir, gt_dir, num_samples=5):
    dir_list = os.listdir(data_dir)
    dir_list.sort()

    total = len(dir_list)
    # total=2000
    total_lpips_distance = 0
    for i in tqdm(range(total), total=total, smoothing=0.01):
        gt_name = os.path.join(gt_dir, f'{str(i)}.png')
        gt_img = lpips.im2tensor(lpips.load_image(gt_name)).to(torch.device('cuda:0'))
        for j in range(num_samples):
            # img = Image.open(os.path.join(os.path.join(data_dir, dir_list[i], f'output_{str(j)}.png')))
            if num_samples == 1:
                # img_name = os.path.join(os.path.join(data_dir, f'{str(i)}_fake_B.png'))
                img_name = os.path.join(os.path.join(data_dir, f'{str(i)}.png'))
            else:
                img_name = os.path.join(os.path.join(data_dir, str(i), f'output_{str(j)}.png'))
            img_calc = lpips.im2tensor(lpips.load_image(img_name)).to(torch.device('cuda:0'))
            current_lpips_distance = loss_fn.forward(gt_img, img_calc)
            total_lpips_distance = total_lpips_distance + current_lpips_distance
    avg_lpips_distance = total_lpips_distance / (total * num_samples)
    print(data_dir)
    print(f'lpips_distance: {avg_lpips_distance}')


calc_LPIPS(data_dir='/home/x/Mine/project/BBDM/output/BBDM-before+norm-concat/CelebAMaskHQ-f4/sample_to_calc/200/without_diff',
           gt_dir='/home/x/Mine/project/BBDM/output/BBDM-before+norm-concat/CelebAMaskHQ-f4/sample_to_calc/ground_truth',
           num_samples=1)


@torch.no_grad()
def random_LPIPS(data_dir, gt_dir):
    dir_list = os.listdir(data_dir)
    dir_list.sort()

    total = len(dir_list)
    # total=2000
    total_lpips_distance = 0
    for i in tqdm(range(total), total=total, smoothing=0.01):
        gt_name = os.path.join(gt_dir, f'{str(i)}.png')
        gt_img = lpips.im2tensor(lpips.load_image(gt_name)).to(torch.device('cuda:0'))
        j = random.randint(0, 4)
        img_name = os.path.join(os.path.join(data_dir, str(i), f'output_{str(j)}.png'))
        img_calc = lpips.im2tensor(lpips.load_image(img_name)).to(torch.device('cuda:0'))
        current_lpips_distance = loss_fn.forward(gt_img, img_calc)
        total_lpips_distance = total_lpips_distance + current_lpips_distance
    avg_lpips_distance = total_lpips_distance / total
    return avg_lpips_distance

@torch.no_grad()
def find_max_min_LPIPS(data_dir, gt_dir):
    max_LPIPS = 0
    min_LPIPS = 10

    for i in range(100):
        avg_LPIPS = random_LPIPS(data_dir, gt_dir)
        if avg_LPIPS > max_LPIPS:
            max_LPIPS = avg_LPIPS
        if avg_LPIPS < min_LPIPS:
            min_LPIPS = avg_LPIPS
        if i % 20 == 0:
            print(f"{i} current_LPIPS = {avg_LPIPS}, max_LPIPS = {max_LPIPS}, min_LPIPS = {min_LPIPS}")
    print(data_dir)
    print(f'max_LPIPS = {max_LPIPS}, min_LPIPS = {min_LPIPS}')


# find_max_min_LPIPS(data_dir="/media/x/disk/BB_experiments/VQBB-checkpoints/edges2shoes/edges2shoes-f16-N/sample_to_calc/200/result",
#                    gt_dir='/media/x/disk/BB_experiments/evaluation_temp_dir/edges2shoes/shoes')

# find_max_min_LPIPS(data_dir="/home/x/Mine/project/paper_samples/latent-diffusion-main/results/ldm-edges2shoes/samples",
#                    gt_dir='/media/x/disk/BB_experiments/evaluation_temp_dir/edges2shoes/shoes')