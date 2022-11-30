import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm.autonotebook import tqdm


def calc_diversity(data_dir, num_samples=5):
    dir_list = os.listdir(data_dir)
    dir_list.sort()

    transform = transforms.Compose([transforms.ToTensor()])

    total = len(dir_list)
    # total=2000
    std = 0
    for i in tqdm(range(total), total=total, smoothing=0.01):
        imgs = []
        for j in range(num_samples):
            # img = Image.open(os.path.join(os.path.join(data_dir, dir_list[i], f'output_{str(j)}.png')))
            img = Image.open(os.path.join(os.path.join(data_dir, str(i), f'output_{str(j)}.png')))
            img = img.convert('RGB')
            img = transform(img)
            img = img * 255.
            imgs.append(img)

        img_mean = torch.zeros_like(imgs[0])
        for j in range(num_samples):
            img_mean = img_mean + imgs[j]
        img_mean = img_mean / num_samples

        img_var = torch.zeros_like(imgs[0])
        for j in range(num_samples):
            img_var = img_var + (imgs[j] - img_mean)**2
        img_var = img_var / num_samples
        img_std = torch.sqrt(img_var)
        std = std + torch.mean(img_std)
    std = std / total
    print(data_dir)
    print(f'diversity: {std}')



# calc_diversity('/home/x/Mine/project/paper_samples/latent-diffusion-main/results/CDE-edges2handbags/samples')
# calc_diversity('/home/x/Mine/project/paper_samples/latent-diffusion-main/results/CDE-faces2comics/samples')
# calc_diversity('/home/x/Mine/project/paper_samples/latent-diffusion-main/results/CDE-CelebAMaskHQ-f4/samples')
# calc_diversity('/home/x/Mine/project/paper_samples/latent-diffusion-main/results/CDE-edges2shoes/samples')
# calc_diversity('/home/x/Mine/project/paper_samples/latent-diffusion-main/results/faces2comics/samples')
# calc_diversity('/home/x/Mine/project/paper_samples/latent-diffusion-main/results/edges2shoes/samples')
# calc_diversity('/home/x/Mine/project/paper_samples/latent-diffusion-main/results/edges2handbags/samples')

# calc_diversity('/media/x/disk/BB_experiments/VQBB-checkpoints/edges2handbags/edges2handbags-ldm-f4-before+norm-128-concat/sample_to_calc/200/result')
calc_diversity('/home/x/Mine/project/BBDM/output/BBDM-before+norm-concat/CelebAMaskHQ-f4/sample_to_calc/200/without_diff')