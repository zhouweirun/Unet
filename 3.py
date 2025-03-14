from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
def add_noise(image, sigma=50):
    """
    Adds Gaussian noise to an image.
    :param image: Input image, numpy array, dtype=uint8, range=[0, 255]
    :param sigma: Standard deviation of the Gaussian noise, default is 50
    :return: Image with added noise, float numpy array, range=[0, 255]
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(0, sigma / 255, image.shape)
    gauss_noise = image + noise
    return np.clip(gauss_noise * 255, 0, 255).astype(np.uint8)

def crop_image(image, s=8):
    h, w, c = image.shape
    image = image[:h - h % s, :w - w % s, :]
    return image
# 手写的SSIM函数
def gaussian_kernel(channel, kernel_size=11, sigma=1.5):
    """生成高斯核"""
    x = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    y = torch.exp(-0.5 * (x / sigma) ** 2)
    y /= y.sum()

    kernel = y.view(1, 1, -1) @ y.view(1, -1, 1)
    kernel = kernel.repeat(channel, 1, 1, 1)

    return kernel


def ssim(img1, img2, data_range=1.0, kernel_size=11, sigma=1.5, K=(0.01, 0.03)):
    """
    计算两张图片之间的SSIM值。
    :param img1: 输入张量1 (B, C, H, W)
    :param img2: 输入张量2 (B, C, H, W)
    :param data_range: 图像数据范围，默认为1.0（对于归一化后的图像）
    :param kernel_size: 高斯核大小
    :param sigma: 高斯核标准差
    :param K: 常数K1, K2
    :return: SSIM值
    """
    device = img1.device

    # 确保输入张量在同一个设备上
    assert img1.shape == img2.shape, "Input images must have the same dimensions."

    # 创建高斯核
    channel = img1.shape[1]
    window = gaussian_kernel(channel, kernel_size, sigma).to(device)

    # 归一化
    mu1 = F.conv2d(img1, window, padding=kernel_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=kernel_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=kernel_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=kernel_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=kernel_size // 2, groups=channel) - mu1_mu2

    K1, K2 = K
    L = data_range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    v1 = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)  # 亮度对比
    v2 = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # 结构对比
    ssim_map = v1 * v2

    return ssim_map.mean()

scaler = GradScaler()


import torch
from torch.utils.data import Dataset
import imageio
import numpy as np
from torchvision import transforms

class DIV2KDataset(Dataset):
    def __init__(self, image_dir, noise_level=50, transform=None):
        self.image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.transform = transform
        self.noise_level = noise_level

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 使用imageio读取图像
        image = imageio.imread(img_path)

        # 裁剪图像，使其宽高为8的倍数
        image = crop_image(image)

        # 添加噪声前先转换为float类型并归一化到[0,1]范围
        noisy_image_np = add_noise(image, sigma=self.noise_level)

        # 将NumPy数组转换回PIL图像
        noisy_image_pil = Image.fromarray(noisy_image_np.astype(np.uint8))
        clean_image_pil = Image.fromarray(image)

        # 如果有transform，则应用变换（例如ToTensor）
        if self.transform:
            noisy_image = self.transform(noisy_image_pil)
            clean_image = self.transform(clean_image_pil)

        return noisy_image, clean_image

    def __len__(self):
        return len(self.image_paths)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        return torch.sigmoid(self.out_conv(dec1))  # 添加sigmoid激活函数

from torch.optim import Adam
def save_checkpoint(model, optimizer, epoch, filename):
    """
    保存模型和优化器的状态字典。
    :param model: 要保存的模型实例
    :param optimizer: 使用的优化器实例
    :param epoch: 当前epoch数
    :param filename: 文件名用于保存模型和优化器状态
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filepath, device):
    """
    加载模型和优化器的状态字典以继续训练。
    :param model: 模型实例
    :param optimizer: 优化器实例
    :param filepath: 文件路径
    :param device: 设备('cuda'或'cpu')
    :return: 加载了权重的模型实例、优化器实例以及最近的epoch数
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def custom_loss(output, target):
    mse_loss = F.mse_loss(output, target)

    # 确保output和target的值在[0, 1]范围内
    output_normalized = (output - output.min()) / (output.max() - output.min())
    target_normalized = (target - target.min()) / (target.max() - target.min())

    # 使用手写的ssim函数
    ssim_loss = 1 - ssim(output_normalized, target_normalized, data_range=1.0)
    return mse_loss + 0.1 * ssim_loss
def load_model(model_path, device):
    """
    加载预训练的UNet模型。
    :param model_path: 模型权重文件路径
    :param device: 设备（'cuda' 或 'cpu'）
    :return: 加载了权重的模型实例
    """
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
def compute_psnr(model, dataloader, device):
    """
    计算验证集上的平均PSNR。
    :param model: 模型实例
    :param dataloader: 数据加载器
    :param device: 设备('cuda' 或 'cpu')
    :return: 平均PSNR值
    """
    model.eval()
    psnr_sum = 0
    count = 0
    with torch.no_grad():
        for noisy_imgs, clean_imgs in dataloader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            for i in range(outputs.shape[0]):
                output_np = (outputs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                clean_img_np = (clean_imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                psnr_sum += psnr(clean_img_np, output_np)
                count += 1
    return psnr_sum / count

if __name__ == "__main__":
    # 参数设置
    batch_size = 4
    epochs = 80
    learning_rate = 0.0001
    psnr_threshold = 30  # 设置期望的PSNR阈值

    # 数据集路径
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 使用期望的高度和宽度
        transforms.ToTensor(),
    ])

    image_dir = 'DIV2K_valid_HR'
    dataset = DIV2KDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    action = input("请输入'action': 'train' 或 'load' 来指定要执行的操作: ")
    if action.lower() == 'train':
        model = UNet().to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        start_epoch = 0
    elif action.lower() == 'load':
        model_path = input("请输入模型文件路径: ")

        model = UNet().to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, model_path, device)
        print(f"Loaded model from epoch {start_epoch}")
    else:
        print("未知的操作类型，请输入'train'或'load'")
        exit()

    scaler = GradScaler()
    best_psnr = 0
    patience = 10  # 如果连续patience个epoch没有改善，则提前停止
    no_improve = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for i, (noisy_imgs, clean_imgs) in enumerate(dataloader):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(noisy_imgs)
                loss = custom_loss(outputs, clean_imgs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_psnr = compute_psnr(model, dataloader, device)
        print(f"Epoch [{epoch+1}/{epochs}], Average PSNR: {avg_psnr:.4f} dB")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            save_checkpoint(model, optimizer, epoch, '../NTIRE2025_Dn50_challenge-main/model_zoo/team34_Unet.pth')
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience or avg_psnr >= psnr_threshold:
            print(f"Early stopping at epoch {epoch + 1}, Best PSNR: {best_psnr:.4f} dB")
            break

    print("Training finished.")