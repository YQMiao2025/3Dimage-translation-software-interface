import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
%matplotlib
inline

class BratsDataset(Dataset):
    def __init__(self, root_dirs, is_train=True, train_ratio=0.8):
        self.root_dirs = root_dirs
        self.all_subfolders = []

            subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

            for folder in subfolders:
                folder_path = os.path.join(root_dir, folder)
                files = os.listdir(folder_path)
                has_t1 = any('t1.nii.gz' in f and not 't1ce' in f for f in files)
                has_t2 = any('t2.nii.gz' in f for f in files)

                if has_t1 and has_t2:
                    self.all_subfolders.append(folder_path)

        train_folders, val_folders = train_test_split(
            self.all_subfolders, test_size=1 - train_ratio, random_state=42
        )

        self.folders = train_folders if is_train else val_folders
        print(f"{'train' if is_train else 'valid'}has {len(self.folders)} samples")

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder_path = self.folders[idx]
        files = os.listdir(folder_path)

        t1_file = next(f for f in files if 't1.nii.gz' in f and not 't1ce' in f)
        t2_file = next(f for f in files if 't2.nii.gz' in f)
        t1_path = os.path.join(folder_path, t1_file)
        t2_path = os.path.join(folder_path, t2_file)
        t1_img = nib.load(t1_path).get_fdata()
        t2_img = nib.load(t2_path).get_fdata()

        # Normalize to the [0, 1] range
        t1_img = (t1_img - np.min(t1_img)) / (np.max(t1_img) - np.min(t1_img) + 1e-8)
        t2_img = (t2_img - np.min(t2_img)) / (np.max(t2_img) - np.min(t2_img) + 1e-8)

        t1_img = torch.tensor(t1_img, dtype=torch.float32).unsqueeze(0)
        t2_img = torch.tensor(t2_img, dtype=torch.float32).unsqueeze(0)

        return t1_img, t2_img

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class UNet3DGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[8, 16, 32, 64, 128]):
        super(UNet3DGenerator, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(ResidualBlock3D(in_channels, feature, stride=1))
            in_channels = feature

        self.bottleneck = nn.Sequential(
            ResidualBlock3D(features[-1], features[-1] * 2, stride=1),
            ResidualBlock3D(features[-1] * 2, features[-1] * 2, stride=1)
        )

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(ResidualBlock3D(feature * 2, feature, stride=1))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            if all([s >= 2 for s in x.shape[2:]]):
                x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:],
                    mode='trilinear', align_corners=True
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.sigmoid(self.final_conv(x))


class Discriminator3D(nn.Module):
    def __init__(self, in_channels=2, features=[16, 32, 64, 128]):
        super(Discriminator3D, self).__init__()
        layers = []

        layers.append(
            nn.Sequential(
                nn.Conv3d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, feature, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(feature),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = feature

        layers.append(nn.Conv3d(in_channels, 1, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.model(x)
        x = self.avg_pool(x).view(x.size(0), 1)
        return x


def show_validation_samples(generator, val_loader, device, epoch, current_loss_g, current_loss_d):
    generator.eval()
    with torch.no_grad():
        t1_images, t2_images = next(iter(val_loader))
        t1_images = t1_images.to(device)

        fake_t2 = generator(t1_images)

        t1_np = t1_images.cpu().numpy()[0, 0]
        t2_real_np = t2_images.numpy()[0, 0]
        t2_fake_np = fake_t2.cpu().numpy()[0, 0]
        t2_fake_np = fake_t2.cpu().numpy()[0, 0]

        mid_slice = t1_np.shape[0] // 2

        plt.figure(figsize=(18, 6))

        plt.subplot(131)
        plt.imshow(t1_np[mid_slice], cmap='gray')
        plt.title('T1 Input')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(t2_real_np[mid_slice], cmap='gray')
        plt.title('Real T2')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(t2_fake_np[mid_slice], cmap='gray')
        plt.title(f'Generated T2 (Epoch {epoch + 1})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print(f"Epoch {epoch + 1} loss:")
        print(f"Gen loss: {current_loss_g:.4f}")
        print(f"Dis loss: {current_loss_d:.4f}")
        print("----------------------------------------")

    generator.train()


def train_loop(generator, discriminator, train_loader, val_loader, opt_g, opt_d, device, total_epochs=200,
               lambda_l1=100):
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    scaler_g = GradScaler()
    scaler_d = GradScaler()
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    model_dir = ''
    os.makedirs(model_dir, exist_ok=True)

    all_losses = {
        'generator': [],
        'discriminator': []
    }

    for epoch in range(total_epochs):
        if epoch == 100:
            for param_group in opt_g.param_groups:
                param_group['lr'] = 1e-4
            for param_group in opt_d.param_groups:
                param_group['lr'] = 5e-5

        loop = tqdm(train_loader, position=0)
        running_loss_g = 0.0
        running_loss_d = 0.0

        for i, (t1_images, t2_images) in enumerate(loop):
            t1_images = t1_images.to(device)
            t2_images = t2_images.to(device)

            with autocast(device_type=device_type):
                fake_t2 = generator(t1_images)
                disc_real = discriminator(t1_images, t2_images).reshape(-1)
                loss_d_real = criterion_gan(disc_real, torch.ones_like(disc_real))
                disc_fake = discriminator(t1_images, fake_t2.detach()).reshape(-1)
                loss_d_fake = criterion_gan(disc_fake, torch.zeros_like(disc_fake))
                loss_d = (loss_d_real + loss_d_fake) / 2

            opt_d.zero_grad()
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            with autocast(device_type=device_type):
                disc_fake = discriminator(t1_images, fake_t2).reshape(-1)
                loss_g_gan = criterion_gan(disc_fake, torch.ones_like(disc_fake))
                loss_g_l1 = criterion_l1(fake_t2, t2_images) * lambda_l1
                loss_g = loss_g_gan + loss_g_l1

            opt_g.zero_grad()
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            running_loss_g += loss_g.item()
            running_loss_d += loss_d.item()
            loop.set_postfix(Epoch=epoch, Loss_G=loss_g.item(), Loss_D=loss_d.item(),
                             LR=opt_g.param_groups[0]['lr'])

        epoch_loss_g = running_loss_g / len(train_loader)
        epoch_loss_d = running_loss_d / len(train_loader)
        all_losses['generator'].append(epoch_loss_g)
        all_losses['discriminator'].append(epoch_loss_d)

        if (epoch + 1) % 10 == 0:
            show_validation_samples(generator, val_loader, device, epoch, epoch_loss_g, epoch_loss_d)

        if (epoch + 1) % 10 == 0:
            model_path_g = os.path.join(model_dir, f'generator_epoch_{epoch + 1}.pth')
            torch.save(generator.state_dict(), model_path_g)
            print(f"Generator saved at: {model_path_g}")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}, Generator Loss: {epoch_loss_g:.4f}, Discriminator Loss: {epoch_loss_d:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_epochs + 1), all_losses['generator'], label='Generator Loss')
    plt.plot(range(1, total_epochs + 1), all_losses['discriminator'], label='Discriminator Loss')
    plt.axvline(x=100, color='r', linestyle='--', label='Learning Rate Change')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over num Epochs')
    plt.legend()
    plt.show()

    return generator, discriminator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_dirs = [
        "",
        "",
        ""
    ]

    try:
        train_dataset = BratsDataset(train_dirs, is_train=True)
        val_dataset = BratsDataset(train_dirs, is_train=False)
    except ValueError as e:
        print(f"data error: {e}")
        return

    batch_size = 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    generator = UNet3DGenerator(in_channels=1, out_channels=1, features=[8, 16, 32, 64, 128]).to(device)
    discriminator = Discriminator3D(in_channels=2, features=[16, 32, 64, 128]).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    trained_generator, _ = train_loop(
        generator, discriminator, train_loader, val_loader,
        optimizer_g, optimizer_d, device, total_epochs=, lambda_l1=
    )

    print("Completed!")


if __name__ == "__main__":
    main()