import os
import torch
import argparse
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from dit import DiTModel
from flowmatching import FlowMatching
import tqdm

def main():
    """
    Warning: This training script has been modified to improve stability and performance.
    Please do not change the parameters below, as they have been carefully selected.
    If the number of DiT block is increased, in the MNIST dataset, will not increase performance
    significantly due to the small image size and dataset complexity. 
    """
    parser = argparse.ArgumentParser()
    # Increased epochs and model capacity, reduced batch size and learning rate for stability
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--lr_adjust_epoch', type=int, default=20, help='number of epochs between learning rate adjustments')
    parser.add_argument('--print_interval', type=int, default=100, help='steps between printing training status')
    parser.add_argument('--save_interval', type=int, default=5, help='epochs between saving model checkpoints')
    # Increase model capacity: wider and deeper transformer
    parser.add_argument('--base_channels', type=int, default=64, help='base channels for DiT model')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers in DiT model')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads in DiT model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir='logs')
    dataset = MNIST(root='data', train=True, download=True, transform = Compose([ToTensor(),Normalize((0.5,), (0.5,))]))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    model = DiTModel(in_channels=1, image_size=28, hidden_size=args.base_channels, patch_size=2, num_layers=args.num_layers, num_heads=args.num_heads).to(device)   
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = StepLR(optimizer, step_size=args.lr_adjust_epoch, gamma=args.gamma)
    flow_matching = FlowMatching(model)
 
    with tqdm.trange(0, args.epochs) as t_e:
        for epoch in t_e:
            for batch, data in enumerate(dataloader):
                x_1, y = data
                x_1 = x_1.to(device)
                y = y.to(device)
                # get_train_tuple now returns y_out for conditioning
                z0 = torch.randn_like(x_1)
                z_t, t, target, y_out = flow_matching.get_train_tuple(z1=x_1, z0=z0, repeat=1, y=y)
                optimizer.zero_grad()
                # pass (t, y_out) as conditioning to model
                v_pred = flow_matching.model(z_t, (t, y_out))
                loss = torch.nn.functional.mse_loss(v_pred, target)
                loss.backward()
                optimizer.step()
                if (batch + 1) % args.print_interval == 0:
                    t_e.set_description(f'Epoch {epoch+1}/{args.epochs}, Batch {batch+1}/{len(dataloader)}, Loss: {loss.item():.4f}')
                    writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch)
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Model checkpoint saved at {checkpoint_path}')
            scheduler.step()
    
if __name__ == '__main__':
    main()
