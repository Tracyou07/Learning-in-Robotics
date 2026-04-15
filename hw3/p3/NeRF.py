from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import imageio
import cv2

def load_colmap_data():
    r"""
    After using colmap2nerf.py to convert the colmap intrinsics and extrinsics,
    read in the transform_colmap.json file

    Expected Returns:
      An array of resized imgs, normalized to [0, 1]
      An array of poses, essentially the transform matrix
      Camera parameters: H, W, focal length

    NOTES:
      We recommend you resize the original images from 800x800 to lower resolution,
      i.e. 200x200 so it's easier for training. Change camera parameters accordingly
    """
    ################### YOUR CODE START ###################
    with open('data/transforms_colmap.json', 'r') as f:
        data = json.load(f)

    camera_angle_x = data['camera_angle_x']
    frames = data['frames']

    # Load images and poses
    images = []
    poses = []
    for frame in frames:
        file_path = frame['file_path']
        # file_path is like "./train/r_0", add .png and prepend data/images/
        img_path = os.path.join('data', 'images', file_path.lstrip('./') + '.png')
        img = imageio.imread(img_path)
        # If RGBA, take only RGB
        img = img[..., :3]
        # Resize from 800x800 to 200x200
        img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        images.append(img)

        pose = np.array(frame['transform_matrix'], dtype=np.float32)
        poses.append(pose)

    images = np.stack(images, axis=0)  # (N, 200, 200, 3)
    poses = np.stack(poses, axis=0)    # (N, 4, 4)

    # Compute focal length
    # Original image width is 800
    orig_W = 800
    focal = 0.5 * orig_W / np.tan(0.5 * camera_angle_x)
    # Scale focal length for resized images
    focal = focal * (100.0 / orig_W)

    H, W = 100, 100
    hwf = [H, W, focal]

    return images, poses, hwf
    ################### YOUR CODE END ###################


def get_rays(height, width, focal_length, trans_matrix):
    r"""Compute rays passing through each pixels

    Expected Returns:
      ray_origins: A tensor of shape (H, W, 3) denoting the centers of each ray.
      ray_directions: A tensor of shape (H, W, 3) denoting the direction of each
        ray. ray_directions[i][j] denotes the direction (x, y, z) of the ray
        passing through the pixel at row index `i` and column index `j`.
    """
    ################### YOUR CODE START ###################
    # Create meshgrid for pixel coordinates
    i = torch.arange(height, dtype=torch.float32, device=trans_matrix.device)
    j = torch.arange(width, dtype=torch.float32, device=trans_matrix.device)
    # ii[row, col] = row index, jj[row, col] = col index
    ii, jj = torch.meshgrid(i, j, indexing='ij')

    # Pixel to camera coordinates
    # Convention: +X right, +Y up, +Z back (looking along -Z)
    directions_cam = torch.stack([
        (jj - width * 0.5) / focal_length,
        -(ii - height * 0.5) / focal_length,
        -torch.ones_like(ii)
    ], dim=-1)  # (H, W, 3)

    # Rotate directions from camera to world frame
    rotation = trans_matrix[:3, :3]  # (3, 3)
    ray_directions = torch.sum(directions_cam[..., None, :] * rotation, dim=-1)  # (H, W, 3)

    # Normalize ray directions
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    # Ray origins are the camera position in world coordinates
    ray_origins = trans_matrix[:3, 3].expand_as(ray_directions)  # (H, W, 3)

    return ray_origins, ray_directions
    ################### YOUR CODE END ###################


def sample_points_from_rays(ray_origins, ray_directions, near, far, num_samples):
    r"""Compute a set of 3D points given the bundle of rays

    Expected Returns:
      sampled_points: axis of the sampled points along each ray, shape (H, W, num_samples, 3)
      depth_values: sampled depth values along each ray, shape (H, W, num_samples)
    """
    ################### YOUR CODE START ###################
    device = ray_origins.device
    H, W, _ = ray_origins.shape

    # Uniformly sample depth values between near and far
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
    depth_values = near + t_vals * (far - near)  # (num_samples,)

    # Add random perturbation for training (stratified sampling)
    # Each bin has width (far - near) / num_samples
    bin_width = (far - near) / num_samples
    noise = torch.rand(H, W, num_samples, device=device) * bin_width
    depth_values = depth_values.unsqueeze(0).unsqueeze(0) + noise  # (H, W, num_samples)

    # Compute 3D sample points: origin + depth * direction
    # ray_origins: (H, W, 3), ray_directions: (H, W, 3), depth_values: (H, W, num_samples)
    sampled_points = ray_origins[..., None, :] + depth_values[..., :, None] * ray_directions[..., None, :]
    # sampled_points: (H, W, num_samples, 3)

    return sampled_points, depth_values
    ################### YOUR CODE END ###################


def positional_encoding(x, num_frequencies=10, include_input=True):
    r"""Apply positional encoding to the input. (Section 5.1 of original paper)
    We use positional encoding to map continuous input coordinates into a
    higher dimensional space to enable our MLP to more easily approximate a
    higher frequency function.

    Expected Returns:
      pos_out: positional encoding of the input tensor.
               (H*W*num_samples, (include_input + 2*freq) * 3)
    """
    ################### YOUR CODE START ###################
    # x: (N, 3)
    encodings = []
    if include_input:
        encodings.append(x)

    for k in range(num_frequencies):
        freq = 2.0 ** k
        encodings.append(torch.sin(freq * x))
        encodings.append(torch.cos(freq * x))

    pos_out = torch.cat(encodings, dim=-1)
    return pos_out
    ################### YOUR CODE END ###################


def volume_rendering(
    radiance_field: torch.Tensor,
    ray_origins: torch.Tensor,
    depth_values: torch.Tensor
) -> torch.Tensor:
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    bundle, and the sampled depth values along them.

    Args:
      radiance_field: at each query location (X, Y, Z), our model predict
        RGB color and a volume density (sigma), shape (H, W, num_samples, 4)
      ray_origins: origin of each ray, shape (H, W, 3)
      depth_values: sampled depth values along each ray, shape (H, W, num_samples)

    Expected Returns:
      rgb_map: rendered RGB image, shape (H, W, 3)
    """
    ################### YOUR CODE START ###################
    # Apply activation functions
    sigma = F.relu(radiance_field[..., 3])    # (H, W, num_samples)
    rgb = torch.sigmoid(radiance_field[..., :3])  # (H, W, num_samples, 3)

    # Compute distances between adjacent samples
    dists = depth_values[..., 1:] - depth_values[..., :-1]  # (H, W, num_samples-1)
    # Append a large distance for the last sample
    dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], dim=-1)
    # dists: (H, W, num_samples)

    # Compute alpha (opacity) for each sample
    alpha = 1.0 - torch.exp(-sigma * dists)  # (H, W, num_samples)

    # Compute transmittance T_i = product of (1 - alpha_j) for j < i
    # T_i = cumprod(1 - alpha) shifted by one (T_0 = 1)
    ones = torch.ones_like(alpha[..., :1])
    transmittance = torch.cumprod(
        torch.cat([ones, 1.0 - alpha + 1e-10], dim=-1), dim=-1
    )[..., :-1]  # (H, W, num_samples)

    # Compute weights
    weights = transmittance * alpha  # (H, W, num_samples)

    # Compute final color
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # (H, W, 3)

    return rgb_map
    ################### YOUR CODE END ###################


class TinyNeRF(torch.nn.Module):
    def __init__(self, pos_dim, fc_dim=128):
      r"""Initialize a tiny nerf network, which composed of linear layers and
      ReLU activation. More specifically: linear - relu - linear - relu - linear
      - relu -linear. The module is intentionally made small so that we could
      achieve reasonable training time

      Args:
        pos_dim: dimension of the positional encoding output
        fc_dim: dimension of the fully connected layer
      """
      super().__init__()

      self.nerf = nn.Sequential(
                    nn.Linear(pos_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, 4)
                  )

    def forward(self, x):
      r"""Output volume density and RGB color (4 dimensions), given a set of
      positional encoded points sampled from the rays
      """
      x = self.nerf(x)
      return x


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def nerf_step_forward(height, width, focal_length, trans_matrix,
                            near_point, far_point, num_depth_samples_per_ray,
                            get_minibatches_function, model):
    r"""Perform one iteration of training, which take information of one of the
    training images, and try to predict its rgb values

    Args:
      height: height of the image
      width: width of the image
      focal_length: focal length of the camera
      trans_matrix: transformation matrix, which is also the camera pose
      near_point: threshhold of nearest point
      far_point: threshold of farthest point
      num_depth_samples_per_ray: number of sampled depth from each rays in the ray bundle
      get_minibatches_function: function to cut the ray bundles into several chunks
        to avoid out-of-memory issue

    Expected Returns:
      rgb_predicted: predicted rgb values of the training image
    """
    ################### YOUR CODE START ###################
    # Get the "bundle" of rays through all image pixels
    ray_origins, ray_directions = get_rays(height, width, focal_length, trans_matrix)

    # Sample points along each ray
    sampled_points, depth_values = sample_points_from_rays(
        ray_origins, ray_directions, near_point, far_point, num_depth_samples_per_ray)

    # Positional encoding, shape of return [H*W*num_samples, (include_input + 2*freq) * 3]
    flattened_points = sampled_points.reshape(-1, 3)
    positional_encoded_points = positional_encoding(flattened_points, num_frequencies=10, include_input=True)

    ################### YOUR CODE END ###################

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(positional_encoded_points, chunksize=16384)
    predictions = []
    for batch in batches:
      predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0) # (H*W*num_samples, 4)

    # "Unflatten" the radiance field.
    unflattened_shape = [height, width, num_depth_samples_per_ray, 4] # (H, W, num_samples, 4)
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape) # (H, W, num_samples, 4)

    ################### YOUR CODE START ###################
    # Perform differentiable volume rendering to re-synthesize the RGB image. # (H, W, 3)
    rgb_predicted = volume_rendering(radiance_field, ray_origins, depth_values)
    return rgb_predicted
    ################### YOUR CODE END ###################


def train(images, poses, hwf, near_point,
          far_point, num_depth_samples_per_ray,
          num_iters, model, DEVICE="cuda"):
    r"""Training a tiny nerf model

    Args:
      images: all the images extracted from dataset (including train, val, test)
      poses: poses of the camera, which are used as transformation matrix
      hwf: [height, width, focal_length]
      near_point: threshhold of nearest point
      far_point: threshold of farthest point
      num_depth_samples_per_ray: number of sampled depth from each rays in the ray bundle
      num_iters: number of training iterations
      model: predefined tiny NeRF model
    """
    H, W, focal_length = hwf
    H = int(H)
    W = int(W)
    n_train = images.shape[0]

    # Optimizer parameters
    lr = 5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    losses = []
    for it in tqdm(range(num_iters)):
      # Randomly pick a training image as the target, get rgb value and camera pose
      train_idx = np.random.randint(n_train)
      train_img_rgb = images[train_idx, ..., :3]
      train_pose = poses[train_idx]

      # Run one iteration of TinyNeRF and get the rendered RGB image.
      rgb_predicted = nerf_step_forward(H, W, focal_length,
                                              train_pose, near_point,
                                              far_point, num_depth_samples_per_ray,
                                              get_minibatches, model)

      # Compute mean-squared error between the predicted and target images
      loss = torch.nn.functional.mse_loss(rgb_predicted, train_img_rgb)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      losses.append(loss.item())

    print('Finish training')
    return losses


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load data
    images, poses, hwf = load_colmap_data()
    H, W, focal = hwf
    print(f"Loaded {images.shape[0]} images, H={H}, W={W}, focal={focal:.2f}")

    # Convert to torch tensors
    images = torch.tensor(images, dtype=torch.float32).to(DEVICE)
    poses = torch.tensor(poses, dtype=torch.float32).to(DEVICE)

    # NeRF parameters
    near = 2.0
    far = 6.0
    num_samples = 32
    num_freq = 10
    include_input = True
    pos_dim = (1 + 2 * num_freq) * 3  # 63
    fc_dim = 32
    num_iters = 1000

    # Create model
    model = TinyNeRF(pos_dim=pos_dim, fc_dim=fc_dim).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    losses = train(images, poses, hwf, near, far, num_samples, num_iters, model, DEVICE)

    # Save training loss plot (raw + smoothed)
    import numpy as _np
    losses_arr = _np.asarray(losses)
    # Moving-average smoothing (window 25)
    win = 25
    if len(losses_arr) >= win:
        smooth = _np.convolve(losses_arr, _np.ones(win) / win, mode='valid')
        smooth_x = _np.arange(win - 1, len(losses_arr))
    else:
        smooth, smooth_x = losses_arr, _np.arange(len(losses_arr))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(losses_arr, color='#9ecae1', alpha=0.6, label='per-iter MSE')
    axes[0].plot(smooth_x, smooth, color='#08519c', linewidth=2.0,
                 label=f'moving avg (w={win})')
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss (linear scale)')
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].semilogy(losses_arr, color='#fdae6b', alpha=0.6, label='per-iter MSE')
    axes[1].semilogy(smooth_x, smooth, color='#a63603', linewidth=2.0,
                     label=f'moving avg (w={win})')
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('MSE Loss (log)')
    axes[1].set_title(f'Training Loss (log scale) — final avg {smooth[-1]:.4f}')
    axes[1].grid(True, alpha=0.3, which='both'); axes[1].legend()

    plt.suptitle(f'TinyNeRF Training: {len(losses_arr)} iterations, '
                 f'$N_{{sample}}={num_samples}$, $h_{{dim}}={fc_dim}$', fontsize=13)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training_loss.png (final smoothed loss = {smooth[-1]:.4f})")

    # Render training images comparison (6 images)
    model.eval()
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    test_indices = np.linspace(0, images.shape[0] - 1, 6, dtype=int)
    with torch.no_grad():
        for idx, ti in enumerate(test_indices):
            # Ground truth
            axes[0, idx].imshow(images[ti].cpu().numpy())
            axes[0, idx].set_title(f'GT #{ti}')
            axes[0, idx].axis('off')

            # Predicted
            rgb_pred = nerf_step_forward(
                int(H), int(W), focal, poses[ti],
                near, far, num_samples, get_minibatches, model)
            axes[1, idx].imshow(rgb_pred.clamp(0, 1).cpu().numpy())
            axes[1, idx].set_title(f'Pred #{ti}')
            axes[1, idx].axis('off')

    plt.suptitle('Training Images: Ground Truth (top) vs Predicted (bottom)')
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved training_comparison.png")

    # Render novel viewpoints by interpolating between training poses
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    with torch.no_grad():
        for idx in range(5):
            # Interpolate between two training poses
            t = idx / 4.0
            pose_a = poses[0]
            pose_b = poses[len(poses) // 4]
            # Simple linear interpolation of pose matrices
            novel_pose = (1 - t) * pose_a + t * pose_b
            rgb_pred = nerf_step_forward(
                int(H), int(W), focal, novel_pose,
                near, far, num_samples, get_minibatches, model)
            axes[idx].imshow(rgb_pred.clamp(0, 1).cpu().numpy())
            axes[idx].set_title(f'Novel View {idx}')
            axes[idx].axis('off')

    plt.suptitle('Novel Viewpoints')
    plt.tight_layout()
    plt.savefig('novel_views.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved novel_views.png")

    print("Done!")
