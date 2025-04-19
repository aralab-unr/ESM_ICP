import open3d as o3d
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add root to sys.path if inside examples/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR.endswith('examples'):
    sys.path.append(os.path.join(BASE_DIR, os.pardir))
    os.chdir(os.path.join(BASE_DIR, os.pardir))

from learning3d.models.pointnet import PointNet
from learning3d.models import DGCNN, DCP, PointNetLK, RPMNet
#from learning3d.models.transform import estimate_transform  # Needed for DeepGMR forward

class DeepGMRCustom(nn.Module):
    def __init__(self, feature_model=None, num_gaussians=64):
        super(DeepGMRCustom, self).__init__()
        self.feature_model = feature_model if feature_model else PointNet()
        self.num_gaussians = num_gaussians

    def forward(self, template_features, source_features, template_pts=None, source_pts=None):
        # Fallback to raw points if not provided
        if template_pts is None:
            template_pts = template_features.transpose(1, 2)  # [B, N, 3]
        if source_pts is None:
            source_pts = source_features.transpose(1, 2)

        # Make sure input to PointNet is [B, 3, N]
        template_features_t = template_features.transpose(1, 2)  # [B, 3, N]
        source_features_t = source_features.transpose(1, 2)      # [B, 3, N]

        self.template_gamma = F.softmax(self.feature_model(template_features_t), dim=2)
        self.source_gamma = F.softmax(self.feature_model(source_features_t), dim=2)

        self.template_pi, self.template_mu, self.template_sigma = gmm_params(self.template_gamma, template_pts)
        self.source_pi, self.source_mu, self.source_sigma = gmm_params(self.source_gamma, source_pts)

        transform = estimate_transform(self.source_mu, self.source_sigma, self.source_pi,
                                    self.template_mu, self.template_sigma, self.template_pi)

        aligned = transform(source_pts)  # [B x N x 3]
        return aligned

def gmm_params(gamma, pts):
    B, K, N = gamma.shape
    Npi = gamma.sum(dim=2) + 1e-8  # (B x K)
    pi = Npi / N

    gamma = gamma / (Npi.unsqueeze(2) + 1e-8)
    pts_expanded = pts.unsqueeze(1).repeat(1, K, 1, 1)      # (B x K x N x 3)
    gamma_expanded = gamma.unsqueeze(3)                     # (B x K x N x 1)
    mu = (gamma_expanded * pts_expanded).sum(dim=2)         # (B x K x 3)

    diff = pts_expanded - mu.unsqueeze(2)
    sigma = torch.matmul(
        diff.transpose(2, 3), gamma_expanded * diff
    ) / (Npi.view(B, K, 1, 1) + 1e-8)                        # (B x K x 3 x 3)

    return pi, mu, sigma

def estimate_transform(mu_x, sigma_x, pi_x, mu_y, sigma_y, pi_y):
    """
    Estimate transformation between two GMMs using weighted Procrustes.
    This version uses closed-form weighted SVD.
    """
    B, K, D = mu_x.shape
    device = mu_x.device

    # Weighted centroids
    cx = torch.sum(mu_x * pi_x.unsqueeze(-1), dim=1, keepdim=True)
    cy = torch.sum(mu_y * pi_y.unsqueeze(-1), dim=1, keepdim=True)

    # Centered points
    mu_x_c = mu_x - cx
    mu_y_c = mu_y - cy

    # Cross-covariance
    W = torch.matmul((pi_x.unsqueeze(-1) * mu_y_c).transpose(1, 2), mu_x_c)

    # SVD
    U, S, Vh = torch.linalg.svd(W)
    R = torch.matmul(U, Vh)

    # Ensure right-handed coordinate system
    det = torch.linalg.det(R)
    Vh[:, :, -1] *= torch.sign(det).unsqueeze(-1)
    R = torch.matmul(U, Vh)

    # Translation
    T = cy.squeeze(1) - torch.bmm(R, cx.transpose(1, 2)).squeeze(2)

    def transform(points):
        return torch.bmm(points, R.transpose(1, 2)) + T.unsqueeze(1)

    return transform

# def estimate_transform(mu_x, sigma_x, pi_x, mu_y, sigma_y, pi_y):
#   #  from learning3d.models.transform import estimate_transform
#     return estimate_transform(mu_x, sigma_x, pi_x, mu_y, sigma_y, pi_y)

def load_pcd(file, n_points=1024):
    pcd = o3d.io.read_point_cloud(file)
    if len(pcd.points) > n_points:
        pcd = pcd.random_down_sample(n_points / len(pcd.points))
    return np.asarray(pcd.points).astype(np.float32)


def random_transform(pc):
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.random.uniform(0, 10.0, 3))
    t = np.random.uniform(-0.5, 0.5, 3)
    transformed = (R @ pc.T).T + t
    return transformed, R, t


def visualize(source, target, aligned):
    def color_cloud(pts, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        return pcd

    o3d.visualization.draw_geometries([
        color_cloud(source, [1, 0, 0]),
        color_cloud(target, [0, 1, 0]),
        color_cloud(aligned, [0, 0, 1])
    ])


def save_aligned_pointcloud(aligned_np, filename='aligned_output.pcd'):
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_np)
    success = o3d.io.write_point_cloud(filename, aligned_pcd, write_ascii=True)
    if success:
        print(f"✅ Aligned point cloud saved to: {filename}")
    else:
        print(f"❌ Failed to save aligned point cloud.")


def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration Test')
    parser.add_argument('--pretrained', default='', type=str, help='Path to pretrained model')
    parser.add_argument('--device', default='cuda:0', type=str, help='CUDA device')
    parser.add_argument('--pcd', type=str, required=True, help='Path to input .pcd file')
    parser.add_argument('--transform', type=str, default=None, help='Path to .npy 4x4 transformation matrix')
    parser.add_argument('--model', default='dcp', type=str,
                        choices=['dcp', 'pointnetlk', 'deepgmr', 'rpmnet'],
                        help='Registration model to use')
    parser.add_argument('--save_path', type=str, default='aligned_output.pcd',
                        help='Where to save the aligned point cloud')
    return parser.parse_args()


def main():
    print("✅ Python script started successfully.")
    args = options()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    default_paths = {
    'dcp':        '/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/exp_dcp/models/best_model.t7',
    'pointnetlk': '/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/exp_pnlk/models/best_model.t7',
    'deepgmr':    '/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/exp_deepgmr/models/best_model.pth',
    'rpmnet':     '/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/exp_rpmnet/models/clean-trained.pth'
     }

    # Select pretrained path
    if args.pretrained:
        pretrained_path = args.pretrained
        print(f"✅ Using user-specified pretrained model: {pretrained_path}")
    else:
        pretrained_path = default_paths.get(args.model)
        if pretrained_path:
            print(f"🔄 Automatically selected pretrained model for '{args.model}': {pretrained_path}")
        else:
            raise ValueError(f"❌ No default pretrained path defined for model '{args.model}'")

    # Load model
    if args.model == 'dcp':
        feature_model = DGCNN(emb_dims=512)
        model = DCP(feature_model=feature_model, cycle=True).to(device)
    elif args.model == 'pointnetlk':
        model = PointNetLK().to(device)
    elif args.model == 'deepgmr':
        feature_model = PointNet()
        model = DeepGMRCustom(feature_model=feature_model).to(device)
    elif args.model == 'rpmnet':
        model = RPMNet().to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Load weights
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"✅ Loading pretrained weights from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device), strict=False)
    else:
        print(f"⚠️ Warning: pretrained model not found at {pretrained_path}")

    model.eval()

    # Load and prepare source point cloud
    source_np = load_pcd(args.pcd, n_points=1024)

    # Apply transformation from file if provided
    if args.transform and os.path.isfile(args.transform):
        print(f"✅ Applying transformation from {args.transform}")
        transform = np.load(args.transform)
        R_gt = transform[:3, :3]
        t_gt = transform[:3, 3]
        target_np = (R_gt @ source_np.T).T + t_gt
    else:
        print("⚠️ No transform provided. Using random transformation.")
        target_np, R_gt, t_gt = random_transform(source_np)

    # Convert to torch
    source = torch.tensor(source_np, dtype=torch.float32).unsqueeze(0).to(device)
    target = torch.tensor(target_np, dtype=torch.float32).unsqueeze(0).to(device)

    
    with torch.no_grad():
        if args.model in ['dcp', 'rpmnet', 'pointnetlk']:
            output = model(target, source)
            R_pred = output['est_R']
            t_pred = output['est_t']
            aligned = torch.bmm(source, R_pred.transpose(2, 1)) + t_pred.unsqueeze(1)
        elif args.model == 'deepgmr':
            aligned = model(target.transpose(1, 2), source.transpose(1, 2), target, source)
            R_pred, t_pred = None, None

    aligned_np = aligned.squeeze(0).cpu().numpy()
    save_aligned_pointcloud(aligned_np, args.save_path)

    print("\n📌 Ground Truth Rotation:\n", R_gt)
    print("📌 Ground Truth Translation:\n", t_gt)
    if R_pred is not None and t_pred is not None:
        print("📌 Predicted Rotation:\n", R_pred.squeeze(0).cpu().numpy())
        print("📌 Predicted Translation:\n", t_pred.squeeze(0).cpu().numpy())

        predicted_T = np.eye(4)
        predicted_T[:3, :3] = R_pred.squeeze(0).cpu().numpy()
        predicted_T[:3, 3] = t_pred.squeeze(0).cpu().numpy()
        np.savetxt("predicted_transform.txt", predicted_T)

    

   # visualize(source_np, target_np, aligned_np)


if __name__ == '__main__':
    main()
