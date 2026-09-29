import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree
import logging
import os
import json

# Try to import pyembree for fast raycasting
try:
    import pyembree
except ImportError:
    print("Warning: 'pyembree' not found. Raycasting will be slow. Install via 'pip install pyembree' or conda.")

logger = logging.getLogger(__name__)

target_volume = 900  
num_points_align = 1000
max_iterations = 100
cost_threshold = 0
num_points_chamfer = 30000

# ----------------- Culling Helpers -----------------

def load_cameras(json_path):
    """
    Loads camera poses from a Blender/NeRF style JSON file.
    Returns an array of 4x4 transformation matrices (c2w).
    """
    if not os.path.exists(json_path):
        logger.error(f"Camera JSON not found at: {json_path}")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    poses = []
    # Blender JSONs usually have a 'frames' list
    if 'frames' in data:
        for frame in data['frames']:
            # transform_matrix is usually c2w
            pose = np.array(frame['transform_matrix'])
            poses.append(pose)
    
    return np.array(poses)

def filter_hidden_geometry(mesh: trimesh.Trimesh, camera_poses, verbose=False):
    """
    Removes mesh triangles that are not visible from any of the provided camera poses.
    """
    if len(camera_poses) == 0:
        logger.warning("No cameras provided for culling. Returning original mesh.")
        return mesh

    logger.info(f"Culling mesh (Faces: {len(mesh.faces)}) using {len(camera_poses)} cameras...")
    
    # Ensure mesh has normals computed
    mesh.fix_normals()
    
    # Get centers and normals of all faces
    centers = mesh.triangles_center
    normals = mesh.face_normals
    
    # Initialize a set to store indices of faces that are visible
    visible_face_indices = set()
    
    # Initialize RayIntersector (uses pyembree if available for speed)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    for i, pose in enumerate(camera_poses):
        # Extract camera position (translation vector)
        cam_origin = pose[:3, 3]
        
        # 1. Back-face Culling Check (Fast)
        # Vector from camera to face center
        ray_vectors = centers - cam_origin
        
        # Calculate dot product: (ViewVector . Normal)
        # If dot > 0, the face normal is pointing same direction as view vector (away from cam)
        # We only want faces where dot < 0 (pointing towards cam)
        # Note: einsum is a fast way to do dot product across rows
        dots = np.einsum('ij,ij->i', ray_vectors, normals)
        
        # Indices of faces that are potentially visible (facing the camera)
        candidate_indices = np.where(dots < 0)[0]
        
        if len(candidate_indices) == 0:
            continue

        # 2. Occlusion Check (Ray Casting)
        # We cast rays from camera to the centers of candidate faces
        
        curr_origins = np.tile(cam_origin, (len(candidate_indices), 1))
        curr_dirs = ray_vectors[candidate_indices]
        
        # intersects_first returns the index of the first face hit by each ray
        # If the ray hits the target face directly, the index will match
        hit_face_indices = intersector.intersects_first(
            ray_origins=curr_origins,
            ray_directions=curr_dirs
        )
        
        # Check where the hit face is actually the target face
        # (This filters out faces that are blocked by other geometry)
        visible_mask = (hit_face_indices == candidate_indices)
        
        # Add the confirmed visible faces to our set
        confirmed_visible = candidate_indices[visible_mask]
        visible_face_indices.update(confirmed_visible)

    logger.info(f"Culling complete. Kept {len(visible_face_indices)} / {len(mesh.faces)} faces.")
    
    if len(visible_face_indices) == 0:
        logger.warning("Culling removed ALL faces. Returning empty mesh.")
        return trimesh.Trimesh()

    # Create a new mesh from only the visible faces
    # submesh returns a list of meshes, we usually just want the first (and only) one
    clean_mesh = mesh.submesh([list(visible_face_indices)], append=True)
    
    return clean_mesh

# ----------------- Existing Logic -----------------

def sample_surface_point(mesh, num_points, even=False):
    if even:
        sample_points, indexes = trimesh.sample.sample_surface_even(mesh, count=num_points)
        while len(sample_points) < num_points:
            more_sample_points, indexes = trimesh.sample.sample_surface_even(mesh, count=num_points)
            sample_points = np.concatenate([sample_points, more_sample_points], axis=0)
    else:
        sample_points, indexes = trimesh.sample.sample_surface(mesh, count=num_points)
    return sample_points[:num_points]


def load_mesh(fpath: str) -> trimesh.Trimesh:
    if fpath.endswith('.npz'):
        mesh_npz = np.load(fpath)
        verts = mesh_npz['verts']
        faces = mesh_npz['faces']
        faces = np.concatenate((faces, faces[:, list(reversed(range(faces.shape[-1])))]), axis=0)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    else:
        mesh = trimesh.load_mesh(fpath)
    return mesh


def compute_trimesh_chamfer(gt_points, gen_mesh, num_mesh_samples=30000):
    if gen_mesh is None or len(gen_mesh.faces) == 0:
        # Handle empty mesh case
        return 1.0, 1.0 # High penalty
        
    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    # only need numpy array of points
    gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer, gen_to_gt_chamfer


def compute_shape_score(output_mesh_path, target_mesh_path, camera_json_path=None):
    if output_mesh_path is None:
        logger.error('output mesh not found')
        return {}
    
    # 1. Load Prediction
    try:
        mesh_result = load_mesh(output_mesh_path)
    except ValueError:
        import traceback; traceback.print_exc()
        mesh_result = None
    
    if mesh_result is None:
        return {}

    # 2. CULLING STEP
    if camera_json_path and os.path.exists(camera_json_path):
        cameras = load_cameras(camera_json_path)
        if len(cameras) > 0:
            # Create a copy to avoid mutating cached data if any
            mesh_to_clean = mesh_result.copy()
            mesh_result = filter_hidden_geometry(mesh_to_clean, cameras)
        else:
            logger.warning(f"No cameras found in {camera_json_path}, skipping culling.")
    else:
        if camera_json_path: 
            logger.warning(f"Camera path provided but file missing: {camera_json_path}")

    # 3. Load GT
    try:
        mesh_scan = load_mesh(target_mesh_path)
    except Exception as e:
        logger.error(f'Could not load target mesh: {target_mesh_path}')
        return {}

    # 4. Compute Chamfer
    gt_to_gen_chamfer, gen_to_gt_chamfer = compute_trimesh_chamfer(mesh_scan, mesh_result, num_points_chamfer)
    
    print(f"GT to Gen Chamfer: {gt_to_gen_chamfer*2000:.6f}"
          f", Gen to GT Chamfer: {gen_to_gt_chamfer*2000:.6f}")
    bidir_chamfer = (gt_to_gen_chamfer + gen_to_gt_chamfer) / 2.
    return {'bidir_chamfer': bidir_chamfer}


def main():
    ap = argparse.ArgumentParser(description='Stanford ORB Chamfer evaluation (x2000, culled to training-view visibility).')
    ap.add_argument('--data_root', default='./load/orb', help='dir with blender_LDR/<scene>/ and ground_truth/<scene>/mesh_blender/mesh.obj')
    ap.add_argument('--exp_root', default='./exp')
    ap.add_argument('--trial', default='final_method_orb')
    ap.add_argument('--scenes', nargs='+', default=['gnome_scene007', 'pitcher_scene001', 'teapot_scene006', 'ball_scene002', 'cactus_scene007'])
    ap.add_argument('--out', default='orb_results.txt')
    args = ap.parse_args()
    dataset_path = args.data_root
    output_filename = args.out
    scenes = args.scenes

    print(f"Starting evaluation for {len(scenes)} scenes...")
    
    chamfer_results = []
    
    with open(output_filename, "w") as f:
        f.write(f"Scene Name, Bidirectional Chamfer Distance (x2000 scaled)\n")
        f.write("-" * 60 + "\n")

        for scene in scenes:
            # Construct Paths
            # Note: I assumed the blender_LDR folder is structured as dataset_path/blender_LDR/{scene}/transforms_train.json
            # You might need to adjust this depending on your exact folder structure
            gt_path = os.path.join(dataset_path, f"ground_truth/{scene}/mesh_blender/mesh.obj")
            pred_path = os.path.join(args.exp_root, f"radtets-{scene}", args.trial, "save", "final_mesh.ply")
            
            # Look for camera json. Assuming typical structure:
            cam_path = os.path.join(dataset_path, f"blender_LDR/{scene}/transforms_train.json")
            
            # Fallback check: sometimes transforms are in the scene root
            if not os.path.exists(cam_path):
                 cam_path_alt = os.path.join(dataset_path, f"{scene}/transforms_train.json")
                 if os.path.exists(cam_path_alt):
                     cam_path = cam_path_alt

            print(f"Processing: {scene}")
            
            if not os.path.exists(gt_path):
                print(f"  [Error] GT file missing: {gt_path}")
                f.write(f"{scene}, ERROR_GT_MISSING\n")
                continue
                
            if not os.path.exists(pred_path):
                print(f"  [Error] Prediction file missing: {pred_path}")
                f.write(f"{scene}, ERROR_PRED_MISSING\n")
                continue

            # Pass camera path to compute function
            scores = compute_shape_score(pred_path, gt_path, camera_json_path=cam_path)
            
            if 'bidir_chamfer' in scores:
                chamfer_dist = scores['bidir_chamfer'] * 2000
                chamfer_results.append(chamfer_dist)
                print(f"  Chamfer: {chamfer_dist:.6f}")
                f.write(f"{scene}, {chamfer_dist:.6f}\n")
            else:
                print(f"  [Error] Computation failed for {scene}")
                f.write(f"{scene}, ERROR_COMPUTATION\n")

        f.write("-" * 60 + "\n")
        if len(chamfer_results) > 0:
            avg_chamfer = np.mean(chamfer_results)
            print("=" * 40)
            print(f"Average Chamfer Distance: {avg_chamfer:.6f}")
            f.write(f"AVERAGE, {avg_chamfer:.6f}\n")
        else:
            print("No valid results to average.")
            f.write("AVERAGE, N/A\n")

    print(f"Done. Results saved to {output_filename}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()