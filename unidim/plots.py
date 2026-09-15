import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import os

def plot_sinogram(sino,save_file):
    g = sino.geometry
    plt.figure(figsize=(12, 6))

    plt.imshow(sino.values, aspect="auto", origin="lower",
        extent=(g.bin_edges[0], g.bin_edges[-1],  # position détecteur max
                g.angles[0]/math.pi*180, g.angles[-1]/math.pi*180 )
    )
    plt.xlabel("Position du détecteur s")
    plt.ylabel("Angle θ (degrés)")
    plt.title("Sinogramme")
    plt.colorbar(label="Densité")
    plt.savefig(save_file)

def plot_final_points(points, save_file):

    plt.figure(figsize=(7, 7))
    points = points.cpu()
    plt.scatter(points[:, 0],points[:, 1],s=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Reconstruction finale — {points.shape[0]} points")
    plt.axis("equal")
    plt.savefig(save_file)

def  plot_points(points, step , img_dir):
    os.makedirs(img_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5, color='blue')
    plt.title(f"Scatterplot step = {step}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{img_dir}/scatterplot_points_step_{step}.png', dpi=150, bbox_inches='tight')
    plt.close()  # Fermer la figure pour libérer la mémoire


def make_gif_from_png(img_dir):
    import glob, os
    from PIL import Image
    from natsort import natsorted
    pngs = natsorted(glob.glob(f"{img_dir}/scatterplot_points_step_*.png"))
    Image.open(pngs[0]).save(f"{img_dir}/scatterplot_points.gif",
                             save_all=True, append_images=[Image.open(p) for p in pngs[1:]],
                             duration=500,
                             loop=0)
    [os.remove(p) for p in pngs]