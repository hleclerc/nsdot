import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

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
    plt.scatter(points[:, 0],points[:, 1],s=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Reconstruction finale — {points.shape[0]} points")
    plt.axis("equal")
    plt.savefig(save_file)
