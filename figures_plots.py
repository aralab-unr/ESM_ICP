import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import os

# Load point clouds
aligned = o3d.io.read_point_cloud("saved_pcds/aligned_cicp_25.pcd")
target = o3d.io.read_point_cloud("saved_pcds/target.pcd")

# Convert to numpy
aligned_np = np.asarray(aligned.points)
target_np = np.asarray(target.points)

# Create figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(aligned_np[:, 0], aligned_np[:, 1], aligned_np[:, 2], s=1, c='r')
ax.scatter(target_np[:, 0], target_np[:, 1], target_np[:, 2], s=1, c='g')
#ax.set_title("Source (Red) vs Target (Green)")
ax.legend()
ax.set_axis_off()
ax.view_init(elev=20, azim=30)

# Save function
def save_view(event=None):
    os.makedirs("plots", exist_ok=True)
    filename = "plots/hp_cicp.pdf"
    
    # Temporarily hide the button before saving
    button_ax.set_visible(False)
    fig.canvas.draw_idle()  # Refresh without the button
    
    print(f"[📸] Saving to '{filename}'...")
    fig.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.0)
    print("✅ Saved!")

    # Restore button after saving
    button_ax.set_visible(True)
    fig.canvas.draw_idle()

# Button setup
button_ax = fig.add_axes([0.81, 0.01, 0.15, 0.06])
save_button = Button(button_ax, 'Save View', hovercolor='0.975')
save_button.on_clicked(save_view)

# Key binding
fig.canvas.mpl_connect('key_press_event', lambda event: save_view() if event.key == 's' else None)

# Show viewer
plt.show()
