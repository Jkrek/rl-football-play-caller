import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import imageio


def draw_play(yardline_start, action, yards_gained, step=None, save=False, folder="frames"):
    """
    Visualizes a single football play on a horizontal football field.
    """

    yardline_end = yardline_start + yards_gained

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Draw field gridlines
    for i in range(0, 101, 10):
        ax.axvline(i, color='gray', linestyle='--', linewidth=0.5)
        ax.text(i, 0.5, str(i), va='center', ha='center', fontsize=8)

    # Draw arrow for play
    ax.add_patch(patches.FancyArrow(
        yardline_start, 0.5, yards_gained, 0,
        width=0.1,
        length_includes_head=True,
        head_width=0.2,
        head_length=2,
        color='green' if yards_gained >= 0 else 'red'
    ))

    # Add text annotations
    ax.text(50, 0.9, f"Action: {action}", ha='center', fontsize=10, fontweight='bold')
    ax.text(50, 0.1, f"Start: {yardline_start} | Gained: {yards_gained} | End: {yardline_end}",
            ha='center', fontsize=9)

    plt.tight_layout()

    if save:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"step_{step}.png")
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def make_gif(folder="frames", filename="play_simulation.gif", duration=0.5):
    """
    Converts saved frame PNGs into a GIF.
    """
    images = []
    files = sorted(os.listdir(folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
    for file in files:
        if file.endswith(".png"):
            images.append(imageio.imread(os.path.join(folder, file)))
    imageio.mimsave(filename, images, duration=duration)
