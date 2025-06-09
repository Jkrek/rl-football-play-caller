import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import imageio


def draw_play(
    yardline_start,
    action,
    yards_gained,
    step,
    down,
    distance,
    score_team,
    score_opp,
    possession,
    save=False,
    folder="frames",
    reset=False
):
    """
    Visualizes a single football play on a horizontal football field with contextual overlays.
    """
    yardline_end = yardline_start + yards_gained

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.set_size_inches(10, 3)
    fig.set_dpi(100)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Draw field
    field = patches.Rectangle((0, 0), 100, 10, linewidth=1, edgecolor='green', facecolor='lightgreen')
    ax.add_patch(field)

    # Yard markers
    for yd in range(10, 100, 10):
        ax.text(yd, 9, str(yd), ha='center', va='center', fontsize=8, color='gray')

    # Line of scrimmage
    ax.axvline(x=yardline_start, color='blue', linestyle='--', lw=2)
    ax.text(yardline_start, 1, "LOS", rotation=90, verticalalignment='bottom', fontsize=8, color='blue')

    # Arrow for yards gained
    ax.annotate(
        "",
        xy=(yardline_end, 5),
        xytext=(yardline_start, 5),
        arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
    )
    ax.text((yardline_start + yardline_end) / 2, 5.7, f"{yards_gained:.1f} yds", ha="center", fontsize=9)

    # Top info: Step, Action, Score, Possession
    ax.text(1, 9.2, f"Step {step}", fontsize=9, weight="bold")
    ax.text(20, 9.2, f"Action: {action}", fontsize=9)
    ax.text(55, 9.2, f"Down: {down}, Distance: {distance}", fontsize=9)
    ax.text(80, 9.2, f"Score: {score_team} - {score_opp}", fontsize=9)
    ax.text(95, 9.2, f"Poss: {possession}", fontsize=9, ha='right')

    if reset:
        ax.text(50, 1, "New Possession", ha='center', fontsize=11, color='purple', weight='bold')

    if save:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"frame_{step:04d}.png")
        plt.savefig(filename, bbox_inches=None)


    plt.close(fig)


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
