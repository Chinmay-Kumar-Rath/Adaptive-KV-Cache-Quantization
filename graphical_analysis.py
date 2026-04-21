import matplotlib.pyplot as plt
def generate_plots(steps, memory_saved_pct, mse_values):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, memory_saved_pct, marker="o", markersize=3, color="blue")
    ax1.set_title("Theoretical Memory Saved vs Tokens")
    ax1.set_xlabel("Number of Tokens")
    ax1.set_ylabel("Memory Saved (%)")
    ax1.grid(True)

    ax2.plot(steps, mse_values, marker="o", markersize=3, color="red")
    ax2.set_title("MSE vs Tokens")
    ax2.set_xlabel("Number of Tokens")
    ax2.set_ylabel("MSE")
    ax2.grid(True)
    fig.suptitle("Simulation Graphical Analysis")
    plt.tight_layout()
    plt.show()