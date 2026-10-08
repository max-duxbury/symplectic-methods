import matplotlib.pyplot as plt
from pynamicalsys import DiscreteDynamicalSystem as dds, PlotStyler

system = dds(model="logistic map")
parameter_values = [2.6, 3.1, 3.5, 3.8]
total_time = 100

ps = PlotStyler()
ps.apply_style()
fig, ax = plt.subplots(figsize=(8, 4))
for r in parameter_values:
    trajectory = system.trajectory(0.2, total_time, parameters=[r])
    ax.plot(range(1, total_time + 1), trajectory, "o-", label=f"$r = {r}$")

ax.set_xlabel("Iteration $n$")
ax.set_ylabel("$x$")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.tight_layout()
plt.show()
