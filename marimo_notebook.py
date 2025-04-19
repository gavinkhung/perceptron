

import marimo

__generated_with = "0.13.1-dev1"
app = marimo.App(width="full", app_title="Perceptron")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    np.random.seed(0)


    def generate_dataset(dims, normal_vector):
        # create 3D grid of points
        points = np.linspace(-1, 1, dims)
        X, Y, Z = np.meshgrid(points, points, points)

        # features are the x, y, z coordinates
        features = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

        # labels are the side each point is on the hyperplane
        distances = np.dot(features, normal_vector)
        labels = np.where(distances >= 0, 1, -1)
        return X, Y, Z, features, labels


    # normalized normal vector
    target_normal_vector = np.array([1.0, 1.0, 1.0])
    target_normal_vector = target_normal_vector / np.linalg.norm(
        target_normal_vector
    )

    scaling = 5
    X, Y, Z, features, labels = generate_dataset(scaling, target_normal_vector)


    def generate_hyperplane(scaling, normal_vector):
        # create 2D points
        points = np.linspace(-1, 1, scaling)
        xx, yy = np.meshgrid(points, points)

        # the z value is the defined by the hyperplane
        zz = -(normal_vector[0] * xx + normal_vector[1] * yy) / normal_vector[2]
        return xx, yy, zz


    # Ground truth
    xx_target, yy_target, zz_target = generate_hyperplane(
        scaling, target_normal_vector
    )


    def hinge_loss(w, x, b, y):
        return max(0.0, -y * (np.dot(w, x) + b))


    def create_plots():
        fig = plt.figure(figsize=(16 / 9.0 * 4, 4 * 1.25), layout="constrained")
        fig.suptitle("Perceptron")

        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Hyperplane Decision Boundary")
        ax.view_init(20, -35)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        return ax


    def plot_graphs(
        ax,
        scaling,
        target_normal_vector,
        predictions,
        features,
        labels,
        weights,
    ):
        ground_truth_legend = ax.plot_surface(
            xx_target,
            yy_target,
            zz_target,
            color="red",
            alpha=0.2,
            label="Ground Truth",
        )
        ax.quiver(
            0,
            0,
            0,
            target_normal_vector[0],
            target_normal_vector[1],
            target_normal_vector[2],
            color="red",
            length=1,
            arrow_length_ratio=0.1,
        )

        # Perceptron predictions using 2D graph to show linear transformation
        def generate_colors(arr):
            return ["green" if d >= 0 else "orange" for d in arr]

        # Perceptron predictions using 3D graph to show hyperplane
        predictions_colors = generate_colors(predictions)
        predictions_norm = np.maximum(1 - np.exp(-(predictions**2)), 0.2)

        ax.scatter(
            features[:, 0],
            features[:, 1],
            features[:, 2],
            c=predictions_colors,
            marker="o",
            alpha=predictions_norm,
        )

        xx, yy, zz = generate_hyperplane(scaling, weights)
        predictions_legend = ax.plot_surface(
            xx,
            yy,
            zz,
            color="blue",
            alpha=0.2,
            label="Prediction",
        )
        ax.quiver(
            0,
            0,
            0,
            weights[0],
            weights[1],
            weights[2],
            color="blue",
            length=1,
            arrow_length_ratio=0.1,
        )

        # Legend
        ax.legend(
            (ground_truth_legend, predictions_legend),
            ("Ground Truth", "Predictions"),
            loc="upper left",
        )
        plt.show()


    x_slider = mo.ui.slider(
        -1.0, 1.0, 0.1, value=0.75, show_value=True, label="$x$"
    )
    y_slider = mo.ui.slider(
        -1.0, 1.0, 0.1, value=-0.75, show_value=True, label="$y$"
    )
    z_slider = mo.ui.slider(
        -1.0, 1.0, 0.1, value=-0.75, show_value=True, label="$z$"
    )
    b_slider = mo.ui.slider(
        -1.0, 1.0, 0.1, value=0.45, show_value=True, label="$b$"
    )

    mo.hstack([mo.vstack([x_slider, y_slider]), mo.vstack([z_slider, b_slider])])
    return (
        b_slider,
        create_plots,
        features,
        labels,
        np,
        plot_graphs,
        scaling,
        target_normal_vector,
        x_slider,
        y_slider,
        z_slider,
    )


@app.cell(hide_code=True)
def _(
    b_slider,
    create_plots,
    features,
    labels,
    np,
    plot_graphs,
    scaling,
    target_normal_vector,
    x_slider,
    y_slider,
    z_slider,
):
    weights = np.array([x_slider.value, y_slider.value, z_slider.value])
    weights = weights / np.linalg.norm(weights)
    bias = b_slider.value

    ax = create_plots()

    predictions = np.array([])

    for x, y in zip(features, labels):
        # Forward Propagation
        output = np.dot(weights, x) + bias
        predictions = np.append(predictions, output)

    plot_graphs(
        ax,
        scaling,
        target_normal_vector,
        predictions,
        features,
        labels,
        weights,
    )
    return


if __name__ == "__main__":
    app.run()
