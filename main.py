import argparse

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, map_coordinates, shift


MAX_ITER = 1000
EPS = 0.007
SMOOTH_SIGMA = 1.0
K2 = 0.9
NORMAL_SHIFT = 4
GRAD_EPS = 1e-15


def load_image(path):
    return np.asarray(Image.open(path), dtype=np.float64)


def load_snake(path):
    snake = np.loadtxt(path, dtype=np.float64)
    return snake[:-1]


def snake_to_mask_array(snake, shape):
    height, width = shape[:2]
    mask = Image.new("L", (width, height), 0)
    points = [(float(x), float(y)) for x, y in snake]
    drawer = ImageDraw.Draw(mask)
    drawer.polygon(points, outline=255, fill=255)
    return np.asarray(mask, dtype=np.uint8)


def save_mask(path, snake, shape):
    mask = snake_to_mask_array(snake, shape)
    Image.fromarray(mask).save(path)


def load_binary_mask(path):
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return mask > 0


def compute_iou(gt_mask_path, snake, shape):
    pred_mask = snake_to_mask_array(snake, shape) > 0
    gt_mask = load_binary_mask(gt_mask_path)

    if pred_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Размеры масок не совпадают: pred={pred_mask.shape}, gt={gt_mask.shape}"
        )

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 1.0

    return intersection / union


def derivative_x(img):
    return -(
        shift(img, (0, 1), mode="nearest")
        - shift(img, (0, -1), mode="nearest")
    ) / 2.0


def derivative_y(img):
    return -(
        shift(img, (1, 0), mode="nearest")
        - shift(img, (-1, 0), mode="nearest")
    ) / 2.0


def resample_snake(snake):
    resampled = np.zeros(snake.shape)

    closed = np.vstack([snake, snake[0]])
    distances = np.cumsum(np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0.0)

    interp_x = interp1d(distances, closed[:, 0], kind="cubic")
    interp_y = interp1d(distances, closed[:, 1], kind="cubic")

    new_distances = np.linspace(0.0, distances[-1], resampled.shape[0])
    resampled[:, 0] = interp_x(new_distances)
    resampled[:, 1] = interp_y(new_distances)

    return resampled


def compute_normals(snake):
    dx = shift(snake[:, 0], -NORMAL_SHIFT, mode="grid-wrap") - shift(
        snake[:, 0], NORMAL_SHIFT, mode="grid-wrap"
    )
    dy = shift(snake[:, 1], -NORMAL_SHIFT, mode="grid-wrap") - shift(
        snake[:, 1], NORMAL_SHIFT, mode="grid-wrap"
    )

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    magnitude = np.maximum(magnitude, 1e-12)

    return np.column_stack([-dy / magnitude, dx / magnitude])


def build_internal_mx(n_points, alpha, beta, tau):
    eye = np.eye(n_points)

    second = (
        np.roll(eye, -1, axis=0)
        + np.roll(eye, 1, axis=0)
        - 2.0 * eye
    )

    fourth = (
        np.roll(eye, -2, axis=0)
        + np.roll(eye, 2, axis=0)
        - 4.0 * np.roll(eye, -1, axis=0)
        - 4.0 * np.roll(eye, 1, axis=0)
        + 6.0 * eye
    )

    a = alpha * n_points ** 2
    b = beta * n_points ** 4

    internal = -a * second + b * fourth
    system = eye + tau * internal

    return np.linalg.inv(system)


def build_external_force(image, w_line, w_edge):
    smooth = gaussian_filter(image, SMOOTH_SIGMA)

    grad_squared = derivative_x(smooth) ** 2 + derivative_y(smooth) ** 2
    potential = -w_line * smooth - w_edge * grad_squared

    potential_x = derivative_x(potential)
    potential_y = derivative_y(potential)

    norm = np.sqrt(potential_x ** 2 + potential_y ** 2)

    force_x = potential_x / (norm + GRAD_EPS)
    force_y = potential_y / (norm + GRAD_EPS)

    return force_x, force_y


def active_contour(image, init_snake, alpha, beta, tau, w_line, w_edge, kappa):
    snake = init_snake.astype(np.float64).copy()
    n_points = len(snake)

    A_inv = build_internal_mx(n_points, alpha, beta, tau)
    force_x_img, force_y_img = build_external_force(image, w_line, w_edge)

    square = snake.shape[0] * snake.shape[1]

    for _ in range(MAX_ITER):
        x = snake[:, 0]
        y = snake[:, 1]

        force_x = map_coordinates(force_x_img, [y, x], order=1, mode="nearest")
        force_y = map_coordinates(force_y_img, [y, x], order=1, mode="nearest")

        normals = compute_normals(snake)

        ext_x = kappa * normals[:, 0] - K2 * force_x
        ext_y = kappa * normals[:, 1] - K2 * force_y

        rhs_x = x + tau * ext_x
        rhs_y = y + tau * ext_y

        x_new = A_inv @ rhs_x
        y_new = A_inv @ rhs_y

        new_snake = np.column_stack([x_new, y_new])

        if np.sum((snake - new_snake) ** 2) <= EPS * square:
            snake = new_snake
            break

        snake = resample_snake(new_snake)
    print (compute_iou("nucleus_mask.png", snake, image.shape))

    return snake


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=str)
    parser.add_argument("initial_snake", type=str)
    parser.add_argument("output_image", type=str)
    parser.add_argument("alpha", type=float)
    parser.add_argument("beta", type=float)
    parser.add_argument("tau", type=float)
    parser.add_argument("w_line", type=float)
    parser.add_argument("w_edge", type=float)
    parser.add_argument("kappa", type=float, nargs="?", default=0.0)
    parser.add_argument("--gt-mask", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    image = load_image(args.input_image)
    init_snake = load_snake(args.initial_snake)

    result_snake = active_contour(
        image=image,
        init_snake=init_snake,
        alpha=args.alpha,
        beta=args.beta,
        tau=args.tau,
        w_line=args.w_line,
        w_edge=args.w_edge,
        kappa=args.kappa,
    )

    save_mask(args.output_image, result_snake, image.shape)

    if args.gt_mask is not None:
        iou = compute_iou(args.gt_mask, result_snake, image.shape)
        print(iou)


if __name__ == "__main__":
    main()