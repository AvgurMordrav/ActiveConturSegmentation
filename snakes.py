import argparse
import utils

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift

from PIL import Image
import numpy as np


def IoU(file1, file2):
    img1 = np.array(Image.open(file1))
    img2 = np.array(Image.open(file2))
    if img1.shape != img2.shape:
        return 0
    intercection = np.count_nonzero(img1 * img2)
    union = np.count_nonzero(img1 + img2)
    return intercection / union

def get_A(alpha, beta, n):
    a = alpha * n ** 2
    b = beta * n ** 4
    res = np.zeros((n, n))
    tmp = np.array([b, -a - 4 * b, 2 * a + 6 * b, -a - 4 * b, b])
    for i in range(n):
        if i < 2:
            res[i, :i+3] = tmp[-i-3:]
            res[i, i-2:] = tmp[:2-i]
        elif i > n - 3:
            res[i, :i+3-n] = tmp[n-i-3:]
            res[i, i-2-n:] = tmp[:n-i+2]
        else:
            res[i, i-2:i+3] = tmp
    return res


dx = lambda img : -(shift(img, (0, 1), mode='nearest') - shift(img, (0, -1), mode='nearest')) / 2
dy = lambda img : -(shift(img, (1, 0), mode='nearest') - shift(img, (-1, 0), mode='nearest')) / 2

def resampling(snake):
    res = np.zeros(snake.shape)
    snake = np.concatenate((snake, snake[0].reshape(1, -1)), axis=0)
    dist = np.cumsum(np.sqrt(np.sum(np.diff(snake, axis=0) ** 2, axis=1)))
    dist = np.insert(dist, 0, 0)
    interpolator = interp1d(dist, snake[:, 0], kind='cubic')
    res[:, 0] = interpolator(np.linspace(0, dist[-1], res.shape[0]))
    interpolator = interp1d(dist, snake[:, 1], kind='cubic')
    res[:, 1] = interpolator(np.linspace(0, dist[-1], res.shape[0]))
    return res

def bilinear_interpolate(mat, x, y):
    x1 = np.floor(x).astype(int)
    x2 = x1 + 1
    y1 = np.floor(y).astype(int)
    y2 = y1 + 1
    x1 = min(mat.shape[1] - 1, max(0, x1))
    x2 = min(mat.shape[1] - 1, max(0, x2))
    y1 = min(mat.shape[0] - 1, max(0, y1))
    y2 = min(mat.shape[0] - 1, max(0, y2))
    f11 = mat[y1, x1]
    f12 = mat[y2, x1]
    f21 = mat[y1, x2]
    f22 = mat[y2, x2]
    a = (x2 - x) * (y2 - y)
    b = (x2 - x) * (y - y1)
    c = (x - x1) * (y2 - y)
    d = (x - x1) * (y - y1)
    return a * f11 + b * f12 + c * f21 + d * f22

def normals(snake):
    dx = shift(snake[:, 0], -4, mode='grid-wrap') - shift(snake[:, 0], 4, mode='grid-wrap')
    dy = shift(snake[:, 1], -4, mode='grid-wrap') - shift(snake[:, 1], 4, mode='grid-wrap') 
    magn = (dx ** 2 + dy ** 2) ** 0.5
    return np.stack((-dy / magn, dx / magn), axis=1)

def get_F(Fx, Fy, snake):
    res = np.zeros(snake.shape)
    for i in range(snake.shape[0]):
        res[i, 0] = bilinear_interpolate(Fx, snake[i, 0], snake[i, 1])
        res[i, 1] = bilinear_interpolate(Fy, snake[i, 0], snake[i, 1])
    return res

def prog(alpha, beta, tau, w_line, w_edge, kappa):
    sigma, k1, k2 = 1, kappa, 0.9

    snake = np.loadtxt(args.initial_snake)[: -1]
    start_snake = np.copy(snake)
    A = get_A(alpha, beta, snake.shape[0])
    AtauI_inv = np.linalg.inv(A * tau + np.eye(snake.shape[0]))

    img = np.array(Image.open(args.input_image), dtype="float64")
    blurred = gaussian_filter(img, sigma)
    grad_squared = dx(blurred) ** 2 + dy(blurred) ** 2
    P = - w_line * blurred - w_edge * grad_squared

    Px =  dx(P)
    Py =  dy(P)
    grad = (Px ** 2 + Py ** 2) ** 0.5
    Fx =  Px / (grad + 1e-15)
    Fy =  Py / (grad + 1e-15)

    square = snake.shape[0] * snake.shape[1]
    
    #(I + τA)X_t+1 = X_t + τF(X_t)

    i = 0
    while(1):    
        new_snake = AtauI_inv @ (snake + tau * (k1 * normals(snake) - k2 * get_F(Fx, Fy, snake)))
        if np.sum((snake - new_snake) ** 2)  <= 0.007 * square or i > 1000:
            break
        i += 1
        snake = resampling(new_snake)
    #utils.display_snake(img, start_snake, new_snake)# use to see result
    utils.save_mask(args.output_image, new_snake, img)
    
parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str)
parser.add_argument('initial_snake', type=str)
parser.add_argument('output_image', type=str)
parser.add_argument('alpha', type=float)
parser.add_argument('beta', type=float)
parser.add_argument('tau', type=float)
parser.add_argument('w_line', type=float)
parser.add_argument('w_edge', type=float)
parser.add_argument('kappa', type=float)
args = parser.parse_args()

prog(args.alpha, args.beta, args.tau, args.w_line, args.w_edge, args.kappa)

'''
optuna results:
======================================================================================================
ASTRANAUT => 0.9637677193366537
python snakes.py astranaut.png astranaut_init_snake.txt astranaut_result.png 9.2e-7 1.49e-8 0.87 1.383 0.649 0.178

COFFEE => 0.9744136460554371
python snakes.py coffee.png coffee_init_snake.txt coffee_result.png 1.8e-6 9.4e-8 1.2 0.139 0.265 0.0212

COINS  => 0.9328802039082413
python snakes.py coins.png coins_init_snake.txt coins_result.png 2.3845e-7 4.12e-6 0.819 1.288 1.373 0.094

MICROARRAY => 0.9774472556786031
python snakes.py microarray.png microarray_init_snake.txt microarray_result.png 3.77e-6 7.4e-9 0.946 0.934 1.342 -0.31

NUCLEUS => 0.9173361726841545
python snakes.py nucleus.png nucleus_init_snake.txt nucleus_result.png 5.3e-5 3.26e-7 1.137 0.66 1.19 -0.214

average = 0.953168999532618
'''