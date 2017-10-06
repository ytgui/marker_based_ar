import time
import numpy as np
import skimage.io, skimage.feature, skimage.transform, skimage.draw, skimage.morphology, skimage.color, skimage.measure, \
    skimage.filters
import matplotlib.pyplot as plt

image = skimage.io.imread('/home/bosskwei/Pictures/singlemarkersoriginal.png')
plt.imshow(image)
print(time.time())

gray = skimage.color.rgb2gray(image)
edge = skimage.feature.canny(gray, sigma=2)
edge = skimage.morphology.dilation(edge, skimage.morphology.square(7))
binary = gray > skimage.filters.threshold_mean(gray)


def check_bounding_rect(points):
    top, bottom = np.floor(np.min(points[:, 0])), np.ceil(np.max(points[:, 0]))
    left, right = np.floor(np.min(points[:, 1])), np.ceil(np.max(points[:, 1]))
    width, height = right - left, bottom - top
    ss = width * height
    tt = np.arctan2(np.minimum(width, height), np.maximum(width, height))
    if 1000 <= ss <= 5000 and np.pi * (1 / 4 - 1 / 12) <= tt <= np.pi / (1 / 4 + 1 / 12):
        valid = True
    else:
        valid = False
    return valid, int(top), int(bottom), int(left), int(right)


roi = []
contours = skimage.measure.find_contours(edge, level=0.8)
for n, contour in enumerate(contours):
    plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1)
    valid, top, bottom, left, right = check_bounding_rect(contour)
    if valid is True:
        rr, cc = skimage.draw.polygon_perimeter([top, top, bottom, bottom], [left, right, right, left])
        plt.plot(cc, rr, 'g-', linewidth=1)
        roi.append([left, top, gray[top:bottom, left:right]])


def select_margin_points(points, height, width):
    idx1 = np.argmin(np.square(points[:, 0] - 0) + np.square(points[:, 1] - 0))
    idx2 = np.argmin(np.square(points[:, 0] - 0) + np.square(points[:, 1] - width))
    idx3 = np.argmin(np.square(points[:, 0] - height) + np.square(points[:, 1] - 0))
    idx4 = np.argmin(np.square(points[:, 0] - height) + np.square(points[:, 1] - width))
    if np.unique([idx1, idx2, idx3, idx4]).size == 4:
        return points[[idx1, idx2, idx3, idx4]]
    else:
        return []


rects = []
for offset_left, offset_top, region in roi:
    coords = skimage.feature.corner_peaks(skimage.feature.corner_harris(region), min_distance=3)
    if len(coords) == 0:
        continue
    coords_filtered = select_margin_points(coords, region.shape[0], region.shape[1])
    if len(coords_filtered) == 0:
        continue
    rects.append(coords_filtered + [offset_top, offset_left])


def estimate(src, dst):
    def _center_and_normalize_points(points):
        import math

        centroid = np.mean(points, axis=0)

        rms = math.sqrt(np.sum((points - centroid) ** 2) / points.shape[0])

        norm_factor = math.sqrt(2) / rms

        matrix = np.array([[norm_factor, 0, 0, -norm_factor * centroid[0]],
                           [0, norm_factor, 0, -norm_factor * centroid[1]],
                           [0, 0, norm_factor, -norm_factor * centroid[2]],
                           [0, 0, 0, 1]])

        pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])

        new_pointsh = np.dot(matrix, pointsh).T

        new_points = new_pointsh[:, :3]
        new_points[:, 0] /= new_pointsh[:, 3]
        new_points[:, 1] /= new_pointsh[:, 3]
        new_points[:, 2] /= new_pointsh[:, 3]

        return matrix, new_points

    try:
        src_matrix, src = _center_and_normalize_points(src)
        dst_matrix, dst = _center_and_normalize_points(dst)
    except ZeroDivisionError:
        params = np.nan * np.empty((4, 4))
        return params#False

    xs = src[:, 0]
    ys = src[:, 1]
    zs = src[:, 2]
    xd = dst[:, 0]
    yd = dst[:, 1]
    rows = src.shape[0]

    # params: a0, a1, a2, b0, b1, b2, c0, c1
    '''
        A[:rows, 0] = xs
        A[:rows, 1] = ys
        A[:rows, 2] = 1
        A[:rows, 6] = - xd * xs
        A[:rows, 7] = - xd * ys
        #
        A[rows:, 3] = xs
        A[rows:, 4] = ys
        A[rows:, 5] = 1
        A[rows:, 6] = - yd * xs
        A[rows:, 7] = - yd * ys
        #
        A[:rows, 8] = xd
        #
        A[rows:, 8] = yd
    '''
    A = np.zeros((rows * 2, 12))
    A[:rows, 0] = xs
    A[:rows, 1] = ys
    A[:rows, 2] = zs
    A[:rows, 3] = 1
    A[:rows, 8] = - xd * xs
    A[:rows, 9] = - xd * ys
    A[:rows, 10] = - xd * zs
    #
    A[rows:2 * rows, 4] = xs
    A[rows:2 * rows, 5] = ys
    A[rows:2 * rows, 6] = zs
    A[rows:2 * rows, 7] = 1
    A[rows:2 * rows, 8] = - yd * xs
    A[rows:2 * rows, 9] = - yd * ys
    A[rows:2 * rows, 10] = - yd * zs
    #
    A[:rows, 11] = xd
    A[rows:2 * rows, 11] = yd

    _, _, V = np.linalg.svd(A)

    H = np.zeros((3, 4))
    # solution is right singular vector that corresponds to smallest
    # singular value
    H.flat = - V[-1, :-1] / V[-1, -1]
    # H[4, 4] = 1

    # De-center and de-normalize
    H = np.dot(np.linalg.inv(dst_matrix), np.dot(H, src_matrix))

    params = H

    return params#True


for rect in rects:
    plt.plot(rect[:, 1], rect[:, 0], 'xb', markersize=15)
    #
    model = skimage.transform.ProjectiveTransform()
    model.estimate(np.array([(64, 64), (64, 128), (128, 64), (128, 128)]), np.array(rect))
    #
    (rr1, cc1), (rr2, cc2) = model([(96, 96), (224, 96)])
    plt.arrow(cc1, rr1, cc2 - cc1, rr2 - rr1, width=2, color='green')
    (rr1, cc1), (rr2, cc2) = model([(96, 96), (96, 224)])
    plt.arrow(cc1, rr1, cc2 - cc1, rr2 - rr1, width=2, color='red')
    #
    # aa = estimate(np.array([(64, 64, 0), (64, 128, 0), (128, 64, 0), (128, 128, 0)]), np.array(rect))
    # (rr1, cc1), (rr2, cc2) = model()
    plt.arrow(cc1, rr1, 0, -64, width=2, color='blue')

print(time.time())
plt.show()
