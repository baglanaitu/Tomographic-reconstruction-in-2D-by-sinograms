import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def Get_points(point1, point2):

    # Initializeing variables
    points = []
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)

    # If it is steep
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swapping
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    #differences
    dx = x2 - x1
    dy = y2 - y1

    #error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # create the points in between
    y = y1
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
    if swapped:
        points.reverse()
    return points



S_cone = Image.open("data/sinogram_2d_cone.tif")
S_parallel = Image.open("data/sinogram_2d_parallel.tif")

Image._show(S_parallel)
#Image._show(S_cone)

sinogram = np.array(S_parallel)
print(sinogram.shape)


def get_path(l, angle, x, y, offset):

    # Radian and L
    theta = np.pi / (180) * angle * (360 / 200)
    L = np.round(x * np.cos(theta) + y * np.sin(theta)) + offset

    # extract only the coordinate where the line passes through
    ind = np.where(L == l)

    # get the start and end points a

    index_lst = list(zip(ind[0], ind[1]))
    if len(index_lst) == 0:
        return (None, None)
    elif len(index_lst) == 1:
        start_point = index_lst[0]
        end_point = index_lst[0]
    else:
        start_point = index_lst[0]
        end_point = index_lst[-1]


    # return the start and end points
    return (start_point, end_point)


def backprojector_2D(sinogram, dim):
    # reconstructed  image
    rec_img = np.zeros(dim)

    # coordinates of recons_obj image
    x_rec = np.zeros(dim)
    y_rec = np.zeros(dim)

    # Size of image (rows, columns)
    N = int(dim[0] / 2)

    # bias
    bias = int(sinogram.shape[1] / 2)

    # fill x with image rows, and y with image columns
    for m, n in enumerate(range(-N, N)):
        x_rec[m, :] = n
    for m, n in enumerate(range(-N, N)):
        y_rec[:, m] = n

    # iterate through all the projection
    for angle in range(sinogram.shape[0]):
        # iterate throigh all the voxels
        for proj_L in range(sinogram.shape[1]):
            # return the start and end coords of the projection
            proj_1, proj_2 = get_path(proj_L, angle, x_rec, y_rec, bias)

            if proj_1:
                # return the path
                x0, y0 = proj_1
                x1, y1 = proj_2
                proj_path = Get_points(proj_1,proj_2)

                # Backpropagation
                for x, y in proj_path:
                    rec_img[x, y] = rec_img[x, y] + sinogram[angle, proj_L]

    final_img = np.rot90(rec_img)
    return np.rot90(rec_img)



# to plot the sinogram to have more intuition
import time
from IPython import display

# reconstruction from sinogram
# image dimention
im_size = [180,180]
rec_image = backprojector_2D(sinogram, im_size)
plt.imshow(rec_image, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show()


def Get_Points_spehrical(point, angle, xlim=100, ylim=100):
    x, y = point
    # check the special cases where the tan is not defined
    if angle % 180 == 0:
        point_end = [-xlim, y]
        point_start = [xlim, y]

    elif angle == 90 and angle == 270:
        point_end = [x, -ylim]
        point_start = [x, y]

    else:
        # calculate the slope
        m = np.tan(angle)
        # calculate the inctercept
        c = y - m * x

        if abs(m) <= 1:
            x2 = xlim
            point_end = [x2, np.round(m * x2 + c)]
            point_start = [-x2, np.round(-m * x2 + c)]
        else:
            y2 = ylim
            point_end = [np.round((y2 - c) / m), y2]
            point_start = [np.round((-y2 - c) / m), -y2]

    return point_start, point_end


def Reconstruct(sinogram, im_dim=[400, 400], voxel_size=1, s_to_obj=300, s_to_ditect=400):
    # initialize the recons_obj image with zeros
    recons_obj = np.zeros(im_dim)

    # define the source to detector geometer
    R = s_to_obj
    D = s_to_ditect - R
    grid_offset = int((im_dim[1]) / 2)

    # get the field of view
    detectorSize = sinogram.shape[1] * voxel_size
    fov = 2 * R * np.sin(np.arctan(detectorSize / 2 / (D + R)))
    # print(detectorSize, fov)

    # create the x and y grid points
    x = np.linspace(-fov / 2, fov / 2, im_dim[0])
    y = np.linspace(-fov / 2, fov / 2, im_dim[1])
    [x_grid, y_grid] = np.meshgrid(x, y)
    # print(x_grid.shape, y_grid.shape)

    # cartesian to polar x, y => rho, phi
    rho = np.sqrt(x_grid ** 2 + y_grid ** 2)
    phi = np.arctan2(y_grid, x_grid)
    # print(rho, phi)

    # arrange projection angles
    alpha = 360 / sinogram.shape[0]
    # dtheta = alpha[1] - alpha[0]
    print(alpha)

    for alpha_ind in range(sinogram.shape[1]):
        for l in range(sinogram.shape[0]):
            angle = alpha * l * (np.pi / 180)

            src = [np.round(-R * np.cos(angle)), np.round(-R * np.sin(angle))]
            det = [np.round(D * np.cos(angle)), np.round(D * np.sin(angle))]

            offset = int((sinogram.shape[1]) / 2) - alpha_ind
            norm = [np.round(offset * np.cos(angle + np.pi / 2)), np.round(offset * np.sin(angle + np.pi / 2))]

            det_pnt = np.array(det) + np.array(norm)

            slope = (src[1] - det_pnt[1]) / (src[0] - det_pnt[0])
            gamma_angle = np.arctan(slope)

            proj_1, proj_2 = Get_Points_spehrical(det_pnt, gamma_angle)
            #             print(proj_1,proj_2)
            proj_1 = [int(i) + 150 for i in proj_1]
            proj_2 = [int(i) + 150 for i in proj_2]

            proj_1 = [399 if i > 399 else i for i in proj_1]
            proj_2 = [399 if i > 399 else i for i in proj_2]

            if proj_1:
                path_points = Get_points(proj_1, proj_2)
                #             # print(path_points)
                #             # backprojection stage
                for i, j in path_points:
                    recons_obj[i, j] = recons_obj[i, j] + sinogram[l, alpha_ind]

            # recons_obj[x_in,x_in] += sinogram[l, alpha_ind]

        plt.imshow(recons_obj, cmap=plt.get_cmap('gray'))
        plt.title([l, alpha_ind])
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()

    return recons_obj


# testing

# show the sinogram
# convert the image into numpy array
sinogram = np.array(S_cone)
recon_image = Reconstruct(sinogram)

plt.imshow(recon_image, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show()