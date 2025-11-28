from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    filter_x, filter_y = None, None

    # sobel filters
    filter_x = [[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]] 

    filter_y = [[ 1,  2,  1],
                [ 0,  0,  0],
                [-1, -2, -1]]

    filter_x = np.array(filter_x)
    filter_y = np.array(filter_y)

    return filter_x, filter_y

def filter_image(image, filter):
    image_filtered = None

    image_filtered = np.zeros(image.shape) # initialize filtered image
    k = filter.shape[0] # k x k kernel size
    image_padded = np.pad(image, k//2, mode="constant") # zero padding

    for u in range(image.shape[0]):
        for v in range(image.shape[1]):
            c = [u + k // 2, v + k // 2] # center of filtered image
            M_i = image_padded[c[0] - k // 2: c[0] + k // 2 + 1, c[1] - k // 2: c[1] + k // 2 + 1]
            convolved = M_i * filter # element-wise multiplication
            filtered_pixel = np.sum(convolved)
            image_filtered[u, v] = filtered_pixel

    return image_filtered

def get_gradient(image_dx, image_dy):
    grad_mag, grad_angle = None, None
    
    grad_mag = np.sqrt(image_dx**2 + image_dy**2)
    grad_angle = np.arctan2(image_dy, image_dx) % np.pi # [0, pi)

    return grad_mag, grad_angle

def build_histogram(grad_mag, grad_angle, cell_size):
    ori_histo = None
    
    m = grad_angle.shape[0]
    n = grad_angle.shape[1]
    bins = 6 # selected for this implementation

    # to avoid going out of bounds 
    m_snipped = m - m % cell_size 
    n_snipped = n - n % cell_size

    ori_histo = np.zeros((m // cell_size, n // cell_size, bins))

    for u in range(m_snipped): 
        for v in range(n_snipped):
            angle = grad_angle[u, v] # [0, pi)
            
            # following a specific binning strategy
            if angle >= np.pi / 12 and angle < 3 * np.pi / 12:
                bin_index = 1
            elif angle >= 3 * np.pi / 12 and angle < 5 * np.pi / 12:
                bin_index = 2
            elif angle >= 5 * np.pi / 12 and angle < 7 * np.pi / 12:
                bin_index = 3
            elif angle >= 7 * np.pi / 12 and angle < 9 * np.pi / 12:
                bin_index = 4
            elif angle >= 9 * np.pi / 12 and angle < 11 * np.pi / 12:
                bin_index = 5
            else:
                bin_index = 0

            ori_histo[u // cell_size, v // cell_size, bin_index] += grad_mag[u, v]

    return ori_histo

def get_block_descriptor(ori_histo, block_size):
    ori_histo_normalized = None

    # definitions for convenience
    M = ori_histo.shape[0]
    N = ori_histo.shape[1]
    bins = ori_histo.shape[2]
    new_M = M - (block_size - 1)
    new_N = N - (block_size - 1)

    # following given equations
    ori_histo_normalized = np.zeros((new_M, new_N, bins * block_size**2)) # placeholder
    e = 0.001 # what was used in example

    for u in range(new_M):
        for v in range(new_N):
            block = ori_histo[u: u + block_size, v: v + block_size, :] # block_size x block_size x bins
            block_vector = block.flatten()
            norm = np.sqrt(np.sum(block_vector**2) + e**2) # following the L2 normalization equation
            normalized_block = block_vector / norm # original vector 
            ori_histo_normalized[u, v, :] = normalized_block 

    return ori_histo_normalized

def extract_hog(image, cell_size=8, block_size=2):
    # convert grey-scale image to double format
    image = image.astype('float') / 255.0
    hog = None

    filter_x, filter_y = get_differential_filter() # getting filters
    # getting differential images
    image_dx = filter_image(image, filter_x) 
    image_dy = filter_image(image, filter_y)
    grad_mag, grad_angle = get_gradient(image_dx, image_dy) # computing gradients
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size) # building histogram 
    ori_histo_normalized = get_block_descriptor(ori_histo, block_size) # getting block descriptors
    hog = ori_histo_normalized.flatten() # final HOG feature vector

    return hog

def face_detection(I_target, I_template):
    bounding_boxes = None

    # template
    bounding_boxes = []
    unfiltered_boxes = []

    I_template = I_template.astype('float') / 255.0
    template_hog = extract_hog(I_template) # a
    template_mean = np.mean(template_hog) # ~a
    template = template_hog - template_mean # a - ~a

    # target (the main image)
    I_target = I_target.astype('float') / 255.0

    u = I_target.shape[0]
    v = I_target.shape[1] 
    m = I_template.shape[0]
    n = I_template.shape[1]

    for i in range(u - m):
        for j in range(v - n):
            patch = I_target[i: i + m, j: j + n] # machine heavy.
            patch_hog = extract_hog(patch) # b
            patch_mean = np.mean(patch_hog) # ~b
            patch = patch_hog - patch_mean # b - ~b

            ncc = np.dot(template, patch) / (np.linalg.norm(template) * np.linalg.norm(patch)) # NCC formula
            
            threshold = 0.45 # hyperparameter

            if ncc > threshold:
                unfiltered_boxes.append([j, i, ncc]) # x, y, score

    def compute_iou(bb1, bb2, width, height): # returns IoU value
        x1 = max(bb1[0], bb2[0])
        y1 = max(bb1[1], bb2[1])
        x2 = min(bb1[0], bb2[0]) + width
        y2 = min(bb1[1], bb2[1]) + height

        if x1 < x2 and y1 < y2:
            u = (x2 - x1) * (y2 - y1)
            denom = (2 * width * height) - u
            return (u / denom)
        else:
            return 0.0

    # using NMS

    unfiltered_boxes = sorted(unfiltered_boxes, key=lambda x: x[2], reverse=True) # sorting by score
    while len(unfiltered_boxes) > 0: # for the break part
        best_box = unfiltered_boxes[0] # getting the highest score

        bounding_boxes.append(best_box) 
        unfiltered_boxes.remove(best_box) 

        for i in range(len(unfiltered_boxes)):
            IoU = compute_iou(best_box, unfiltered_boxes[i], n, m) 
            if IoU > 0.5: # suppressing boxes with greater than certain threshold
                unfiltered_boxes[i] = None

        while None in unfiltered_boxes:
            unfiltered_boxes.remove(None) 

    bounding_boxes = np.array(bounding_boxes)
    return bounding_boxes

# ----- Visualization Functions -----

def visualize_hog(image, hog, cell_size=8, block_size=2, num_bins=6):
    image = image.astype('float') / 255.0
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = image.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()

def visualize_face_detection(I_target, bounding_boxes, box_size):

    I_target = I_target.convert("RGB")
    ww, hh = I_target.size

    draw = ImageDraw.Draw(I_target)

    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size[1]
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size[0]

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=1)
        bbox_text = f'{bounding_boxes[ii, 2]:.2f}'
        draw.text((x1 + 1, y1 + 2), bbox_text, fill=(0, 255, 0))

    plt.imshow(np.array(I_target), vmin=0, vmax=1)
    plt.axis("off")
    plt.show()

# ----- Visualization Functions -----

if __name__=='__main__':

    # ----- HOG ----- #

    image = Image.open('cameraman.tif')
    image_array = np.array(image)
    hog = extract_hog(image_array)
    visualize_hog(image_array, hog, 8, 2)

    # ----- Face Detection ----- #

    I_target = Image.open('target.png')
    I_target_array = np.array(I_target.convert('L'))

    I_template = Image.open('template.png')
    I_template_array = np.array(I_template.convert('L'))

    bounding_boxes=face_detection(I_target_array, I_template_array)
    visualize_face_detection(I_target, bounding_boxes, I_template_array.shape)