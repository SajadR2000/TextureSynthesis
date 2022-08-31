import cv2
import numpy as np
import matplotlib.pyplot as plt


def minimum_cut_path(overlapping_1, overlapping_2, cut_shape, max_allowable_movement):
    """
    Takes two overlapping parts and finds the minimum cut path
    :param overlapping_1: overlapping part of left or up patch
    :param overlapping_2: overlapping part of right or down patch
    :param cut_shape: vertical or horizontal cut
    :param max_allowable_movement: neighbours
    :return: minimum cut path indices
    """
    if cut_shape == 'horizontal_cut':
        # For horizontal minimum cut path, first transpose the matrices, then do a vertical minimum cut path
        overlapping_1 = overlapping_1.transpose((1, 0, 2))
        overlapping_2 = overlapping_2.transpose((1, 0, 2))
    overlapping_1 = overlapping_1.reshape((overlapping_1.shape[0], overlapping_1.shape[1], -1))
    overlapping_2 = overlapping_2.reshape((overlapping_2.shape[0], overlapping_2.shape[1], -1))
    cost = np.square(overlapping_1 - overlapping_2).sum(axis=2)

    # Vertical Minimum Cut Path
    path_cost = {"Path": [[i] for i in range(overlapping_1.shape[1])],
                 "Cost": [cost[0, i] for i in range(overlapping_1.shape[1])]}

    for i in range(1, overlapping_1.shape[0]):
        path_cost_temp = {"Path": [[] for i in range(overlapping_1.shape[1])],  # Resetting path_cost_temp
                          "Cost": [0 for i in range(overlapping_1.shape[1])]}

        for j in range(overlapping_1.shape[1]):
            if j < max_allowable_movement:
                neighbors_cost = path_cost["Cost"][:j+max_allowable_movement + 1]
                ii = np.argmin(neighbors_cost)
                path_cost_temp["Path"][j] = path_cost["Path"][ii] + [j]
                path_cost_temp["Cost"][j] = path_cost["Cost"][ii] + cost[i, j]
            elif max_allowable_movement <= j < overlapping_1.shape[1] - max_allowable_movement:
                neighbors_cost = path_cost["Cost"][j-max_allowable_movement:j + max_allowable_movement + 1]
                ii = np.argmin(neighbors_cost) + j - max_allowable_movement
                path_cost_temp["Path"][j] = path_cost["Path"][ii] + [j]
                path_cost_temp["Cost"][j] = path_cost["Cost"][ii] + cost[i, j]
            else:
                neighbors_cost = path_cost["Cost"][j - max_allowable_movement:]
                ii = np.argmin(neighbors_cost) + j - max_allowable_movement
                path_cost_temp["Path"][j] = path_cost["Path"][ii] + [j]
                path_cost_temp["Cost"][j] = path_cost["Cost"][ii] + cost[i, j]
        path_cost = path_cost_temp  # Copy doesn't work for complex data types. you can use copy.deepcopy(). I reset it.

    # print(path_cost)
    min_cost_idx = np.argmin(path_cost["Cost"])
    return path_cost["Path"][min_cost_idx]


def find_matching_patch_one_direction(texture, patch_overlapping_part, patch_shape):
    """
    finds next patch
    :param texture: the source texture
    :param patch_overlapping_part: overlapping part of the previous patch
    :param patch_shape: patch shape
    :return: next patch
    """
    number_of_best_matches = 15  # adding randomness
    # Calculating SSD
    ssd = cv2.matchTemplate(texture, patch_overlapping_part, cv2.TM_SQDIFF)[:-patch_shape[0], :-patch_shape[1]]
    ssd_flatten = ssd.ravel()
    ssd_argsort = np.argsort(ssd_flatten)  # sort from min to max
    # Choosing best matches
    chosen_points = ssd_argsort[:number_of_best_matches]
    # Calculating probabilities using soft-max
    chosen_points_prob = np.exp(-np.square(ssd_flatten[chosen_points]) /
                                np.square(ssd_flatten[chosen_points]).max())
    chosen_points_prob = chosen_points_prob / chosen_points_prob.sum()
    random_chosen_point = np.random.choice(chosen_points, 1, False, p=chosen_points_prob)[0]
    # converting index to x, y
    chosen_point_x = random_chosen_point // ssd.shape[1]
    chosen_point_y = np.mod(random_chosen_point, ssd.shape[1])
    out = texture[chosen_point_x:chosen_point_x + patch_shape[0], chosen_point_y:chosen_point_y + patch_shape[1]]
    return out


def find_matching_patch_L_shape(texture, patch_overlapping_horizontal, patch_overlapping_vertical, patch_shape):
    """
    finding best matches for L shape cuts
    :param texture: source
    :param patch_overlapping_horizontal: overlapping part of up patch
    :param patch_overlapping_vertical: overlapping part of left patch
    :param patch_shape: patch shape
    :return:
    """
    number_of_best_matches = 15
    ssd_horizontal = cv2.matchTemplate(texture, patch_overlapping_horizontal, cv2.TM_SQDIFF)
    ssd_vertical = cv2.matchTemplate(texture, patch_overlapping_vertical, cv2.TM_SQDIFF)
    ssd_horizontal = ssd_horizontal[:-(patch_shape[0] - patch_overlapping_horizontal.shape[0]), :]
    ssd_vertical = ssd_vertical[:, :-(patch_shape[1] - patch_overlapping_vertical.shape[1])]
    ssd = ssd_vertical + ssd_horizontal  # Adding left and up ssd
    ssd_flatten = ssd.ravel()  # matrix to vector
    ssd_argsort = np.argsort(ssd_flatten)  # sort from min to max
    chosen_points = ssd_argsort[:number_of_best_matches]
    chosen_points_prob = np.exp(-np.square(ssd_flatten[chosen_points]) /
                                np.square(ssd_flatten[chosen_points]).max())
    chosen_points_prob = chosen_points_prob / chosen_points_prob.sum()
    random_chosen_point = np.random.choice(chosen_points, 1, False, p=chosen_points_prob)[0]
    chosen_point_x = random_chosen_point // ssd.shape[1]
    chosen_point_y = np.mod(random_chosen_point, ssd.shape[1])
    out = texture[chosen_point_x:chosen_point_x + patch_shape[0], chosen_point_y:chosen_point_y + patch_shape[1]]
    return out


def merging(left_patch, top_patch, next_patch, cut_shape, overlapping_factor, max_allowable_movement):
    """
    Seperates overlapping and nonoverlapping parts and does the cuts for overlapping parts
    :param left_patch: left_patch
    :param top_patch: top_patch
    :param next_patch: next_patch
    :param cut_shape: vertical, horizontal or L shape cut
    :param overlapping_factor: overlapping_factor
    :param max_allowable_movement: max_allowable_movement
    :return: separated parts
    """
    if cut_shape == 'L_shape_cut':
        horizontal_overlapping_up = top_patch[-int(overlapping_factor*top_patch.shape[0]):, :]
        horizontal_overlapping_down = next_patch[:int(overlapping_factor*next_patch.shape[0]), :]

        vertical_overlapping_left = left_patch[:, -int(overlapping_factor*left_patch.shape[1]):]
        vertical_overlapping_right = next_patch[:, :int(overlapping_factor*next_patch.shape[1])]

        horizontal_cut = minimum_cut_path(horizontal_overlapping_up,
                                          horizontal_overlapping_down,
                                          'horizontal_cut',
                                          max_allowable_movement)

        vertical_cut = minimum_cut_path(vertical_overlapping_left,
                                        vertical_overlapping_right,
                                        'vertical_cut',
                                        max_allowable_movement)

        horizontal_overlapped = np.zeros(horizontal_overlapping_up.shape, dtype=np.uint8)
        # Assign above the cut to the up patch and below the cut to the next patch
        for i in range(horizontal_overlapped.shape[1]):
            horizontal_overlapped[:horizontal_cut[i], i] = horizontal_overlapping_up[:horizontal_cut[i], i]
            horizontal_overlapped[horizontal_cut[i]:, i] = horizontal_overlapping_down[horizontal_cut[i]:, i]
        # Correct the pixels on the left of the vertical cut
        for i in range(horizontal_overlapped.shape[0]):
            horizontal_overlapped[i, :vertical_cut[i]] = horizontal_overlapping_up[i, :vertical_cut[i]]

        vertical_overlapped = np.zeros(vertical_overlapping_left.shape, dtype=np.uint8)
        # Assign left of the cut to the left patch and right of the cut to the next patch
        for i in range(vertical_overlapped.shape[0]):
            vertical_overlapped[i, :vertical_cut[i]] = vertical_overlapping_left[i, :vertical_cut[i]]
            vertical_overlapped[i, vertical_cut[i]:] = vertical_overlapping_right[i, vertical_cut[i]:]
        # Correct the pixels above the horizontal cut
        for i in range(vertical_overlapped.shape[1]):
            vertical_overlapped[:horizontal_cut[i], i] = vertical_overlapping_left[:horizontal_cut[i], i]

        non_overlapping_left = left_patch[:, :-vertical_overlapped.shape[1]]
        non_overlapping_top = top_patch[:-horizontal_overlapped.shape[0], :]
        non_overlapping_next = next_patch[horizontal_overlapped.shape[0]:, vertical_overlapped.shape[1]:]
        return non_overlapping_left, non_overlapping_top, non_overlapping_next, vertical_overlapped,\
               horizontal_overlapped
    else:
        if cut_shape == 'horizontal_cut':
            patch1 = top_patch.transpose((1, 0, 2))
            patch2 = next_patch.transpose((1, 0, 2))
        elif cut_shape == 'vertical_cut':
            patch1 = left_patch
            patch2 = next_patch
        else:
            raise Exception("Cut shape not defined!")

        overlapping1 = patch1[:, -int(patch1.shape[1] * overlapping_factor):]
        overlapping2 = patch2[:, :int(patch2.shape[1] * overlapping_factor)]
        path = minimum_cut_path(overlapping1, overlapping2, 'vertical_cut', max_allowable_movement)

        overlapped = np.zeros(overlapping1.shape, dtype=np.uint8)
        for i in range(overlapped.shape[0]):
            overlapped[i, :path[i]] = overlapping1[i, :path[i]]
            overlapped[i, path[i]:] = overlapping2[i, path[i]:]

        non_overlapping1 = patch1[:, :-int(patch1.shape[1] * overlapping_factor)]
        non_overlapping2 = patch2[:, int(patch2.shape[1] * overlapping_factor):]

        if cut_shape == 'horizontal_cut':
            overlapped = overlapped.transpose((1, 0, 2))
            non_overlapping1 = non_overlapping1.transpose((1, 0, 2))
            non_overlapping2 = non_overlapping2.transpose((1, 0, 2))

        return non_overlapping1, overlapped, non_overlapping2


def texture_synthesis(texture, final_shape, patch_shape, overlapping_factor, max_allowable_movement=4):
    # Initial patch
    initial_patch_x = np.random.choice(np.arange(texture.shape[0] - patch_shape[0]), 1)[0]
    initial_patch_y = np.random.choice(np.arange(texture.shape[1] - patch_shape[1]), 1)[0]
    initial_patch = texture[initial_patch_x:initial_patch_x+patch_shape[0],
                            initial_patch_y:initial_patch_y+patch_shape[1]]
    # overlapping part of the initial patch
    overlapping = initial_patch[:, -int(overlapping_factor * patch_shape[1]):].copy()
    # next_patch
    next_patch = find_matching_patch_one_direction(texture.copy(), overlapping.copy(), patch_shape).copy()
    first_row, overlapped, non_overlapping2 = merging(initial_patch.copy(),
                                                      None,
                                                      next_patch.copy(),
                                                      'vertical_cut',
                                                      overlapping_factor,
                                                      max_allowable_movement)

    vertical_overlap_width = int(overlapping_factor * patch_shape[1])
    number_of_cols = (final_shape[1] - vertical_overlap_width) // (patch_shape[1] - vertical_overlap_width) + 1
    # while number_of_cols * (patch_shape[1] - vertical_overlap_width) + vertical_overlap_width <= final_shape[1]:
    #     number_of_cols += 1

    # Constructing first row as explained in the report
    for _ in range(1, number_of_cols):
        last_patch = np.concatenate((overlapped, non_overlapping2), axis=1)
        overlapping = last_patch[:, -int(overlapping_factor * patch_shape[1]):].copy()
        next_patch = find_matching_patch_one_direction(texture.copy(), overlapping, patch_shape)
        non_overlapping1, overlapped, non_overlapping2 = merging(last_patch,
                                                                 None,
                                                                 next_patch,
                                                                 'vertical_cut',
                                                                 overlapping_factor,
                                                                 max_allowable_movement)
        first_row = np.concatenate((first_row, non_overlapping1), axis=1)
    first_row = np.concatenate((first_row, overlapping), axis=1)
    ###########################################################################
    final = first_row[:-int(overlapping_factor * patch_shape[0]), :].copy()
    last_row = first_row.copy()
    horizontal_overlap_height = int(overlapping_factor * patch_shape[0])
    number_of_rows = (final_shape[0] - horizontal_overlap_height) // (patch_shape[0] - horizontal_overlap_height) + 1
    # while number_of_rows * (patch_shape[0] - horizontal_overlap_height) + horizontal_overlap_height < final_shape[0]:
    #     number_of_rows += 1

    # Completing next rows
    for j in range(1, number_of_rows):
        # first patch of next row
        upper_patch = last_row[:, :patch_shape[1]]
        overlapping = upper_patch[-int(overlapping_factor * patch_shape[0]):, :]
        next_patch = find_matching_patch_one_direction(texture, overlapping, patch_shape)
        _, overlapping_h, non_overlapping2 = merging(None,
                                                     upper_patch,
                                                     next_patch,
                                                     'horizontal_cut',
                                                     overlapping_factor,
                                                     max_allowable_movement)
        last_row[-overlapping_h.shape[0]:, :overlapping_h.shape[1]] = overlapping_h.copy()
        next_row = np.concatenate((overlapping_h, non_overlapping2), axis=0)
        next_patch = next_row.copy()
        next_row = next_row[:, :-int(overlapping_factor*patch_shape[1])]
        # Other patches of next row
        for i in range(1, number_of_cols):
            left_patch = next_patch.copy()
            current_top_patch_start = i*(patch_shape[1] - int(overlapping_factor * patch_shape[1]))
            top_patch = last_row[:, current_top_patch_start:current_top_patch_start + patch_shape[1]]
            vertical_overlapping_part = left_patch[:, -int(patch_shape[1]*overlapping_factor):]
            horizontal_overlapping_part = top_patch[-int(patch_shape[0]*overlapping_factor):, :]
            next_patch = find_matching_patch_L_shape(texture.copy(),
                                                     horizontal_overlapping_part,
                                                     vertical_overlapping_part,
                                                     patch_shape)
            non_overlapping_left, non_overlapping_top, non_overlapping_next, vertical_overlapped, horizontal_overlapped\
                = merging(left_patch, top_patch, next_patch, 'L_shape_cut', overlapping_factor, max_allowable_movement)

            next_patch = np.concatenate(
                (horizontal_overlapped[:, vertical_overlapped.shape[1]:], non_overlapping_next), axis=0)
            next_patch = np.concatenate((vertical_overlapped, next_patch), axis=1)
            next_row = np.concatenate((next_row, next_patch[:, :-int(overlapping_factor*patch_shape[1])]), axis=1)
            last_row[-horizontal_overlapped.shape[0]:,
                     current_top_patch_start:current_top_patch_start + patch_shape[1]] = horizontal_overlapped
        next_row = np.concatenate((next_row, next_patch[:, -int(overlapping_factor*patch_shape[1]):]), axis=1)
        last_row = next_row.copy()
        final = np.concatenate((final, next_row[:-int(overlapping_factor*patch_shape[0]), :]), axis=0)
        if j == number_of_rows - 1:
            final = np.concatenate((final, next_row[-int(overlapping_factor*patch_shape[0]):, :]), axis=0)
    return final[:final_shape[0], :final_shape[1]]


def run(img_dir, final_shape=(2500, 2500), patch_factor=0.35, overlapping_factor=0.25):
    """
    runs the whole program
    :param img_dir: directory of the image
    :param final_shape: final shape of the output image
    :param patch_factor: determines shape of the patches based on size of the input texture
    :param overlapping_factor: determines shape of the overlapping part based on the shape of patches
    :return:
    """
    texture = read_img(img_dir)
    h, w = texture.shape[0], texture.shape[1]
    texture = texture.reshape((h, w, -1))
    patch_shape = (int(patch_factor * h), int(patch_factor * w))
    synthesized = texture_synthesis(texture.copy(), final_shape, patch_shape, overlapping_factor)
    final = final_output(texture, synthesized)
    return final


def read_img(img_dir):
    # loads the image
    img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise Exception("Couldn't load the image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def final_output(texture, synthesized):
    """
    Adds the source texture to synthesized image
    :param texture: source texture
    :param synthesized: synthesized image
    :return: final output
    """
    if texture.shape[0] > synthesized.shape[0]:
        raise Exception("Not implemented! Texture should be smaller than final output")

    temp = np.zeros((synthesized.shape[0], texture.shape[1] + 20, texture.shape[2]), dtype=np.uint8)
    temp[:texture.shape[0], :texture.shape[1], :] = texture
    out = np.concatenate((temp, synthesized), axis=1)
    return out


out1 = run('./Textures/texture01.jpg')
plt.imsave('res11.jpg', out1)
out2 = run('./Textures/texture02.png')
plt.imsave('res12.jpg', out2)
out3 = run('./my_texture1.jpg')
plt.imsave('res13.jpg', out3)
out4 = run('./my_texture2.jpg')
plt.imsave('res14.jpg', out4)

# testing = np.sqrt(np.array([[1, 3, 4, 1], [2, 1, 2, 3], [4, 2, 1, 4]])).reshape((3, 4, 1))
# print(minimum_cut_path(testing, np.zeros((3, 4, 1)), 'horizontal_cut', 1))
