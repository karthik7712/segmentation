import math
from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score, confusion_matrix, classification_report
import cv2

frequencies = {}
gcd_values = set()
thresholds = {}
results = []

def convert_pgm_to_jpg(pgm_file_path, jpg_file_path):
    """Convert a PGM image to JPG format using OpenCV."""
    # Read the image in grayscale
    image = cv2.imread(pgm_file_path, flags=0)
    # Save the image in JPG format
    cv2.imwrite(jpg_file_path, image)

# Example usage
for x in range(1,323):
    if x < 10:
        s = f'00{x}'
    else:
        if x<100:
            s = f"0{x}"
        else:
            s = f'{x}'
    pgm_path = f"./mias/mdb{s}.pgm"
    jpg_path = f"images/mias{x}.jpg"
    # print(x)
    convert_pgm_to_jpg(pgm_path, jpg_path)


def calculate_metrics(true_image, predicted_image):
    if len(true_image.shape) == 3:
        true_image = cv2.cvtColor(true_image, cv2.COLOR_BGR2GRAY)
    if len(predicted_image.shape) == 3:
        predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2GRAY)

    true_image = cv2.resize(true_image, (predicted_image.shape[1], predicted_image.shape[0]))

    _, true_image = cv2.threshold(true_image, 127, 1, cv2.THRESH_BINARY)
    _, predicted_image = cv2.threshold(predicted_image, 127, 1, cv2.THRESH_BINARY)


    true_image = true_image.flatten()
    predicted_image = predicted_image.flatten()


    iou = jaccard_score(true_image, predicted_image, average='binary')

    report = classification_report(true_image, predicted_image, target_names=['background', 'object'])
    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_image, predicted_image)

    return iou, report, conf_matrix


def divide(image_path, type):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape[:2]
    half_height = height // 2
    half_width = width // 2

    part1 = np.array(gray_image[:half_height, :half_width])
    part2 = np.array(gray_image[:half_height, half_width:])
    part3 = np.array(gray_image[half_height:, :half_width])
    part4 = np.array(gray_image[half_height:, half_width:])

    thres1 = np.mean(list(get_gcds(part1.flatten())))
    thres2 = np.mean(list(get_gcds(part2.flatten())))
    thres3 = np.mean(list(get_gcds(part3.flatten())))
    thres4 = np.mean(list(get_gcds(part4.flatten())))

    # print("Four thresholds are: ",thres1," ",thres2," ",thres3," ",thres4)


    im1_segment = segmentation(gray_image, type, thres1)
    im2_segment = segmentation(gray_image, type, thres2)
    im3_segment = segmentation(gray_image, type, thres3)
    im4_segment = segmentation(gray_image, type, thres4)

    cv2.imwrite(f'{type}_first_segmented_{image_path}', im1_segment)
    cv2.imwrite(f'{type}_second_segmented_{image_path}', im2_segment)
    cv2.imwrite(f'{type}_third_segmented_{image_path}', im3_segment)
    cv2.imwrite(f'{type}_fourth_segmented_{image_path}', im4_segment)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_gcds(intensities):
    gcds = set()
    for i in range(len(intensities)):
        if i == len(intensities)-1:
            continue
        x, y = intensities[i], intensities[i+1]
        i+=1
        g = math.gcd(x, y)
        if g != 1:
            gcds.add(g)
    # print(np.mean(list(gcds)))
    return gcds

def select_numbers_with_least_variation(numbers, k):
    numbers.sort()
    n = len(numbers)
    interval = n // (k + 1)

    selected_numbers = []
    for i in range(1, k + 1):
        index = i * interval
        selected_numbers.append(numbers[index])

    return selected_numbers


def find_gcd_vectorized(arr):
    gcd_vectorized = np.frompyfunc(math.gcd, 2, 1)
    return gcd_vectorized.reduce(arr, axis=0)


def find_filter1():
    global frequencies
    top_gcd = []
    count = 0
    for key in frequencies.keys():
        if count<9:
            top_gcd.append(key)
            count += 1
        else:
            break

    matrix = [top_gcd[i:i + 3] for i in range(0, len(top_gcd), 3)]

    vals = np.array(matrix)
    x = np.sum(vals)
    filter_3x3 = vals / x
    # print("The filter is :\n", filter_3x3)
    return filter_3x3


def smooth1(image_path,filter):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_array = np.array(gray_image)
    filtered_image = np.zeros_like(img_array)

    for y in range(1, img_array.shape[0] - 1):
        for x in range(1, img_array.shape[1] - 1):
            filtered_image[y, x] = round(np.sum(img_array[y - 1:y + 2, x - 1:x + 2] * filter), 0)

    filtered_image = np.clip(filtered_image, 0, 255)

    smoothed_image = Image.fromarray(filtered_image.astype(np.uint8))
    smoothed_image.save(f"smoothed_1_{image_path}")


def find_filter2():
    global gcd_values
    top_gcd2 = select_numbers_with_least_variation(list(gcd_values),9)
    # print(top_gcd2)
    filter_matrix2 = np.array(top_gcd2).reshape(3, 3) / np.sum(top_gcd2)
    # print("The second filter is :\n",filter_matrix2)
    return filter_matrix2


def smooth2(image_path,filter):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_array = np.array(gray_image)
    padded_array = np.pad(gray_image.astype(np.float64),((1, 1), (1, 1)), mode='edge')

    smoothed_image = np.zeros_like(img_array, dtype=np.float64)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            window = padded_array[i:i + 3, j:j + 3]
            smoothed_image[i, j] = np.sum(window * filter)

    smoothed_image2 = np.clip(smoothed_image, 0, 255).astype(np.uint8)

    smoothed_image_2_pil = Image.fromarray(smoothed_image2)
    smoothed_image_2_pil.save(f'smoothed_2_{image_path}')


def gcd_threshold_segmentation(image_path):
    global thresholds
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    horizontal = []
    vertical = []
    spiral = []
    rows = len(gray_image)
    cols = len(gray_image[0])
    unique_values = np.unique(gray_image.flatten())
    top_row, bottom_row = 0, rows - 1
    left_col, right_col = 0, cols - 1

    # print("Intensity Values:",unique_values)

    thresholds["mean"] = np.mean(unique_values)
    thresholds["median"] = np.median(unique_values)
    # print("Mean of intensity values:",np.mean(unique_values))
    # print("Median of intensity values:",np.median(unique_values))
    #ALL PAIRS
    global frequencies
    for i in range(len(unique_values)):
        for j in range(i + 1, len(unique_values)):
            x, y = unique_values[i], unique_values[j]
            gcd = math.gcd(x, y)
            if(gcd!=1):
                if gcd in frequencies :
                    frequencies[gcd] += 1
                else:
                    frequencies[gcd] = 1
                gcd_values.add(gcd)
    frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))
    # print("Frequencies:",frequencies)
    # print("GCDs",gcd_values)

    #ITERATE HORIZONTALLY
    for row in range(rows):
        for col in range(cols):
            horizontal.append(gray_image[row][col])
    # horizontal = np.unique(horizontal)

    #ITERATE VERTICALLY
    for col in range(cols):
        for row in range(rows):
            vertical.append(gray_image[row][col])
    # vertical = np.unique(vertical)

    #ITERATE IN SPIRAL
    while top_row <= bottom_row and left_col <= right_col:
        for col in range(left_col, right_col + 1):
            spiral.append(gray_image[top_row][col])
        top_row += 1

        for row in range(top_row, bottom_row + 1):
            spiral.append(gray_image[row][right_col])
        right_col -= 1

        if top_row <= bottom_row:
            for col in range(right_col, left_col - 1, -1):
                spiral.append(gray_image[bottom_row][col])
            bottom_row -= 1

        if left_col <= right_col:
            for row in range(bottom_row, top_row - 1, -1):
                spiral.append(gray_image[row][left_col])
            left_col += 1
    # spiral = np.unique(spiral)

    #ITERATE DIAGONAL
    if rows<cols:
        main_diagonal = [gray_image[i][i] for i in range(rows)]
        anti_diagonal = [gray_image[i][rows-1-i] for i in range(rows)]
    else:
        main_diagonal = [gray_image[i][i] for i in range(cols)]
        anti_diagonal = [gray_image[i][cols-1-i] for i in range(cols)]

    horizontal_gcds = get_gcds(horizontal)
    vertical_gcds = get_gcds(vertical)
    spiral_gcds = get_gcds(spiral)
    diagonal_gcds = get_gcds(main_diagonal+anti_diagonal)

    mean_allvals = round(np.mean(list(gcd_values)),2)

    mean_horizontal = round(np.mean(list(horizontal_gcds)),2)
    median_horizontal = round(np.median(list(horizontal_gcds)),2)

    mean_vertical = round(np.mean(list(vertical_gcds)),2)
    median_vertical = round(np.median(list(vertical_gcds)),2)

    mean_spiral = round(np.mean(list(spiral_gcds)),2)
    median_spiral = round(np.median(list(spiral_gcds)),2)

    mean_diagonal = round(np.mean(list(diagonal_gcds)),2)
    median_diagonal = round(np.median(list(diagonal_gcds)),2)

    thresholds['allvalues'] = mean_allvals
    thresholds['horizontal'] = mean_horizontal
    thresholds['vertical'] = mean_vertical
    thresholds['spiral'] = mean_spiral
    thresholds['diagonal'] = mean_diagonal

    # print("Threshold values for different images:", thresholds)

    return thresholds

def determine_brightness(image):
    pass

def apply_filter_based_on_brightness(image):
    pass

def segmentation(gimage,type,T):
    global results
    m, n = gimage.shape
    img_thresh = np.zeros((m, n), dtype=np.uint8)

    for i in range(m):
        for j in range(n):
            if gimage[i, j] < T:
                img_thresh[i, j] = 0
            else:
                img_thresh[i, j] = 255
    if type != "parted":
        cv2.imwrite(f'{type}_segmented_{image_path}',img_thresh)
        results.append([img_thresh,type,T])
    return img_thresh.astype(np.uint8)


if __name__ == "__main__":
    # image_dict = {"mias": 322}
    # for name,num in image_dict.items():
    #     for x in range(1,num+1):
    #         image_path = f"images/{name}{x}.jpg"
    #         image = cv2.imread(image_path)
    #         filtered_image = apply_filter_based_on_brightness(image)
            # gimage = cv2.imread(image_path, 0)
            # threshold_values = gcd_threshold_segmentation(image_path)
            # filter1 = find_filter1()
            # filter2 = find_filter2()
            # smooth1(image_path,filter1)
            # smooth2(image_path,filter2)
            # for type,threshold_value in threshold_values.items():
            #     segmentation(gimage,type,int(threshold_value))
            # divide(image_path,"parted")

    #EXAMPLE
    image_path = f"images/BT_orig.png"
    image = cv2.imread(image_path)
    filtered_image = apply_filter_based_on_brightness(image)
    gimage = cv2.imread(image_path, 0)
    threshold_values = gcd_threshold_segmentation(image_path)
    filter1 = find_filter1()
    filter2 = find_filter2()
    smooth1(image_path, filter1)
    smooth2(image_path, filter2)
    print("Most common")
    print(filter1)
    print("Least Variant filter")
    print(filter2)
    for type, threshold_value in threshold_values.items():
        segmentation(gimage, type, int(threshold_value))
    divide(image_path, "parted")