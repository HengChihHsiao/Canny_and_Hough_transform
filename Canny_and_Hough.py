import cv2
import os
import numpy as np

def show_img(title: str, img: np.ndarray) -> None:
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rgb2gray(rgb_image: np.ndarray) -> np.ndarray:
    # rgb_image: image to be processed
    w, h = rgb_image.shape[0], rgb_image.shape[1]
    gray_image = np.zeros((w, h), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            gray_image[i][j] = int(rgb_image[i][j][0]*0.299 + rgb_image[i][j][1]*0.587 + rgb_image[i][j][2]*0.114)
    
    return gray_image

def gaussian_blur(gray_img: np.ndarray, kernal_size: int) -> np.ndarray:
    w, h = gray_img.shape
    gaussian_blur_img = np.zeros((w, h), dtype=np.uint8)
    kernal = np.zeros((kernal_size, kernal_size))
    for i in range(kernal_size):
        for j in range(kernal_size):
            kernal[i][j] = 1 / (kernal_size * kernal_size)

    for i in range(w):
        for j in range(h):
            if i < kernal_size // 2 or i >= w - kernal_size // 2 or j < kernal_size // 2 or j >= h - kernal_size // 2:
                gaussian_blur_img[i][j] = gray_img[i][j]
            else:
                gaussian_blur_img[i][j] = np.sum(gray_img[i - kernal_size // 2 : i + kernal_size // 2 + 1, j - kernal_size // 2 : j + kernal_size // 2 + 1] * kernal)
    return gaussian_blur_img


class CannyEdgeDetector:
    def __init__(self):
        self.gray_img = None
        self.noise_reduction_img = None
        self.gradient_calc_img = None
        self.theta = None
        self.non_maximum_suppression_img = None
        self.double_threshold_img = None
        self.weak = None
        self.strong = None
        self.edge_tracking_img = None
        self.canny_result_img = None

    def noise_reduction(self, gray_img: np.ndarray) -> np.ndarray:
        noise_reduction_img = gaussian_blur(gray_img, 5)
        # self.noise_reduction_img = noise_reduction_img
        return noise_reduction_img


    def sobel_filter(self, noise_reduction_img: np.ndarray) -> np.ndarray:

        sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_filter_img = np.zeros(noise_reduction_img.shape, dtype="uint8")
        theta = np.zeros(noise_reduction_img.shape, dtype="float")
        w, h = noise_reduction_img.shape
        
        for i in range(w - 2):
            for j in range(h - 2):
                gx = np.sum(np.multiply(sobel_filter_x, noise_reduction_img[i:i + 3, j:j + 3]))  # x direction
                gy = np.sum(np.multiply(sobel_filter_y, noise_reduction_img[i:i + 3, j:j + 3]))  # y direction
                sobel_filter_img[i + 1, j + 1] = np.abs(gx) + np.abs(gy)

        for i in range(1,w - 2):
            for j in range(1, h - 2):
                if (((theta[i, j] >= -22.5) and (theta[i, j]< 22.5)) or ((theta[i, j] <= -157.5) and (theta[i, j] >= -180)) or ((theta[i, j] >= 157.5) and (theta[i, j] < 180))):
                    theta[i, j] = 0.0
                elif (((theta[i, j] >= 22.5) and (theta[i, j]< 67.5)) or ((theta[i, j] <= -112.5) and (theta[i, j] >= -157.5))):
                    theta[i, j] = 45.0
                elif (((theta[i, j] >= 67.5) and (theta[i, j]< 112.5)) or ((theta[i, j] <= -67.5) and (theta[i, j] >= -112.5))):
                    theta[i, j] = 90.0
                elif (((theta[i, j] >= 112.5) and (theta[i, j]< 157.5)) or ((theta[i, j] <= -22.5) and (theta[i, j] >= -67.5))):
                    theta[i, j] = -45.0

        return sobel_filter_img, theta

    def gradient_calc(self, noise_reduction_img: np.ndarray) -> np.ndarray:
        gradient_calc_img, theta = self.sobel_filter(noise_reduction_img)
        return gradient_calc_img, theta
    
    def non_maximum_suppression(self, gradient_calc_img: np.ndarray, theta: int) -> np.ndarray:

        non_maximum_suppression_img = np.zeros(gradient_calc_img.shape)
        for i in range(1,non_maximum_suppression_img.shape[0] - 1):
            for j in range(1,non_maximum_suppression_img.shape[1] - 1):
                if (theta[i, j] == 0.0) and (gradient_calc_img[i, j] == np.max([gradient_calc_img[i, j],gradient_calc_img[i+1, j],gradient_calc_img[i-1, j]]) ):
                        non_maximum_suppression_img[i, j] = gradient_calc_img[i, j]

                if (theta[i, j] == -45.0) and gradient_calc_img[i, j] == np.max([gradient_calc_img[i, j],gradient_calc_img[i-1, j-1],gradient_calc_img[i+1, j+1]]):
                        non_maximum_suppression_img[i, j] = gradient_calc_img[i, j]

                if (theta[i, j] == 90.0) and  gradient_calc_img[i, j] == np.max([gradient_calc_img[i, j],gradient_calc_img[i, j+1],gradient_calc_img[i, j-1]]):
                        non_maximum_suppression_img[i, j] = gradient_calc_img[i, j]

                if (theta[i, j] == 45.0) and gradient_calc_img[i, j] == np.max([gradient_calc_img[i, j],gradient_calc_img[i-1, j+1],gradient_calc_img[i+1, j-1]]):
                        non_maximum_suppression_img[i, j] = gradient_calc_img[i, j]
        return non_maximum_suppression_img
    
    def double_threshold(self, non_maximum_suppression_img: np.ndarray, high_threshold: float, low_threshold: float) -> np.ndarray:

        print("High Threshold: ", high_threshold)
        print("Low Threshold: ", low_threshold)
        w, h = non_maximum_suppression_img.shape
        double_threshold_img = np.zeros((w, h), dtype=np.uint8)

        weak = np.uint8(25)
        strong = np.uint8(255)

        strong_i, strong_j = np.where(non_maximum_suppression_img >= high_threshold)
        weak_i, weak_j = np.where((non_maximum_suppression_img <= high_threshold) & (non_maximum_suppression_img >= low_threshold))

        double_threshold_img[strong_i, strong_j] = strong
        double_threshold_img[weak_i, weak_j] = weak

        return (double_threshold_img, weak, strong)
    
    def edge_tracking(self, double_threshold_img: np.ndarray, weak: int, strong: int) -> np.ndarray:
        w, h = double_threshold_img.shape
        for i in range(1, w - 1):
            for j in range(1, h - 1):
                if(double_threshold_img[i, j] == weak):
                    try:
                        if((double_threshold_img[i+1, j-1] == strong) or (double_threshold_img[i+1, j] == strong)
                            or (double_threshold_img[i+1, j+1] == strong) or (double_threshold_img[i, j-1] == strong) 
                            or (double_threshold_img[i, j+1] == strong) or (double_threshold_img[i-1, j-1] == strong) 
                            or (double_threshold_img[i-1, j] == strong) or (double_threshold_img[i-1, j+1] == strong)):
                            double_threshold_img[i, j] = strong

                        else:
                            double_threshold_img[i, j] = 0
                    except IndexError as e:
                        pass
        return double_threshold_img
    
    def get_canny_result(self, gaussian_blur_img: np.ndarray, high_threshold: int, low_threshold: int) -> np.ndarray:
        gradient_calc_img, theta = self.gradient_calc(gaussian_blur_img)
        # show_img('gradient_calc_img', gradient_calc_img)
        non_maximum_suppression_img = self.non_maximum_suppression(gradient_calc_img, theta)
        # show_img('non_maximum_suppression_img', non_maximum_suppression_img)  
        double_threshold_img, weak, strong = self.double_threshold(non_maximum_suppression_img, high_threshold, low_threshold)
        # show_img('double_threshold_img', double_threshold_img)
        canny_result_img = self.edge_tracking(double_threshold_img, weak, strong)
        return canny_result_img

class HoughTransform:
    def __init__(self):
        pass
    
    def hough(self, canny_result_img: np.ndarray) -> np.ndarray:
        w, h = canny_result_img.shape               
        diag_len = int(np.ceil(np.sqrt(w * w + h * h)))
        rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
        thetas = np.deg2rad(np.arange(0.0, 180.0, 1.0))
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        num_thetas = len(thetas)

        accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
        y_idxs, x_idxs = np.nonzero(canny_result_img)

        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            for t_idx in range(num_thetas):
                rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
                accumulator[rho, t_idx] += 1
        
        return accumulator, thetas, rhos

    def draw_hough_line(self, img: np.ndarray, accumulator: np.ndarray, thetas: np.ndarray, rhos: np.ndarray) -> np.ndarray:
        for i in range(len(thetas)):
            for j in range(len(rhos)):
                if(accumulator[j, i] > 150):
                    rho = rhos[j]
                    theta = thetas[i]

                    a = np.cos(theta)
                    b = np.sin(theta)

                    x0 = a * rho
                    y0 = b * rho

                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return img
    
    def get_hough_result(self, img: np.ndarray, canny_result_img: np.ndarray) -> np.ndarray:
        accumulator, thetas, rhos = self.hough(canny_result_img)
        hough_result_img = self.draw_hough_line(img, accumulator, thetas, rhos)
        return hough_result_img



if __name__ == '__main__':
    img_list = os.listdir('test_img')

    for img_name in img_list:
        img = cv2.imread('test_img/' + img_name)
        gray_img = rgb2gray(img)
        ##### Q1 Gaussian Blur #####
        gaussian_blur_img = gaussian_blur(gray_img, 3)
        cv2.imwrite('result_img/Blur_' + img_name, gaussian_blur_img)
        show_img('gaussian_blur_img', gaussian_blur_img)

        ##### Q2 Canny Edge Detector #####
        canny_result_img = CannyEdgeDetector().get_canny_result(gaussian_blur_img, 100, 30)
        cv2.imwrite('result_img/Canny_' + img_name, canny_result_img)
        show_img('canny_result_img', canny_result_img)

        ##### Q3 Hough Transform #####
        hough_transform_img = HoughTransform().get_hough_result(img, canny_result_img)
        cv2.imwrite('result_img/Hough_' + img_name, hough_transform_img)
        show_img('hough_transform_img', hough_transform_img)        

        cv2.destroyAllWindows()

