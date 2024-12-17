#导入工具包
import cv2
import numpy as np
from matplotlib import pyplot as plt

#滤波器
def butterworth_bandstop_kernel(img,D0,W,n=1,type='pass'):
    assert img.ndim == 2
    r,c = img.shape[1],img.shape[0]
    u = np.arange(r)
    v = np.arange(c)
    u, v = np.meshgrid(u, v) #生成网络点坐标矩阵
    low_pass = np.sqrt( (u-r/2)**2 + (v-c/2)**2 ) #相当于公式里的D(u,v),距频谱图矩阵中中心的距离
    kernel = (1/(
            1+((low_pass*W)/(low_pass**2-D0**2))**(2*n))) #变换公式
    if type == "pass":
        kernel=1-kernel
    visualization=1#是否可视化滤波器
    if visualization:
        visual=np.ones_like(img)*255*kernel
        cv2.imwrite('output/kernel.jpg', visual)
    return kernel

def butterworth_bandpass_filter(img,D0,W,n):
    assert img.ndim == 2
    kernel = butterworth_bandstop_kernel(img,D0,W,n)  #得到滤波器
    gray = np.float64(img)  #将灰度图片转换为opencv官方规定的格式
    # gray_fft = np.fft.fft2(gray) #傅里叶变换
    gray_fft = cv2.dct(gray) #傅里叶变换
    gray_fftshift = np.fft.fftshift(gray_fft) #将频谱图低频部分转到中间位置
    #dst = np.zeros_like(gray_fftshift)
    dst_filtered = kernel * gray_fftshift #频谱图和滤波器相乘得到新的频谱图
    dst_ifftshift = np.fft.ifftshift(dst_filtered) #将频谱图的中心移到左上方
    # dst_ifft = np.fft.ifft2(dst_ifftshift) #傅里叶逆变换
    dst_ifft = cv2.idct(dst_ifftshift) #傅里叶逆变换
    dst = np.abs(np.real(dst_ifft))
    dst = np.clip(dst,0,255)
    return np.uint8(dst)

def split_merge(img,D0,W,n):
    b, g, r = cv2.split(img)
    b=butterworth_bandpass_filter(b, D0, W, n)
    g=butterworth_bandpass_filter(g, D0, W, n)
    r=butterworth_bandpass_filter(r, D0, W, n)
    merged_img = cv2.merge([r, g, b])
    return merged_img

if __name__=="__main__":
    # 读取图像dage
    img = cv2.imread('/home/underwater/Code/Data/DeepfishDomain/images/7117_Caranx_sexfasciatus_juvenile_f000000.jpg')
    gray_img = cv2.imread('/home/underwater/Code/Data/DeepfishDomain/images/7117_Caranx_sexfasciatus_juvenile_f000000.jpg',0)
    see=butterworth_bandpass_filter(gray_img, D0=5, W=10, n=2)
    cv2.imwrite(f'output/see.jpg', see)
    # black_img0 = np.zeros_like(img)
    # # 得到处理后的图像
    # new_image1 = split_merge(img, D0=6, W=10, n=2)
    # black_img1 = split_merge(black_img0, D0=6, W=10, n=2)
    # new_image2 = split_merge(img, D0=15, W=10, n=2)
    # black_img2 = split_merge(black_img0, D0=15, W=10, n=2)
    # new_image3 = split_merge(img, D0=15, W=10, n=2)
    # black_img3 = split_merge(black_img0, D0=25, W=10, n=2)
    #
    # # 显示原始图像和带通滤波处理后的图像
    # images = [img, new_image1, new_image2, new_image3]
    # black_images = [black_img0, black_img1, black_img2, black_img3]
    # for i in np.arange(4):
    #     cv2.imwrite(f'output/image{i}.jpg', images[i])
    #     cv2.imwrite(f'output/black_image{i}.jpg', black_images[i])

