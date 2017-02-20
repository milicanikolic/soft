import cv2
import numpy as np

def overlay_image(src, overlay, posx, posy):
    src_r, src_g, src_b = cv2.split(src)
    src_a = np.ones((src.shape[0], src.shape[1]), 'uint8')
    src_alpha = cv2.merge((src_r, src_g, src_b, src_a))

    rows, cols, channels = overlay.shape
    posx = posx - cols/2
    posy = posy - rows/2
    if posy<0:
        posy = 0
    roi = src_alpha[posy:rows+posy, posx:cols+posx]

    # Now create a mask of logo and create its inverse mask also
    # img2gray = cvtColor(overlay, COLOR_BGR2GRAY)
    r,g,b,a = cv2.split(overlay)
    ret, mask = cv2.threshold(a, 254, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(overlay, overlay, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.cvtColor(cv2.add(img1_bg, img2_fg), cv2.COLOR_BGRA2BGR)
    src[posy:rows + posy, posx:cols + posx] = dst
    return src




lice= cv2.imread("Lica/lik.png")
lice_gray= cv2.cvtColor(lice, cv2.COLOR_BGR2GRAY)

r = np.ones((lice.shape[0], lice.shape[1]), 'uint8')+254
a = np.zeros((lice.shape[0], lice.shape[1]), 'uint8')+180
g=np.zeros((lice.shape[0], lice.shape[1]), 'uint8')
b=np.zeros((lice.shape[0], lice.shape[1]), 'uint8')
crveniFilter = cv2.merge((g, b, r, a))
crveni=cv2.resize(crveniFilter,(int(lice.shape[1]*0.4),int(lice.shape[0]*0.4)))
izborani=cv2.Canny(lice_gray,25,65)

src_r, src_g, src_b = cv2.split(lice)
#src_a = np.ones((src.shape[0], src.shape[1]), 'uint8')
src_r1=src_r*10
src_alpha = cv2.merge((src_r1, src_g, src_b))

nova=overlay_image(lice,crveni,lice.shape[0]/2,lice.shape[1]/2)

print lice.shape
print lice[0].shape
cv2.imshow("cartoon", crveni)
cv2.imshow(cv2.waitKey(0))




