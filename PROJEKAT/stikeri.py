import cv2
import numpy as np
import dlib
import prepoznavanjeLica
import matplotlib.pyplot as plt
import math
from scipy import ndimage

def dodajCvike(slika,cvike, tackeLica):
    try:
        levo_oko = (tackeLica[36, 0], tackeLica[36, 1])
        desno_oko = (tackeLica[45, 0], tackeLica[45, 1])
        rastojanjeOci = rastojanje(levo_oko, desno_oko)
        koefOci = (np.float(levo_oko[1] - desno_oko[1]) / np.float(levo_oko[0] - desno_oko[0]))
        ugao = np.rad2deg(np.arctan(koefOci))

        visina = cvike.shape[0]
        sirina = cvike.shape[1]

        novaSirina = rastojanjeOci * 1.5
        # za usne +20, a za cvike +90

        if novaSirina < sirina:
            procenatSkaliranja = novaSirina / sirina
        else:
            procenatSkaliranja = sirina / novaSirina

        novaVisina = visina * procenatSkaliranja

        cvikeResized = cv2.resize(cvike, (np.int(novaSirina), np.int(novaVisina)))
        cvikeRotirane = ndimage.rotate(cvikeResized, -ugao)
        overlay_image(slika, cvikeRotirane, tackeLica[27, 0], tackeLica[27, 1])  # pozicijaCvikera[0],pozicijaCvikera[1])
    except:
        pass

    return slika
def dodajPlaveCvike(slika,cvike, tackeLica):
    try:
        levo_oko = (tackeLica[36, 0], tackeLica[36, 1])
        desno_oko = (tackeLica[45, 0], tackeLica[45, 1])
        rastojanjeOci = rastojanje(levo_oko, desno_oko)
        koefOci = (np.float(levo_oko[1] - desno_oko[1]) / np.float(levo_oko[0] - desno_oko[0]))
        ugao = np.rad2deg(np.arctan(koefOci))

        visina = cvike.shape[0]
        sirina = cvike.shape[1]

        novaSirina = rastojanjeOci * 1.7
        # za usne +20, a za cvike +90

        if novaSirina < sirina:
            procenatSkaliranja = novaSirina / sirina
        else:
            procenatSkaliranja = sirina / novaSirina

        novaVisina = visina * procenatSkaliranja

        cvikeResized = cv2.resize(cvike, (np.int(novaSirina), np.int(novaVisina)))
        cvikeRotirane = ndimage.rotate(cvikeResized, -ugao)
        overlay_image(slika, cvikeRotirane, tackeLica[27, 0], tackeLica[27, 1])  # pozicijaCvikera[0],pozicijaCvikera[1])
    except:
        pass
    return slika
def dodajKrunu(slika,kruna, tackeLica):
    levo = (tackeLica[1, 0], tackeLica[1, 1])
    desno = (tackeLica[16, 0], tackeLica[16, 1])
    sirinaGlave = rastojanje(levo, desno)
    koefOci = (np.float(levo[1] - desno[1]) / np.float(levo[0] - desno[0]))
    ugao = np.rad2deg(np.arctan(koefOci))

    visina = kruna.shape[0]
    sirina = kruna.shape[1]

    novaSirina = sirinaGlave * 1.2
    # za usne +20, a za cvike +90

    if novaSirina < sirina:
        procenatSkaliranja = novaSirina / sirina
    else:
        procenatSkaliranja = sirina / novaSirina

    novaVisina = visina * procenatSkaliranja

    cvikeResized = cv2.resize(kruna, (np.int(novaSirina), np.int(novaVisina)))
    cvikeRotirane = ndimage.rotate(cvikeResized, -ugao)
    overlay_image(slika, cvikeRotirane, tackeLica[27, 0], tackeLica[27, 1])  # pozicijaCvikera[0],pozicijaCvikera[1])

    return slika
def dodajUsne(slika,usne,tackeLica):
    try:
        usna_levo = (tackeLica[48, 0], tackeLica[48, 1])
        usna_desno = (tackeLica[54, 0], tackeLica[54, 1])
        pozicijaUsana = ((tackeLica[62, 0] + tackeLica[66, 0]) / 2, (tackeLica[62, 1] + tackeLica[66, 1]) / 2)

        visina = usne.shape[0]
        sirina = usne.shape[1]
        rastojanjeUsne = rastojanje(usna_levo, usna_desno)

        novaSirina = rastojanjeUsne * 1.2
        # za usne +20, a za cvike +90

        if novaSirina < sirina:
            procenatSkaliranja = novaSirina / sirina
        else:
            procenatSkaliranja = sirina / novaSirina

        novaVisina = visina * procenatSkaliranja

        usneResized = cv2.resize(usne, (np.int(novaSirina), np.int(novaVisina)))

        koefUsne = (np.float(usna_levo[1] - usna_desno[1]) / np.float(usna_levo[0] - usna_desno[0]))
        ugaoUsne = np.rad2deg(np.arctan(koefUsne))

        usneRotated = ndimage.rotate(usneResized, -ugaoUsne)

        overlay_image(slika, usneRotated, pozicijaUsana[0], pozicijaUsana[1])
    except:
        pass
    return slika
def cartoon(slika):
    numDownSamples = 2  # number of downscaling steps
    numBilateralFilters = 50  # number of bilateral filtering steps
    img_rgb=slika
    img_color = img_rgb
    for _ in xrange(numDownSamples):
        img_color = cv2.pyrDown(img_color)
    for _ in xrange(numBilateralFilters):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
    for _ in xrange(numDownSamples):
        img_color = cv2.pyrUp(img_color)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 7, 3)
    (x, y, z) = img_color.shape
    img_edge = cv2.resize(img_edge, (y, x))
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("edge.png", img_edge)
    return cv2.bitwise_and(img_color, img_edge)

    """img=slika

    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    edges = cv2.blur(gray, (3, 3))  # this blur gets rid of some noise
    edges = cv2.Canny(edges, 50, 150, apertureSize=3)  # this is the edge detection

    # the edges are a bit thin, this blur and threshold make them a bit fatter
    kernel = np.ones((3, 3), dtype=np.float) / 12.0
    edges = cv2.filter2D(edges, 0, kernel)
    edges = cv2.threshold(edges, 50, 255, 0)[1]

    # and back to colour...
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # this is the expensive operation, it blurs things but keeps track of
    # colour boundaries or something, heck, just play with it
    shifted = cv2.pyrMeanShiftFiltering(img, 5, 20)

    # now compose with the edges, the edges are white so take them away
    # to leave black
    return cv2.subtract(shifted, edges)"""

def kontrast(slika, intenzitet):
    maxIntensity = 255.0  # depends on dtype of image data
    x = np.arange(maxIntensity)

    phi = 1
    theta = 1

    newImage0 = (maxIntensity / phi) * (slika / (maxIntensity / theta)) ** intenzitet.get()
    newImage0 = np.array(newImage0, dtype=np.uint8)
    return newImage0

def dodajNos(slika,nos,tackeLica):
    try:
        levo_oko = (tackeLica[36, 0], tackeLica[36, 1])
        desno_oko = (tackeLica[45, 0], tackeLica[45, 1])
        koefOci = (np.float(levo_oko[1] - desno_oko[1]) / np.float(levo_oko[0] - desno_oko[0]))
        ugao = np.rad2deg(np.arctan(koefOci))

        levi_nos = (tackeLica[31, 0], tackeLica[31, 1])
        desni_nos = (tackeLica[35, 0], tackeLica[35, 1])
        pozicijaNos=(tackeLica[30,0],tackeLica[30,1])

        rastojanjeNos = rastojanje(levi_nos, desni_nos)

        visina = nos.shape[0]
        sirina = nos.shape[1]

        novaSirina = rastojanjeNos * 1.6
        # za usne +20, a za cvike +90

        if novaSirina < sirina:
            procenatSkaliranja = novaSirina / sirina
        else:
            procenatSkaliranja = sirina / novaSirina

        novaVisina = visina * procenatSkaliranja

        nosResized = cv2.resize(nos, (np.int(novaSirina), np.int(novaVisina)))
        nosRotated = ndimage.rotate(nosResized, -ugao)

        overlay_image(slika,nosRotated,pozicijaNos[0],pozicijaNos[1])
    except:
        pass
    return slika
def dodajDalmatinac(slika,njuska,levoUvo,desnoUvo,jezik,tackeLica,uvecanje):
    #da li da se doda jezik
    try:
        jezikBool=False
        gornjaUsna=(tackeLica[62,0],tackeLica[62,1])
        donjaUsna=(tackeLica[66,0],tackeLica[66,1])
        otvorUsta=rastojanje(gornjaUsna,donjaUsna)
        granica=rastojanje((tackeLica[30,0],tackeLica[30,1]),(tackeLica[33,0],tackeLica[33,1]))
        if otvorUsta>granica:
            jezikBool=True

        #deo za pronalazenje sredine cela
        pocetakNosa = (tackeLica[27, 0], tackeLica[27, 1])
        krajNosa = (tackeLica[33, 0], tackeLica[33, 1])
        brada = (tackeLica[8, 0], tackeLica[8, 1])
        brada_nos = rastojanje(krajNosa, brada)
        #CESTO PUCA DELJENJE NULOM
        koefNos = (np.float(pocetakNosa[1] - krajNosa[1]) / np.float(pocetakNosa[0] - krajNosa[0]))

        yCelo = int(pocetakNosa[1] - brada_nos / 2)
        xCelo = int((yCelo - krajNosa[1]) / koefNos + krajNosa[0])
        sredinaCela = (xCelo, yCelo)


        levo_glava = (tackeLica[2, 0], tackeLica[2, 1])
        desno_glava=(tackeLica[14,0],tackeLica[14,1])
        sredina_glava = (tackeLica[30, 0], tackeLica[30, 1])
        pola_glaveLeve=rastojanje(levo_glava,sredina_glava)
        pola_glaveDesne=rastojanje(desno_glava,sredina_glava)

        levo_oko = (tackeLica[36, 0], tackeLica[36, 1])
        desno_oko = (tackeLica[45, 0], tackeLica[45, 1])
        koefOci = (np.float(levo_oko[1] - desno_oko[1]) / np.float(levo_oko[0] - desno_oko[0]))
        ugao = np.rad2deg(np.arctan(koefOci))

        xLevoUvo = int(xCelo - np.sqrt(pola_glaveLeve*pola_glaveLeve/(1+koefOci*koefOci)))
        yLevoUvo=int(koefOci*(xLevoUvo-xCelo)+yCelo)

        xDesnoUvo = int(xCelo + np.sqrt(pola_glaveDesne * pola_glaveDesne / (1 + koefOci * koefOci)))
        yDesnoUvo = int(koefOci * (xDesnoUvo - xCelo) + yCelo)

        #cv2.circle(slika,(xLevoUvo,yLevoUvo),3,(255,255,0),1)
        #cv2.circle(slika, (xDesnoUvo, yDesnoUvo), 3, (0, 0, 255), 1)
        #cv2.circle(slika, (xCelo, yCelo), 3, (0, 255, 0), 1)

        #cv2.circle(slika, (levo_glava[0], levo_glava[1]), 3, (0, 255, 255), 1)
        #cv2.circle(slika, (desno_glava[0], desno_glava[1]), 3, (0, 255, 255), 1)


        #pozicije za usi,njusku,jezik

        pozijaNjuska=(tackeLica[30,0],tackeLica[30,1])
        pozicijaJezik=(tackeLica[66,0],tackeLica[66,1])



        levi_nos = (tackeLica[31, 0], tackeLica[31, 1])
        desni_nos = (tackeLica[35, 0], tackeLica[35, 1])
        pozicijaNos=(tackeLica[30,0],tackeLica[30,1])


        duzinaNosa = rastojanje(pocetakNosa, krajNosa)

        visina = levoUvo.shape[0]
        sirina = levoUvo.shape[1]

        novaVisina= float(duzinaNosa*uvecanje)


        procenatSkaliranja = float(novaVisina / visina)

        novaSirina=int(sirina*procenatSkaliranja)

        novaVisinaNjuska=int(njuska.shape[0]*procenatSkaliranja)
        novaSirinaNjuska=int(njuska.shape[1]*procenatSkaliranja)

        levoUvoResized = cv2.resize(levoUvo, (np.int(novaSirina), np.int(novaSirina)))
        desnoUvoResized = cv2.resize(desnoUvo, (np.int(novaSirina), np.int(novaVisina)))
        njuskaResized = cv2.resize(njuska, (np.int(novaSirinaNjuska), np.int(novaVisinaNjuska)))
        jezikResized = cv2.resize(jezik, (np.int(novaSirina*0.8), np.int(novaVisina*0.8)))

        levoUvoRotated = ndimage.rotate(levoUvoResized, -ugao)
        njuskaRotated = ndimage.rotate(njuskaResized, -ugao)
        desnoUvoRotated = ndimage.rotate(desnoUvoResized, -ugao)
        jezikRotated = ndimage.rotate(jezikResized, -ugao)


        overlay_image(slika,levoUvoRotated,xLevoUvo,yLevoUvo)
        overlay_image(slika, desnoUvoRotated, xDesnoUvo, yDesnoUvo)
        overlay_image(slika, njuskaRotated, pozijaNjuska[0], pozijaNjuska[1])
        if jezikBool==True:
            overlay_image(slika, jezikRotated, pozicijaJezik[0], pozicijaJezik[1])


    except:
        pass
    return slika

def dodajTreceOko(slika,oko,tackeLica):
    try:
        ldesno_oko = (tackeLica[42, 0], tackeLica[42, 1])
        ddesno_oko = (tackeLica[45, 0], tackeLica[45, 1])
        dlevo_oko=(tackeLica[39,0],tackeLica[39,1])

        koefOci = (np.float(dlevo_oko[1] - ldesno_oko[1]) / np.float(dlevo_oko[0] - ldesno_oko[0]))
        ugao = np.rad2deg(np.arctan(koefOci))

        pocetakNosa=(tackeLica[27, 0], tackeLica[27, 1])
        krajNosa=(tackeLica[33, 0], tackeLica[33, 1])
        brada=(tackeLica[8, 0], tackeLica[8, 1])
        brada_nos=rastojanje(krajNosa,brada)
        koefNos = (np.float(pocetakNosa[1] - krajNosa[1]) / np.float(pocetakNosa[0] - krajNosa[0]))

        yCelo=int(pocetakNosa[1]-brada_nos/2)
        xCelo= int((yCelo-krajNosa[1])/koefNos+krajNosa[0])
        treceOkoPozicija=(xCelo,yCelo)




        pozicijaNos=(tackeLica[30,0],tackeLica[30,1])

        rastojanjeOko = rastojanje(ldesno_oko, ddesno_oko)

        visina = oko.shape[0]
        sirina = oko.shape[1]

        novaSirina = rastojanjeOko * 1.6
        # za usne +20, a za cvike +90

        if novaSirina < sirina:
            procenatSkaliranja = novaSirina / sirina
        else:
            procenatSkaliranja = sirina / novaSirina

        novaVisina = visina * procenatSkaliranja

        okoResized = cv2.resize(oko, (np.int(novaSirina), np.int(novaVisina)))
        okoRotirane = ndimage.rotate(okoResized, -ugao)

        overlay_image(slika,okoRotirane,treceOkoPozicija[0],treceOkoPozicija[1])
    except:
        pass
    return slika
def dodajBrkove(slika,brkovi,tackeLica):
    try:
        usna_levo = (tackeLica[48, 0], tackeLica[48, 1])
        usna_desno = (tackeLica[54, 0], tackeLica[54, 1])

        visina = brkovi.shape[0]
        sirina = brkovi.shape[1]
        rastojanjeUsne = rastojanje(usna_levo, usna_desno)

        novaSirina = rastojanjeUsne * 1.8

        if novaSirina < sirina:
            procenatSkaliranja = novaSirina / sirina
        else:
            procenatSkaliranja = sirina / novaSirina
        novaVisina = visina * procenatSkaliranja

        brkoviResized = cv2.resize(brkovi, (np.int(novaSirina), np.int(novaVisina)))

        koefUsne = (np.float(usna_levo[1] - usna_desno[1]) / np.float(usna_levo[0] - usna_desno[0]))
        ugaoUsne = np.rad2deg(np.arctan(koefUsne))

        brkoviRotated = ndimage.rotate(brkoviResized, -ugaoUsne)

        pozivijaBrkova=((tackeLica[51,0]+tackeLica[33,0])/2,(tackeLica[51,1]+tackeLica[33,1])/2)
        overlay_image(slika, brkoviRotated, pozivijaBrkova[0], pozivijaBrkova[1])
    except:
        pass
    return slika

'''def dodajTrepavice(slika,trepavice,tackeLica):
    levo_oko = (tackeLica[36, 0], tackeLica[36, 1])
    desno_oko = (tackeLica[39, 0], tackeLica[39, 1])
    pozicijaLeve=(tackeLica[37, 0], tackeLica[37, 1])
    pozicijaDesne=(tackeLica[44, 0], tackeLica[44, 1])

    rastojanjeOci = rastojanje(levo_oko, desno_oko)
    koefOci = (np.float(levo_oko[1] - desno_oko[1]) / np.float(levo_oko[0] - desno_oko[0]))
    ugao = np.rad2deg(np.arctan(koefOci))

    visina = trepavice.shape[0]
    sirina = trepavice.shape[1]

    novaSirina = rastojanjeOci * 1.5
    # za usne +20, a za cvike +90

    if novaSirina < sirina:
        procenatSkaliranja = novaSirina / sirina
    else:
        procenatSkaliranja = sirina / novaSirina

    novaVisina = visina * procenatSkaliranja

    trepavicaResized = cv2.resize(trepavice, (np.int(novaSirina), np.int(novaVisina)))

    overlay_image(slika,trepavicaResized,pozicijaDesne[0],pozicijaDesne[1])
    return slika
'''
def rastojanje(pozicija1,pozicija2):
    x = (pozicija1[0] - pozicija2[0]) * (pozicija1[0] - pozicija2[0])
    y = (pozicija1[1] - pozicija2[1]) * (pozicija1[1] - pozicija2[1])
    res=np.sqrt(x+y)
    return res

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

