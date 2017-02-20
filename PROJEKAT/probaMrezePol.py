import pickle
import cv2
import prepoznavanjeLica
import dlib
import numpy as np


with open('mrezaPol','rb') as f:

    novi = pickle.load(f)

def rastojanje(pozicija1,pozicija2):
    x = (pozicija1[0] - pozicija2[0]) * (pozicija1[0] - pozicija2[0])
    y = (pozicija1[1] - pozicija2[1]) * (pozicija1[1] - pozicija2[1])
    res=np.sqrt(x+y)
    return res

def prepoznaj(slika):

    predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    ulazNovi = []
    ulaz = []

    lik = slika
    tackeLica = prepoznavanjeLica.prepoznajLice(lik, detector, predictor)
    levaObrva = (tackeLica[19, 0], tackeLica[19, 1])
    desnaObrva = (tackeLica[24, 0], tackeLica[24, 1])
    gornjaSrednja = ((levaObrva[0] + desnaObrva[0]) / 2, (levaObrva[1] + desnaObrva[1]) / 2)
    brada = (tackeLica[8, 0], tackeLica[8, 1])
    levaNozdrva = (tackeLica[31, 0], tackeLica[31, 1])
    desnaNozdrva = (tackeLica[35, 0], tackeLica[35, 1])
    dnoNosa = (tackeLica[33, 0], tackeLica[33, 1])
    levoLevoOko = (tackeLica[36, 0], tackeLica[36, 1])
    levoDesnoOko = (tackeLica[39, 0], tackeLica[39, 1])
    levoDoleDesnoOko = (tackeLica[40, 0], tackeLica[40, 1])
    levoDoleLevoOko = (tackeLica[41, 0], tackeLica[41, 1])
    desnoLevoOko = (tackeLica[42, 0], tackeLica[42, 1])
    desnoDesnoOko = (tackeLica[45, 0], tackeLica[45, 1])
    desnoDoleLevoOko = (tackeLica[47, 0], tackeLica[47, 1])
    desnoDoleDesnoOko = (tackeLica[46, 0], tackeLica[46, 1])
    ustaLeviUgao = (tackeLica[48, 0], tackeLica[48, 1])
    ustaDesniugao = (tackeLica[54, 0], tackeLica[54, 1])
    ustaSredinaGore = (tackeLica[51, 0], tackeLica[51, 1])
    ustaSredinaDole = (tackeLica[57, 0], tackeLica[57, 1])
    ustaGoreLevo = (tackeLica[50, 0], tackeLica[50, 1])
    ustaGoreDesno = (tackeLica[52, 0], tackeLica[52, 1])

    visinaLica = rastojanje(gornjaSrednja, brada)

    ulaz.append(levaObrva)
    ulaz.append(desnaObrva)
    ulaz.append(gornjaSrednja)
    ulaz.append(brada)
    ulaz.append(levaNozdrva)
    ulaz.append(desnaNozdrva)
    ulaz.append(dnoNosa)
    ulaz.append(levoLevoOko)
    ulaz.append(levoDesnoOko)
    ulaz.append(levoDoleDesnoOko)
    ulaz.append(levoDoleLevoOko)
    ulaz.append(desnoLevoOko)
    ulaz.append(desnoDesnoOko)
    ulaz.append(desnoDoleLevoOko)
    ulaz.append(desnoDoleDesnoOko)
    ulaz.append(ustaLeviUgao)
    ulaz.append(ustaDesniugao)
    ulaz.append(ustaSredinaGore)
    ulaz.append(ustaSredinaDole)
    ulaz.append(ustaGoreLevo)
    ulaz.append(ustaGoreDesno)
    svi = np.asarray(ulaz)
    rastojanja = []
    for br in xrange(0, svi.shape[0] - 1):
        for dr in xrange(br + 1, svi.shape[0]):
            pozicija1 = svi[br]
            pozicija2 = svi[dr]
            rastojanja.append(visinaLica / rastojanje(pozicija1, pozicija2))
    ulazNovi.append(rastojanja)
    input1 = np.asarray(ulazNovi)
    t = novi.predict(input1)

    tekst=''
    if(t[0][0]>t[0][1]):
        tekst='MUSKARAC'
    else:
        tekst='ZENA'

    print t
    print tekst
    return tekst