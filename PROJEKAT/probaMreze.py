import pickle
import cv2
import prepoznavanjeLica
import dlib
import numpy as np


def rastojanje(pozicija1,pozicija2):
    x = (pozicija1[0] - pozicija2[0]) * (pozicija1[0] - pozicija2[0])
    y = (pozicija1[1] - pozicija2[1]) * (pozicija1[1] - pozicija2[1])
    res=np.sqrt(x+y)
    return res


def izracunavanjeOdnosa(slika):
    tackeLica = prepoznavanjeLica.prepoznajLice(slika, detector, predictor)

    levaTacka = (tackeLica[1, 0], tackeLica[1, 1])
    desnaTacka = (tackeLica[15, 0], tackeLica[15, 1])
    levaObrva = (tackeLica[19, 0], tackeLica[19, 1])
    desnaObrva = (tackeLica[24, 0], tackeLica[24, 1])
    gornjaSrednja = ((levaObrva[0] + desnaObrva[0]) / 2, (levaObrva[1] + desnaObrva[1]) / 2)
    brada = (tackeLica[8, 0], tackeLica[8, 1])
    nos = (tackeLica[33, 0], tackeLica[33, 1])
    pocetaLeveObrve = (tackeLica[21, 0], tackeLica[21, 1])
    pocetakDesneObrve = (tackeLica[22, 0], tackeLica[22, 1])
    levaNozdrva = (tackeLica[31, 0], tackeLica[31, 1])
    desnaNozdrva = (tackeLica[35, 0], tackeLica[35, 1])
    vrhNosa = (tackeLica[27, 0], tackeLica[27, 1])
    dnoNosa = (tackeLica[33, 0], tackeLica[33, 1])
    levoLevoOko = (tackeLica[36, 0], tackeLica[36, 1])
    levoDesnoOko = (tackeLica[39, 0], tackeLica[39, 1])
    levoGoreOko = (tackeLica[38, 0], tackeLica[38, 1])
    levoDoleOko = (tackeLica[40, 0], tackeLica[40, 1])
    desnoLevoOko = (tackeLica[42, 0], tackeLica[42, 1])
    desnoDesnoOko = (tackeLica[45, 0], tackeLica[45, 1])
    desnoGoreOko = (tackeLica[43, 0], tackeLica[43, 1])
    desnoDoleOko = (tackeLica[47, 0], tackeLica[47, 1])

    sirinaLica = rastojanje(levaTacka, desnaTacka)
    visinaLica = rastojanje(gornjaSrednja, brada)

    obrveNos = rastojanje(gornjaSrednja, nos)
    nosBrada = rastojanje(nos, brada)
    izmedjuObrva = rastojanje(pocetaLeveObrve, pocetakDesneObrve)
    nozdrve = rastojanje(levaNozdrva, desnaNozdrva)
    nos = rastojanje(vrhNosa, dnoNosa)
    levoOkoSirina = rastojanje(levoLevoOko, levoDesnoOko)
    levoOkoVisina = rastojanje(levoGoreOko, levoDoleOko)
    desnoOkoSirina = rastojanje(desnoLevoOko, desnoDesnoOko)
    desnoOkoVIsina = rastojanje(desnoGoreOko, desnoDoleOko)
    levoOkoObrva = rastojanje((tackeLica[18, 0], tackeLica[18, 1]), (tackeLica[36, 0], tackeLica[36, 1]))
    desnoOkoObrva = rastojanje((tackeLica[25, 0], tackeLica[25, 1]), (tackeLica[45, 0], tackeLica[45, 1]))

    koefLevaObrva = np.abs(
        (np.float(tackeLica[19, 1] - tackeLica[21, 1]) / np.float(tackeLica[19, 0] - tackeLica[21, 0])))
    koefDesnaObrva = np.abs(
        (np.float(tackeLica[24, 1] - tackeLica[22, 1]) / np.float(tackeLica[24, 0] - tackeLica[22, 0])))
    ugaoLevaObrva = np.rad2deg(np.arctan(koefLevaObrva))
    ugaoDesnaObrva = np.rad2deg(np.arctan(koefDesnaObrva))

    srednjiUgaoObrva = (ugaoDesnaObrva + ugaoLevaObrva) / 2

    visina_sirina = visinaLica / sirinaLica
    visina_nosObrve = visinaLica / obrveNos
    visina_nosBrada = visinaLica / nosBrada
    visina_razmakIzmedjuObrva = visinaLica / izmedjuObrva
    visina_nozdrve = visinaLica / nozdrve
    visina_nos = visinaLica / nos
    visina_lOkoSirina = visinaLica / levoOkoSirina
    visina_lOkoVisina = visinaLica / levoOkoVisina
    visina_dOkoSirina = visinaLica / desnoOkoSirina
    visina_dOkoVisina = visinaLica / desnoOkoVIsina

predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


ulazNovi=[]

for ind in xrange(1, 2):

    putanja = "muskarciPNG/" + str(ind) + ".png"

    lik = cv2.imread(putanja, cv2.IMREAD_UNCHANGED)
    #lik = cv2.imread("Lica/lik.png", cv2.IMREAD_UNCHANGED)




with open('mrezaZaPol2','rb') as f:
    novi = pickle.load(f)
    t = novi.predict(input1, verbose=1)
    print t