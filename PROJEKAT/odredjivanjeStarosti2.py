import dlib
import cv2
import prepoznavanjeLica
import numpy as np
#--------------- ANN ------------------
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
import pickle

def rastojanje(pozicija1,pozicija2):
    x = (pozicija1[0] - pozicija2[0]) * (pozicija1[0] - pozicija2[0])
    y = (pozicija1[1] - pozicija2[1]) * (pozicija1[1] - pozicija2[1])
    res=np.sqrt(x+y)
    return res

def racunanjeOdnosa(slika):
    matrica = prepoznavanjeLica.prepoznajLice(slika, detector, predictor)

    t21 = (matrica[21, 0], matrica[21, 1])
    t22 = (matrica[22, 0], matrica[22, 1])
    t27 = (matrica[27, 0], matrica[27, 1])
    t21_t22 = ((t21[0] + t22[0]) / 2, (t21[1] + t22[1]) / 2)

    FMN = ((t21_t22[0] + t27[0]) / 2, (t21_t22[1] + t27[1]) / 2)
    LEL = (matrica[36, 0], matrica[36, 1])
    HEL = ((matrica[37, 0] + matrica[38, 0]) / 2, (matrica[37, 1] + matrica[38, 1]) / 2)
    LMLEL = ((matrica[41, 0] + matrica[40, 0]) / 2, (matrica[41, 1] + matrica[40, 1]) / 2)  # 41-40
    MEL = (matrica[39, 0], matrica[39, 1])
    MER = (matrica[42, 0], matrica[42, 1])
    HER = ((matrica[43, 0] + matrica[44, 0]) / 2, (matrica[43, 1] + matrica[44, 1]) / 2)  # 43-44
    LMLER = ((matrica[46, 0] + matrica[47, 0]) / 2, (matrica[46, 1] + matrica[47, 1]) / 2)  # 47-46
    LER = (matrica[45, 0], matrica[45, 1])
    PHL = (matrica[1, 0], matrica[1, 1])
    PHR = (matrica[15, 0], matrica[15, 1])
    PNT = (matrica[30, 0], matrica[30, 1])
    INTUL = (matrica[33, 0], matrica[33, 1])
    LNAR = (matrica[31, 0], matrica[31, 1])
    HULR = (matrica[35, 0], matrica[35, 1])
    LAML = (matrica[3, 0], matrica[3, 1])
    LAMR = (matrica[13, 0], matrica[13, 1])
    LMLC = (matrica[8, 0], matrica[8, 1])
    MULLM = (matrica[48, 0], matrica[48, 1])
    MLLL = (matrica[54, 0], matrica[54, 1])
    HULL = (matrica[51, 0], matrica[51, 1])
    MPLL = (matrica[57, 0], matrica[57, 1])
    LULMR = ((matrica[62, 0] + matrica[66, 0]) / 2, (matrica[62, 1] + matrica[66, 1]) / 2)  # 62-66

    nos_brada = rastojanje(LMLC, INTUL)
    HH = (FMN[0], int(FMN[1] - nos_brada))
    MFL = (PHL[0], int(FMN[1] - nos_brada / 2))
    MFR = (PHR[0], int(FMN[1] - nos_brada / 2))

    cv2.circle(lik, FMN, 5, (255, 0, 0), 1)
    cv2.circle(lik, LEL, 5, (0, 255, 255), 1)
    cv2.circle(lik, HEL, 5, (0, 255, 255), 1)
    cv2.circle(lik, LMLEL, 5, (0, 255, 255), 1)
    cv2.circle(lik, MEL, 5, (0, 255, 255), 1)
    cv2.circle(lik, MER, 5, (0, 255, 255), 1)
    cv2.circle(lik, HER, 5, (0, 255, 255), 1)
    cv2.circle(lik, LMLER, 5, (0, 255, 255), 1)
    cv2.circle(lik, LER, 5, (0, 255, 255), 1)
    cv2.circle(lik, PHL, 5, (0, 255, 255), 1)
    cv2.circle(lik, PHR, 5, (0, 255, 255), 1)
    cv2.circle(lik, PNT, 5, (0, 255, 255), 1)
    cv2.circle(lik, INTUL, 5, (0, 255, 255), 1)
    cv2.circle(lik, LNAR, 5, (0, 255, 255), 1)
    cv2.circle(lik, HULR, 5, (0, 255, 255), 1)
    cv2.circle(lik, LAML, 5, (0, 255, 255), 1)
    cv2.circle(lik, LAMR, 5, (0, 255, 255), 1)
    cv2.circle(lik, LMLC, 5, (0, 255, 255), 1)
    cv2.circle(lik, MULLM, 5, (0, 255, 255), 1)
    cv2.circle(lik, MLLL, 5, (0, 255, 255), 1)
    cv2.circle(lik, HULL, 5, (0, 255, 255), 1)
    cv2.circle(lik, MPLL, 5, (0, 255, 255), 1)
    cv2.circle(lik, LULMR, 5, (0, 255, 255), 1)
   # cv2.circle(lik, HH, 5, (0, 255, 255), 1)
   # cv2.circle(lik, MFL, 5, (222, 0, 0), 1)
   # cv2.circle(lik, MFR, 5, (255, 0, 0), 1)

    MFL_LAML = rastojanje(MFL, LAML)
    MFL_PHR = rastojanje(MFL, PHR)
    MFL_MFR = rastojanje(MFL, MFR)
    PHL_PHR = rastojanje(PHL, PHR)
    PHL_MFR = rastojanje(PHL, MFR)
    MFL_LMLC = rastojanje(MFL, LMLC)
    LMLC_LMLEL = rastojanje(LMLC, LMLEL)
    LAMR_PHR = rastojanje(LAMR, PHR)
    LAMR_MFR = rastojanje(LAMR, MFR)
    PHL_LAML = rastojanje(PHL, LAML)
    LMLC_MEL = rastojanje(LMLC, MEL)
    PHR_MFR = rastojanje(PHR, MFR)
    LMLC_PHR = rastojanje(LMLC, PHR)
    LMLC_MFR = rastojanje(LMLC, MFR)
    LMLC_LEL = rastojanje(LMLC, LEL)
    LEL_LER = rastojanje(LEL, LER)
    LEL_HER = rastojanje(LEL, HER)
    HEL_LER = rastojanje(HEL, LER)
    LMLEL_LER = rastojanje(LMLEL, LER)
    MEL_LER = rastojanje(MEL, LER)
    LMLC_HEL = rastojanje(LMLC, HEL)
    LEL_LMLER = rastojanje(LEL, LMLER)
    LEL_MER = rastojanje(LEL, MER)
    HEL_HER = rastojanje(HEL, HER)
    HEL_LMLER = rastojanje(HEL, LMLER)
    LMLEL_HER = rastojanje(LMLEL, HER)
    MEL_HER = rastojanje(MEL, HER)
    MEL_LMLER = rastojanje(MEL, LMLER)
    LMLC_LER = rastojanje(LMLC, LER)
    LMLC_HER = rastojanje(LMLC, HER)
    LMLC_LMLER = rastojanje(LMLC, LMLER)
    LMLC_MER = rastojanje(LMLC, MER)

    niz = []

    r1 = MFL_LAML / MFL_PHR
    r2 = MFL_LAML / MFL_MFR
    r3 = MFL_LAML / PHL_PHR
    r4 = MFL_LAML / PHL_MFR
    r5 = MFL_LMLC / MFL_MFR
    r6 = MFL_PHR / LMLC_LMLEL
    r7 = MFL_PHR / LAMR_PHR
    r8 = MFL_PHR / LAMR_MFR
    r9 = MFL_PHR / PHL_LAML
    r10 = MFL_MFR / LMLC_LMLEL
    r11 = MFL_MFR / LMLC_MEL
    r12 = MFL_MFR / LAMR_PHR
    r13 = MFL_MFR / LAMR_MFR
    r14 = MFL_MFR / PHR_MFR
    r15 = MFL_MFR / PHL_LAML
    r16 = MFL_MFR / LMLC_PHR
    r17 = MFL_MFR / LMLC_MFR
    r18 = PHL_LAML / PHL_PHR
    r19 = PHL_LAML / PHL_MFR
    r20 = PHL_PHR / LAMR_MFR
    r21 = PHL_MFR / LAMR_MFR
    r22 = LMLC_LEL / LEL_LER
    r23 = LMLC_LEL / LEL_HER
    r24 = LMLC_LEL / HEL_LER
    r25 = LMLC_LEL / LMLEL_LER
    r26 = LMLC_LEL / MEL_LER
    r27 = LMLC_HEL / LEL_LER
    r28 = LMLC_HEL / HEL_LER
    r29 = LMLC_HEL / MEL_LER
    r30 = LMLC_LMLEL / LEL_LER
    r31 = LMLC_LMLEL / LEL_HER
    r32 = LMLC_LMLEL / LEL_LMLER
    r33 = LMLC_LMLEL / LEL_MER
    r34 = LMLC_LMLEL / HEL_LER
    r35 = LMLC_LMLEL / HEL_HER
    r36 = LMLC_LMLEL / HEL_LMLER
    r37 = LMLC_LMLEL / LMLEL_LER
    r38 = LMLC_LMLEL / LMLEL_HER
    r39 = LMLC_LMLEL / MEL_LER
    r40 = LMLC_LMLEL / MEL_HER
    r41 = LMLC_LMLEL / MEL_LMLER
    r42 = LMLC_MEL / LEL_LER
    r43 = LMLC_MEL / LEL_HER
    r44 = LMLC_MEL / HEL_LER
    r45 = LMLC_MEL / MEL_LER
    r46 = LMLC_MEL / MEL_HER
    r47 = LMLC_LER / LEL_HER
    r48 = LMLC_HER / LEL_HER
    r49 = LMLC_LMLER / LEL_LER
    r50 = LMLC_LMLER / LEL_HER
    r51 = LMLC_LMLER / LEL_LMLER
    r52 = LMLC_LMLER / LEL_MER
    r53 = LMLC_LMLER / HEL_HER
    r54 = LMLC_LMLER / LMLEL_HER
    r55 = LMLC_LMLER / MEL_LER
    r56 = LMLC_LMLER / MEL_HER
    r57 = LMLC_MER / LEL_LER
    r58 = LMLC_MER / LEL_HER
    r59 = LMLC_MER / LEL_LMLER
    r60 = LMLC_MER / LEL_MER
    r61 = LMLC_MER / MEL_LER
    r62 = LMLC_MER / MEL_HER

    jedanUlaz = []
    jedanUlaz.append(r1)
    jedanUlaz.append(r2)
    jedanUlaz.append(r3)
    jedanUlaz.append(r4)
    jedanUlaz.append(r5)
    jedanUlaz.append(r6)
    jedanUlaz.append(r7)
    jedanUlaz.append(r8)
    jedanUlaz.append(r9)
    jedanUlaz.append(r10)
    jedanUlaz.append(r11)
    jedanUlaz.append(r12)
    jedanUlaz.append(r13)
    jedanUlaz.append(r14)
    jedanUlaz.append(r15)
    jedanUlaz.append(r16)
    jedanUlaz.append(r17)
    jedanUlaz.append(r18)
    jedanUlaz.append(r19)
    jedanUlaz.append(r20)
    jedanUlaz.append(r21)
    jedanUlaz.append(r22)
    jedanUlaz.append(r23)
    jedanUlaz.append(r24)
    jedanUlaz.append(r25)
    jedanUlaz.append(r26)
    jedanUlaz.append(r27)
    jedanUlaz.append(r28)
    jedanUlaz.append(r29)
    jedanUlaz.append(r30)
    jedanUlaz.append(r31)
    jedanUlaz.append(r32)
    jedanUlaz.append(r33)
    jedanUlaz.append(r34)
    jedanUlaz.append(r35)
    jedanUlaz.append(r36)
    jedanUlaz.append(r37)
    jedanUlaz.append(r38)
    jedanUlaz.append(r39)
    jedanUlaz.append(r40)
    jedanUlaz.append(r41)
    jedanUlaz.append(r42)
    jedanUlaz.append(r43)
    jedanUlaz.append(r44)
    jedanUlaz.append(r45)
    jedanUlaz.append(r46)
    jedanUlaz.append(r47)
    jedanUlaz.append(r48)
    jedanUlaz.append(r49)
    jedanUlaz.append(r50)
    jedanUlaz.append(r51)
    jedanUlaz.append(r52)
    jedanUlaz.append(r53)
    jedanUlaz.append(r54)
    jedanUlaz.append(r55)
    jedanUlaz.append(r56)
    jedanUlaz.append(r57)
    jedanUlaz.append(r58)
    jedanUlaz.append(r59)
    jedanUlaz.append(r60)
    jedanUlaz.append(r61)
    jedanUlaz.append(r62)

    ulaz = np.asarray(jedanUlaz)
    return ulaz

def to_categorical(niz, n):
    retVal = np.zeros((len(niz), n), dtype='int')
    ll = np.array(list(enumerate(niz)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal


predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


lik=cv2.imread("DATA_SET/Bebe/11.png")
#jedanUlaz=racunanjeOdnosa(lik)
#cv2.imshow("",lik)
#cv2.imshow(cv2.waitKey(0))



sviUlazi=[]
sviIzlazi=[]

for ii in xrange(1,121):
    if ii!=11:
        putanja1 = "Dataset_Starost/Bebe/"+str(ii)+".png"
        lice=cv2.imread(putanja1)
        jedanUlaz=racunanjeOdnosa(lice)
        #print ii
        jedanIzlaz=0
        sviUlazi.append(jedanUlaz)
        sviIzlazi.append(jedanIzlaz)

for i in xrange(1,121):
    putanja2="Dataset_Starost/Deca/"+str(i)+".png"
    lice = cv2.imread(putanja2)
    jedanUlaz = racunanjeOdnosa(lice)
    #print i
    jedanIzlaz = 0
    sviUlazi.append(jedanUlaz)
    sviIzlazi.append(jedanIzlaz)

for i in xrange(1,121):
    putanja3="Dataset_Starost/Mladi/"+str(i)+".png"
    lice = cv2.imread(putanja3)
    jedanUlaz = racunanjeOdnosa(lice)
    #print i
    jedanIzlaz = 1
    sviUlazi.append(jedanUlaz)
    sviIzlazi.append(jedanIzlaz)

for i in xrange(1,121):
    putanja4="Dataset_Starost/Stari/"+str(i)+".png"
    lice = cv2.imread(putanja4)
    jedanUlaz = racunanjeOdnosa(lice)
    #print i
    jedanIzlaz = 1
    sviUlazi.append(jedanUlaz)
    sviIzlazi.append(jedanIzlaz)

sviUlazi1=np.asarray(sviUlazi)
sviIzlazi1=np.asarray(sviIzlazi)
izlaz=to_categorical(sviIzlazi1,2)

'''nesto=sviUlazi1
noviUlaz=sviUlazi1
srednjaVrednost=sviUlazi1.mean(axis=0)

for j in xrange(0,277):
    nesto[j,:]-=srednjaVrednost
    nesto[j,:]=nesto[j,:]**2
print nesto.shape
oduz=sviUlazi1-srednjaVrednost
meanMalo=nesto.mean(axis=0)
s=np.sqrt(meanMalo)

for j in xrange(0,277):
    noviUlaz[j,:]=(sviUlazi1[j,:]-srednjaVrednost)/s'''

# prepare model
model = Sequential()
model.add(Dense(8, input_dim=62))
model.add(Activation('tanh'))
#model.add(Dense(50))
#model.add(Activation('tanh'))
model.add(Dense(2))
model.add(Activation('tanh'))

# compile model with optimizer
sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
model.compile(loss='mean_squared_error', optimizer=sgd)

# training
training = model.fit(sviUlazi1, izlaz, nb_epoch=500, batch_size=400, verbose=1)
print training.history['loss'][-1]

with open('mrezaStarostDveGrupeV2', 'wb') as f:
    pickle.dump(model, f)

#cv2.imshow("",lik)
#cv2.imshow(cv2.waitKey(0))