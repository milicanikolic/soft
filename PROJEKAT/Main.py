import dlib
import cv2
from Tkinter import *
from PIL import Image, ImageTk
from qtconsole.mainwindow import background
import prepoznavanjeLica
import provaMrezeStarost

import stikeri as st
import probaMrezePol as pol


def dodajCheckBoxes(stikeri):
    # ************BRKOVI************
    filter1Frame = Frame(stikeri, width=50, height=20)
    filter1Frame.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    filter1 = PhotoImage(file='ikonice/rsz_brkovi2.gif')

    labelaFil1 = Label(filter1Frame, image=filter1)
    labelaFil1.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    labelaFil1.image = filter1

    var1 = IntVar()
    Checkbutton(filter1Frame, variable=var1, command=brkoviClick(var1)).grid(row=0, column=1)
    # *************************************

    # ************SVINJA************
    filter2Frame = Frame(stikeri, width=50, height=20)
    filter2Frame.grid(row=0, column=1, padx=10, pady=2)
    filter2 = PhotoImage(file='ikonice/rsz_pig.gif')

    labelaFil2 = Label(filter2Frame, image=filter2)
    labelaFil2.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    labelaFil2.image = filter2

    var2 = IntVar()
    Checkbutton(filter2Frame, variable=var2, command=svinjaClick(var2)).grid(row=0, column=1)
    # *************************************

    # ************OKO************
    filter3Frame = Frame(stikeri, width=50, height=20)
    filter3Frame.grid(row=0, column=2, padx=10, pady=2)
    filter3 = PhotoImage(file='ikonice/rsz_oko.gif')

    labelaFil3 = Label(filter3Frame, image=filter3)
    labelaFil3.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    labelaFil3.image = filter3

    var3 = IntVar()
    Checkbutton(filter3Frame, variable=var3, command=okoClick(var3)).grid(row=0, column=1)
    # *************************************

    # ************PAS************
    filter4Frame = Frame(stikeri, width=50, height=20)
    filter4Frame.grid(row=0, column=3, padx=10, pady=2)
    filter4 = PhotoImage(file='ikonice/rsz_pas.gif')

    labelaFil4 = Label(filter4Frame, image=filter4)
    labelaFil4.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    labelaFil4.image = filter4

    var4 = IntVar()
    Checkbutton(filter4Frame, variable=var4, command=pasClick(var4)).grid(row=0, column=1)
    # *************************************

    # ************DALMATINAC************
    filter5Frame = Frame(stikeri, width=50, height=20)
    filter5Frame.grid(row=0, column=4, padx=10, pady=2)
    filter5 = PhotoImage(file='ikonice/rsz_dalm.gif')

    labelaFil5 = Label(filter5Frame, image=filter5)
    labelaFil5.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    labelaFil5.image = filter5

    var5 = IntVar()
    Checkbutton(filter5Frame, variable=var5, command=dalClick(var5)).grid(row=0, column=1)
    # *************************************

    # ************CUCLADECAK************
    filter6Frame = Frame(stikeri, width=50, height=20)
    filter6Frame.grid(row=0, column=5, padx=10, pady=2)
    filter6 = PhotoImage(file='ikonice/rsz_cuclabata.gif')

    labelaFil6 = Label(filter6Frame, image=filter6)
    labelaFil6.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    labelaFil6.image = filter6

    var6 = IntVar()
    Checkbutton(filter6Frame, variable=var6, command=cuclaPlavaClick(var6)).grid(row=0, column=1)
    # *************************************

    # ************USNE************
    filter7Frame = Frame(stikeri, width=50, height=20)
    filter7Frame.grid(row=1, column=0, padx=10, pady=2)
    filter7 = PhotoImage(file='ikonice/rsz_lips2.gif')

    labelaFil7 = Label(filter7Frame, image=filter7)
    labelaFil7.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    labelaFil7.image = filter7

    var7 = IntVar()
    Checkbutton(filter7Frame, variable=var7, command=usneClick(var7)).grid(row=0, column=1)
    # *************************************

    # ************KLOVN************
    filter8Frame = Frame(stikeri, width=50, height=20)
    filter8Frame.grid(row=1, column=1, padx=10, pady=2)
    filter8 = PhotoImage(file='ikonice/rsz_klovn.gif')

    labelaFil8 = Label(filter8Frame, image=filter8)
    labelaFil8.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    labelaFil8.image = filter8

    var8 = IntVar()
    Checkbutton(filter8Frame, variable=var8, command=klovnClick(var8)).grid(row=0, column=1)
    # *************************************

    # ************NAOCARE SRCA************
    filter9Frame = Frame(stikeri, width=50, height=20)
    filter9Frame.grid(row=1, column=2, padx=10, pady=2)
    filter9 = PhotoImage(file='ikonice/rsz_sunglasses.gif')

    labelaFil9 = Label(filter9Frame, image=filter9)
    labelaFil9.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    labelaFil9.image = filter9

    var9 = IntVar()
    Checkbutton(filter9Frame, variable=var9, command=cvikeSrceClick(var9)).grid(row=0, column=1)
    # *************************************

    # ************NAOCARE PLAVE************
    filter10Frame = Frame(stikeri, width=50, height=20)
    filter10Frame.grid(row=1, column=3, padx=10, pady=2)
    filter10 = PhotoImage(file='ikonice/rsz_sunglassesblue.gif')

    labelaFil10 = Label(filter10Frame, image=filter10)
    labelaFil10.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    labelaFil10.image = filter10

    var10 = IntVar()
    Checkbutton(filter10Frame, variable=var10, command=cvikePlaveClick(var10)).grid(row=0, column=1)
    # *************************************

    # ************CUCLA DEVOJ************
    filter11Frame = Frame(stikeri, width=50, height=20)
    filter11Frame.grid(row=1, column=4, padx=10, pady=2)
    filter11 = PhotoImage(file='ikonice/rsz_cuclaseka.gif')

    labelaFil11 = Label(filter11Frame, image=filter11)
    labelaFil11.grid(row=0, column=0, padx=10, pady=2, sticky=W)
    labelaFil11.image = filter11

    var11 = IntVar()
    Checkbutton(filter11Frame, variable=var11, command=cuclaRozaClick(var11)).grid(row=0, column=1)
    # *************************************

def brkoviClick():
    global brkoviBool
    if var1.get()==1:
        brkoviBool=True
    else:
        brkoviBool = False
def svinjaClick():
    global svinjaBool
    if var2.get()==1:
        svinjaBool=True
    else:
        svinjaBool = False
def okoClick():
    global okoBool
    if var3.get()==1:
        okoBool=True
    else:
        okoBool = False
def pasClick():
    global pasBool
    if var4.get()==1:
        pasBool=True
    else:
        pasBool = False
def dalClick():
    global dalBool
    if var5.get()==1:
        dalBool=True
    else:
        dalBool = False
def cuclaPlavaClick():
    global cuclaPlavaBool
    if var6.get()==1:
        cuclaPlavaBool=True
    else:
        cuclaPlavaBool = False
def usneClick():
    global usneBool
    if var7.get()==1:
        usneBool=True
    else:
        usneBool = False
def klovnClick():
    global klovnBool
    if var8.get()==1:
        klovnBool=True
    else:
        klovnBool = False
def cvikeSrceClick():
    global cvikeSrcaBool
    if var9.get()==1:
        cvikeSrcaBool=True
    else:
        cvikeSrcaBool = False
def cvikePlaveClick():
    global cvikePlaveBool
    if var10.get()==1:
        cvikePlaveBool=True
    else:
        cvikePlaveBool = False
def cuclaRozaClick():
    global cuclaRozaBool
    if var11.get()==1:
        cuclaRozaBool=True
    else:
        cuclaRozaBool = False
def cartoonClick():
    global cartoonBool
    if var12.get()==1:
        cartoonBool=True
    else:
        cartoonBool = False
def kontraastClick():
    global kontrastBool
    kontrastBool=True
def polClick():
    global polBool,tekstPol
    polBool=True
    tekstPol=''
def starostClick():
    global starostBool,tekstStarost
    tekstStarost=''
    starostBool=True

predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


brkovi=cv2.imread('Dodaci/brkovi2.png',cv2.IMREAD_UNCHANGED)
cvike=cv2.imread('Dodaci/glasses.png',cv2.IMREAD_UNCHANGED)
cvikeSun=cv2.imread('Dodaci/sunglasses.png',cv2.IMREAD_UNCHANGED)
usne=cv2.imread('Dodaci/lips2.png',cv2.IMREAD_UNCHANGED)
nos=cv2.imread('Dodaci/klovn.png',cv2.IMREAD_UNCHANGED)
plaveCvike=cv2.imread('Dodaci/sunglassesBlue.png',cv2.IMREAD_UNCHANGED)
kruna=cv2.imread('Dodaci/kruna.png',cv2.IMREAD_UNCHANGED)
trepavica=cv2.imread('Dodaci/desna_trepuska.png',cv2.IMREAD_UNCHANGED)
zvezde=cv2.imread('Dodaci/zvezde.png',cv2.IMREAD_UNCHANGED)
oko=cv2.imread('Dodaci/oko.png',cv2.IMREAD_UNCHANGED)
svinja=cv2.imread('Dodaci/pig.png',cv2.IMREAD_UNCHANGED)
macaNos=cv2.imread('Dodaci/macaNos.png',cv2.IMREAD_UNCHANGED)
dalNos=cv2.imread('Dodaci/dalmatinacNjuska.png',cv2.IMREAD_UNCHANGED)

zecNos=cv2.imread('Dodaci/bunny.png',cv2.IMREAD_UNCHANGED)
cuclaPlava=cv2.imread('Dodaci/cuclabata.png',cv2.IMREAD_UNCHANGED)
cuclaRoza=cv2.imread('Dodaci/cuclasRoza.png',cv2.IMREAD_UNCHANGED)
dalDesnoUvo=cv2.imread('Dodaci/dalmatinacDesnoUvo.png',cv2.IMREAD_UNCHANGED)
dalLevoUvo=cv2.imread('Dodaci/dalmatinacLeoUvo.png',cv2.IMREAD_UNCHANGED)
dalNjuska=cv2.imread('Dodaci/dalmatinacNjuska.png',cv2.IMREAD_UNCHANGED)
dalJezik=cv2.imread('Dodaci/kerJezik.png',cv2.IMREAD_UNCHANGED)

kerDesnoUvo=cv2.imread('Dodaci/desnoUvoKer.png',cv2.IMREAD_UNCHANGED)
kerLevoUvo=cv2.imread('Dodaci/levoUvoKer.png',cv2.IMREAD_UNCHANGED)
kerNos=cv2.imread('Dodaci/kerNjuska.png',cv2.IMREAD_UNCHANGED)

brkoviBool=False
svinjaBool=False
okoBool=False
pasBool=False
dalBool=False
cuclaPlavaBool=False
usneBool=False
klovnBool=False
cvikeSrcaBool=False
cvikePlaveBool=False
cuclaRozaBool=False
cartoonBool=False
kontrastBool=False
polBool=False
starostBool=False

tekstPol=''
tekstStarost=''

'''cap=cv2.VideoCapture(0)
brojac=0



while cap.isOpened():
    if brojac%5==0:
        ret,frame=cap.read()
        matrica=prepoznavanjeLica.prepoznajLice(frame,detector,predictor)
        slika=stikeri.dodajNos(frame,slikaNosa,matrica)
        cv2.imshow(' ', slika)

    brojac+=1
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()'''
broj=0

root = Tk() #Makes the window
root.wm_title("Nasa aplikacija") #Naslov iznad prozora
root.config(background = "#FFFFFF", width=700, height=600) #postavljanje pozadine na rozu
root.resizable(0,0) #da ne moze da se uvecava prozor

#put widgets here
#LEVI DEO EKRANA, GDE CE SE PRIKAZIVATI VIDEO
leftFrame = Frame(root, width=500, height = 500)
leftFrame.grid(row=0, column=0, padx=10, pady=2)

width, height = 500, 500
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root.bind('<Escape>', lambda e: root.quit())
lmain = Label(leftFrame)
lmain.grid(row=0, column=0, padx=10, pady=2)
#lmain.pack()




def show_frame():
    #print broj
    global broj
    broj+=1
    _, frame = cap.read()
    #frame=cv2.resize(frame,(int(frame.shape[1]*0.7),int(frame.shape[0]*0.7)))
    frame = cv2.flip(frame, 1)
    #if broj==5:
        #n=provaMrezeStarost.sve(frame)
        #print n
    if brkoviBool | svinjaBool | okoBool | pasBool | dalBool | cuclaPlavaBool | usneBool | klovnBool |cvikeSrcaBool | cvikePlaveBool | cuclaRozaBool:
        matrica = prepoznavanjeLica.prepoznajLice(frame, detector, predictor)
        if brkoviBool:
            frame=st.dodajBrkove(frame,brkovi,matrica)
        if svinjaBool:
            frame= st.dodajNos(frame,svinja,matrica)
        if okoBool:
            frame = st.dodajTreceOko(frame,oko,matrica)
        if pasBool:
            frame = st.dodajDalmatinac(frame,kerNos,kerLevoUvo,kerDesnoUvo,dalJezik,matrica,1.2)
        if dalBool:
            frame = st.dodajDalmatinac(frame,dalNos,dalLevoUvo,dalDesnoUvo,dalJezik,matrica,1.5)
        if cuclaPlavaBool:
            frame = st.dodajUsne(frame,cuclaPlava,matrica)
        if cuclaRozaBool:
            frame = st.dodajUsne(frame,cuclaRoza,matrica)
        if usneBool:
            frame = st.dodajUsne(frame,usne,matrica)
        if klovnBool:
            frame = st.dodajNos(frame,nos,matrica)
        if cvikeSrcaBool:
            frame = st.dodajCvike(frame,cvikeSun,matrica)
        if cvikePlaveBool:
            frame = st.dodajPlaveCvike(frame,plaveCvike,matrica)
    if cartoonBool:
        frame= st.cartoon(frame)
    if kontrastBool:
        frame=st.kontrast(frame, kont)
    if polBool:
        global tekstPol,polBool

        labelaPol22 = Label(rightFrame, text='                           ')
        labelaPol22.grid(row=3, column=0)
        tekstPol=pol.prepoznaj(frame)
        labelaPol2 = Label(rightFrame, text=tekstPol)
        labelaPol2.grid(row=3, column=0)
        polBool=False
    if starostBool:
        global tekstStarost,starostBool

        labelaStarost111 = Label(rightFrame, text='                  ')
        labelaStarost111.grid(row=6, column=0)
        tekstStarost=provaMrezeStarost.odrediStarost(frame)
        labelaStarost1=Label(rightFrame,text=tekstStarost)
        labelaStarost1.grid(row=6,column=0)
        starostBool=False

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()

#DESNI DEO PROZORA GDE CE BITI DUGMICI...

rightFrame = Frame(root, width=200, height = 500)
rightFrame.grid(row=0, column=1, padx=10, pady=2)


stikeri= Frame(leftFrame,width=50, height = 20)
stikeri.grid(row=1, column=0, padx=10, pady=2,sticky=W)

#dodajCheckBoxes(stikeri)
# ************BRKOVI************
filter1Frame = Frame(stikeri, width=50, height=20)
filter1Frame.grid(row=0, column=0, padx=10, pady=2, sticky=W)
filter1 = PhotoImage(file='ikonice/rsz_brkovi2.gif')

labelaFil1 = Label(filter1Frame, image=filter1)
labelaFil1.grid(row=0, column=0, padx=10, pady=2, sticky=W)
labelaFil1.image = filter1

var1 = IntVar()
Checkbutton(filter1Frame, variable=var1, command=brkoviClick).grid(row=0, column=1)
# *************************************

# ************SVINJA************
filter2Frame = Frame(stikeri, width=50, height=20)
filter2Frame.grid(row=0, column=1, padx=10, pady=2)
filter2 = PhotoImage(file='ikonice/rsz_pig.gif')

labelaFil2 = Label(filter2Frame, image=filter2)
labelaFil2.grid(row=0, column=0, padx=10, pady=2, sticky=W)
labelaFil2.image = filter2

var2 = IntVar()
Checkbutton(filter2Frame, variable=var2, command=svinjaClick).grid(row=0, column=1)
# *************************************

# ************OKO************
filter3Frame = Frame(stikeri, width=50, height=20)
filter3Frame.grid(row=0, column=2, padx=10, pady=2)
filter3 = PhotoImage(file='ikonice/rsz_oko.gif')

labelaFil3 = Label(filter3Frame, image=filter3)
labelaFil3.grid(row=0, column=0, padx=10, pady=2, sticky=W)
labelaFil3.image = filter3

var3 = IntVar()
Checkbutton(filter3Frame, variable=var3, command=okoClick).grid(row=0, column=1)
# *************************************

# ************PAS************
filter4Frame = Frame(stikeri, width=50, height=20)
filter4Frame.grid(row=0, column=3, padx=10, pady=2)
filter4 = PhotoImage(file='ikonice/rsz_pas.gif')

labelaFil4 = Label(filter4Frame, image=filter4)
labelaFil4.grid(row=0, column=0, padx=10, pady=2, sticky=W)
labelaFil4.image = filter4

var4 = IntVar()
Checkbutton(filter4Frame, variable=var4, command=pasClick).grid(row=0, column=1)
# *************************************

# ************DALMATINAC************
filter5Frame = Frame(stikeri, width=50, height=20)
filter5Frame.grid(row=0, column=4, padx=10, pady=2)
filter5 = PhotoImage(file='ikonice/rsz_dalm.gif')

labelaFil5 = Label(filter5Frame, image=filter5)
labelaFil5.grid(row=0, column=0, padx=10, pady=2, sticky=W)
labelaFil5.image = filter5

var5 = IntVar()
Checkbutton(filter5Frame, variable=var5, command=dalClick).grid(row=0, column=1)
# *************************************

# ************CUCLADECAK************
filter6Frame = Frame(stikeri, width=50, height=20)
filter6Frame.grid(row=0, column=5, padx=10, pady=2)
filter6 = PhotoImage(file='ikonice/rsz_cuclabata.gif')

labelaFil6 = Label(filter6Frame, image=filter6)
labelaFil6.grid(row=0, column=0, padx=10, pady=2, sticky=W)
labelaFil6.image = filter6

var6 = IntVar()
Checkbutton(filter6Frame, variable=var6, command=cuclaPlavaClick).grid(row=0, column=1)
# *************************************

# ************USNE************
filter7Frame = Frame(stikeri, width=50, height=20)
filter7Frame.grid(row=1, column=0, padx=10, pady=2)
filter7 = PhotoImage(file='ikonice/rsz_lips2.gif')

labelaFil7 = Label(filter7Frame, image=filter7)
labelaFil7.grid(row=0, column=0, padx=10, pady=2, sticky=W)
labelaFil7.image = filter7

var7 = IntVar()
Checkbutton(filter7Frame, variable=var7, command=usneClick).grid(row=0, column=1)
# *************************************

# ************KLOVN************
filter8Frame = Frame(stikeri, width=50, height=20)
filter8Frame.grid(row=1, column=1, padx=10, pady=2)
filter8 = PhotoImage(file='ikonice/rsz_klovn.gif')

labelaFil8 = Label(filter8Frame, image=filter8)
labelaFil8.grid(row=0, column=0, padx=10, pady=2, sticky=W)
labelaFil8.image = filter8

var8 = IntVar()
Checkbutton(filter8Frame, variable=var8, command=klovnClick).grid(row=0, column=1)
# *************************************

# ************NAOCARE SRCA************
filter9Frame = Frame(stikeri, width=50, height=20)
filter9Frame.grid(row=1, column=2, padx=10, pady=2)
filter9 = PhotoImage(file='ikonice/rsz_sunglasses.gif')

labelaFil9 = Label(filter9Frame, image=filter9)
labelaFil9.grid(row=0, column=0, padx=10, pady=2, sticky=W)
labelaFil9.image = filter9

var9 = IntVar()
Checkbutton(filter9Frame, variable=var9, command=cvikeSrceClick).grid(row=0, column=1)
# *************************************

# ************NAOCARE PLAVE************
filter10Frame = Frame(stikeri, width=50, height=20)
filter10Frame.grid(row=1, column=3, padx=10, pady=2)
filter10 = PhotoImage(file='ikonice/rsz_sunglassesblue.gif')

labelaFil10 = Label(filter10Frame, image=filter10)
labelaFil10.grid(row=0, column=0, padx=10, pady=2, sticky=W)
labelaFil10.image = filter10

var10 = IntVar()
Checkbutton(filter10Frame, variable=var10, command=cvikePlaveClick).grid(row=0, column=1)
# *************************************

# ************CUCLA DEVOJ************
filter11Frame = Frame(stikeri, width=50, height=20)
filter11Frame.grid(row=1, column=4, padx=10, pady=2)
filter11 = PhotoImage(file='ikonice/rsz_cuclaseka.gif')

labelaFil11 = Label(filter11Frame, image=filter11)
labelaFil11.grid(row=0, column=0, padx=10, pady=2, sticky=W)
labelaFil11.image = filter11

var11 = IntVar()
Checkbutton(filter11Frame, variable=var11, command=cuclaRozaClick).grid(row=0, column=1)

#*********************FILTERI*******************

filteriFrame= Frame(rightFrame, width=200, height=100)
filteriFrame.grid(row=0, column=0, padx=2, pady=2)

labelaText=Label(filteriFrame,text='Izaberite filter: ')
labelaText.grid(sticky=N)
# ****************CARTOON*********************

filter12Frame = Frame(filteriFrame, width=50, height=20)
filter12Frame.grid(row=1, column=0, padx=10, pady=2,sticky=N)

var12 = IntVar()
Checkbutton(filter12Frame,text='CARTOON', variable=var12, command=cartoonClick).grid(row=0, column=1)
#*********************************************

#********************KONTRAST******************

kont=Scale(filteriFrame,label='KONTRAST', length=80, from_=1, to_=7, orient=HORIZONTAL, command=kontraastClick())
kont.grid(row=2, column=0, padx=10, pady=2,sticky=N)

#***********************************************


labelaPol= Label(rightFrame,text='Prepoznaj: ')
labelaPol.grid(row=1,column=0)
polDugme=Button(rightFrame,text='POL',command=polClick)
polDugme.grid(row=2,column=0)

labelaStarost= Label(rightFrame,text='Prepoznaj: ')
labelaStarost.grid(row=4,column=0)
starostDugme=Button(rightFrame,text='STAROST',command=starostClick)
starostDugme.grid(row=5,column=0)

root.mainloop()