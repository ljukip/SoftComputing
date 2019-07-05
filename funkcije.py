import numpy as np
import cv2
import matplotlib.pyplot as plt 
import math

from keras.models import Sequential
from keras.models import load_model
from keras.models import save_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D,Activation
from keras.optimizers import SGD

import tensorflow as tf

from scipy.spatial import distance as dist
from collections import OrderedDict

kernel = np.ones((3, 3))

nizSlikeZelena = []
nizSlikePlava = []
konture = []


"""
    Funkcija koja za zadatu sliku vraca koordinate linija        
"""
def getLine(frame):
    minLineLength = 100
    maxLineGap = 10

    zelena = frame.copy()
    plava = frame.copy()

    zelena[:, :, 0] = 0
    plava[:, :, 1] = 0

    sivaZelena = cv2.cvtColor(zelena, cv2.COLOR_BGR2GRAY)
    sivaPlava = cv2.cvtColor(plava, cv2.COLOR_BGR2GRAY)

    _, zelenaThresh = cv2.threshold(sivaZelena, 25, 255, cv2.THRESH_BINARY)
    _, plavaThresh = cv2.threshold(sivaPlava, 25, 255, cv2.THRESH_BINARY)

    zeleneLinije = cv2.HoughLinesP(zelenaThresh, 1, np.pi / 180, 100, minLineLength, maxLineGap) # sika-grayscale, 1 piksel, 1 stepen u radijanima- PI / 180, tresh, minim br piksela koju mogu da formiraju liniju, minimalni razmak izmedju 2 linije
    plaveLinije = cv2.HoughLinesP(plavaThresh, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    x1 = min(zeleneLinije[:, 0, 0])
    y1 = max(zeleneLinije[:, 0, 1])
    x2 = max(zeleneLinije[:, 0, 2])
    y2 = min(zeleneLinije[:, 0, 3])


    a1 = min(plaveLinije[:, 0, 0])
    b1 = max(plaveLinije[:, 0, 1])
    a2 = max(plaveLinije[:, 0, 2])
    b2 = min(plaveLinije[:, 0, 3])


    return [(x1, y1), (x2, y2), (a1, b1), (a2, b2)]

def funkcija(naziv):
	del nizSlikeZelena[:]
	del nizSlikePlava[:]
	video = cv2.VideoCapture(naziv)
	ret,frame = video.read()
	

	kopijaSlike = frame.copy()
	koordinateLinija = getLine(kopijaSlike)

	ct = NumberTracker()

	while True : #beskonacna petlja
		ret,frame = video.read()
		if ret == False: # ako nije ucitan frejm, kraj
				break
		

		temp = frame.copy()
		temp[:, :, 1] = 0 # uklonim zelenu komponentu jer pravi problem sa konturama



		gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
		thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, kernel, iterations=2)
		contours, _hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		konture = []
		for c in contours:
			(x, y, w, h) = cv2.boundingRect(c)
			if w < 5 or h < 5: # ako je kontura "mala", preskoci je
				continue
			""" cv2.circle(frame, (x, y), 5, (0, 0, 255), 3) """

			# za zelenu liniju
			x1z = koordinateLinija[0][0]
			y1z = koordinateLinija[0][1]
			x2z = koordinateLinija[1][0]
			y2z = koordinateLinija[1][1]

			# za plavu liniju
			x1p = koordinateLinija[2][0]
			y1p = koordinateLinija[2][1]
			x2p = koordinateLinija[3][0]
			y2p = koordinateLinija[3][1]

			# jednacina prave kroz dve tacke
			# zelena prava
			kz = (y2z - y1z)
			kz = kz / (x2z - x1z)
			nz = -1 * x1z * (y2z - y1z)
			nz = nz / (x2z - x1z)
			nz = nz + y1z

			yyzIznad = kz * x + nz  
			yyzIspod = kz * (x + w) + nz    

			# plava prava
			kp = (y2p - y1p)
			kp = kp / (x2p - x1p)
			np = -1 * x1p * (y2p - y1p)
			np = np / (x2p - x1p)
			np = np + y1p

			yypIznad = kp * x + np # vrednost gornjeg levog ugla konture na liniji
			yypIspod = kp * (x + w) + np # vrednost gornjeg desnog ugla konture na liniji
			
			presekZelena = False
			presekPlava = False

			if y <= yyzIznad and y + h >= yyzIspod and x >= x1z and x <= x2z: # ako dodiruje zelenu liniju
				""" cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) """
				presekZelena = True
			if y <= yypIznad and y + h >= yypIspod and x >= x1p and x <= x2p: # ako dodiruje plavu liniju
				""" cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) """
				presekPlava = True
			
			
			konture.append([x, y, x + w, y + h, presekZelena, presekPlava])
			""" cv2.circle(frame, (koordinateLinija[0][0], koordinateLinija[0][1]), 1, (255, 255, 255), 5)
			cv2.circle(frame, (koordinateLinija[1][0], koordinateLinija[1][1]), 1, (255, 255, 255), 5) """
		""" for elem in koordinateLinija:
			cv2.circle(frame, (elem[0], elem[1]), 1, (255, 255, 255), 5) """
		""" cv2.imshow('neki naziv frejma',frame)
		cv2.waitKey(50) """


		_objects = ct.update(konture, frame)


		
		""" for (objectID, centroid) in objects.items():
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
			print(objectID)         
			cv2.imshow("Frame", frame)
			cv2.waitKey(1) 
		"""
	

	video.release()
	cv2.destroyAllWindows()
	return [nizSlikePlava
, nizSlikeZelena]

class NumberTracker():
	def __init__(self, maxDisappeared=10):
		# maxDisappeared -na koliko uzastopnih frejmova se 
        # cifra ne smo pojaviti da bi bila izbrisana
		# U recnicima ce se nalaziti koordinate, slika, flag za dodorivanje zelene i/ili
        # plave linije, gde ce kljuc biti ID
		self.nextObjectID = 0 # poput statickog polja u klasi
		self.objects = OrderedDict()# recnik koji pamti u kom redosledu su dodavani clanovi, objects = (x, y)
		self.disappeared = OrderedDict()# za svaki id se broji koliko ga nije bilo
		self.slike = OrderedDict()# za svaki id se cuva slika
		self.zelenaLinija = OrderedDict()# da li je dodirnuo liniju
		self.plavaLinija = OrderedDict()          
		# za svaku cifru se vodi evidencija o broju frejmova na kojima se nije pojavila, zarad uklanjanja
		self.maxDisappeared = maxDisappeared # posle koliko frejmova je potrebno ukloniti
	#koor = x, y, w, h
	def register(self, centroid, frame, koor): # ima centroid -  x y, frame je slika, koor su x, y, x + w, y + h
		# dodavanje nove cifre u tracker
		       
		self.slike[self.nextObjectID] = frame[koor[1] : koor[3], koor[0] : koor[2]]#isecanje slike
		self.objects[self.nextObjectID] = centroid # (x, y)
		self.disappeared[self.nextObjectID] = 0
		self.plavaLinija[self.nextObjectID] = False
		self.zelenaLinija[self.nextObjectID] = False
		self.nextObjectID += 1

	def deregister(self, objectID):
		# uklanjanje iz evidencije
               
		del self.objects[objectID]
		del self.disappeared[objectID]
		del self.slike[objectID]
		del self.zelenaLinija[objectID]    
		del self.plavaLinija[objectID] 
        
        		
		     
	def update(self, rects, frame):#rect su konture, frame je slika na kojoj su konture pronadjene
		if len(rects) == 0:
			# ako nema kontura, povecaj svima broj frejmova na kojima ih nije bilo
			for objectID in self.disappeared.keys():
				self.disappeared[objectID] += 1

				# ukoliko se cifra nije pojavila dovoljno puta uzastupno
                # ukloni je
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# posto nema ko0ntura, vrati se
			return self.objects

		# inicijalizacija za cuvanje podataka
		inputCentroids = np.zeros((len(rects), 2), dtype="int") #za cuvanje centroida  - (x, y)
		koordinate = [None] * len(rects) # za cuvanje koordinata
		nizPresloZelenu = [None] * len(rects)        
		nizPresloPlavu = [None] * len(rects)        
		# iteriranje kroz konture
		for (i, (startX, startY, endX, endY, ze, pl)) in enumerate(rects):
			# cuvanje podataka u lokalne promenljive
			nizPresloZelenu[i] = ze
			nizPresloPlavu[i] = pl                
			koordinate[i] = (startX, startY, endX, endY)            
			inputCentroids[i] = (startX, startY)

		# ako ne pratimo cifre, sve konture ubaci kao nove
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i], frame, koordinate[i])
				

        # u suprotnom, imamo cifre koje pratimo,
        # neophodno je "prespojiti" cifre, na osnovu euklidskog rastojanja
		else:
			objectIDs = list(self.objects.keys()) # id-ijevi
			objectCentroids = list(self.objects.values()) # centroidi

            # racunanje distance izmedju svakog para pracenih cifara i ulaznih kontura
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # prvi korak prespajanja cifara, pronalazenje
            # najmanje vrednosti iz svakog reda, a zatim
            # sortirati redove na osnovu njihovih najmanjih vrednosti, tako
            # da red sa najmanjom vrednosti bude "napred"
			rows = D.min(axis=1).argsort()

			# isto, samo za kolone
			cols = D.argmin(axis=1)[rows]

            # za pracenje iskoriscenih redova i kolona
			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				# ako sam vec proverio ovaj red/kolonu, dalje
				if row in usedRows or col in usedCols:
					continue

                # u suprotnim, uzimam sledecu cifru, restartujem joj br frejmova na kojimae nije bilo
				# nasao sam neku konturu koja ima najmanje rastojanje sa nekom cifrom koju smo pratili
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col] #azuriram x i y komponentu
				if self.plavaLinija[objectID] == False and nizPresloPlavu[col] == True: # ako cifra nije bila presla liniju a sada joj kazem da je presla
								self.plavaLinija[objectID] = True
								   
								nizSlikePlava.append(self.slike[objectID])                                
				if self.zelenaLinija[objectID] == False and nizPresloZelenu[col] == True:
								self.zelenaLinija[objectID] = True  
								   
								nizSlikeZelena.append(self.slike[objectID])                                  
				self.disappeared[objectID] = 0

				# iskoristio sam red i kolonu
				usedRows.add(row)
				usedCols.add(col)

			# odredi kolone i redove koje jos uvek nisam iskoristio
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # ako je broj cifara koje pratim veci ili jednak broju ulaznih 
            # kontura, treba proveriti da li je neki objekat nestao11
			if D.shape[0] >= D.shape[1]:
				# iteriraj kroz neiskorisc redove
				for row in unusedRows:
					# povecaj im brojac
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# ako cifra dovoljno dugo nije bila tu, izbaci je
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

            # ako je broj ulaznih kontura veci nego broj cifara koje pratim
            # sve preostale je potrebno ubaciti u evidenciju
			else:
				for col in unusedCols:
					self.register(inputCentroids[col], frame, koordinate[col])
					                  
                    

		# vrati objekte
		return self.objects

def cnn_model(x_train,x_test,y_train,y_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) #prebaci podatke u oblik koji treba mrezi, prvi argument je broj slika, zatim dimenzija1 , dimenzija2, 1 jer ima samo jedan kanal boja -grayscale 
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) # isto

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255 # Nnormalizovanje rgb vrednosti deljenjem
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of slikes in x_train', x_train.shape[0])
    print('Number of slikes in x_test', x_test.shape[0])
    
    model = Sequential() # sekvencijalni model, pravi se sloj po sloj,,sa add() dodajemo sloj
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1))) # relu == Rectified Linear Activation { if input > 0: return input else: return 0 }
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten()) # veza izmedju conv i dense sloja, prebacivanje iz matrice u niz
    model.add(Dense(10, activation='softmax')) # aktivacija je softmax, suma izlaza je 1, pa izlaze mozemo da posmatramo kao verovatnoce
    # ADAM prilagodjava ucenje tokom treninga, tj koliko brzo se dodje do optimalnih tezina
	# sparse categorical crossentropy - najcesca, sto je manja, to je bolje
	# metrika koja se koristi je accuracy
    model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs=10)
    model.evaluate(x_test, y_test)
    
    model.save('modelConvo.h5')
    
    return model    



def loadModel():
    model = None
    try:
        model = load_model('modelConvo.h5')
        if model is None:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()#podeli u trening i test podatke
            model = cnn_model(x_train,x_test,y_train,y_test)
    except NameError:
        print('Cant find model')
    
    return model    
def load(path):
    model = load_model(path)
    return model


def reshape_data(input_data):
    # transformisemo u oblik pogodan za scikit-learn
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))    

def invert(img):
    return 255 - img


def dilate(slike):
    kernel = np.ones((2,2)) # strukturni element 3x3 blok
    return cv2.dilate(slike, kernel, iterations=1)
