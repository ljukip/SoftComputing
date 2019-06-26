from funkcije import funkcija, invert, load, cv2, tf, cnn_model


file = open('out.txt','w')
file.write('ra244/2016 Ljubica Prelic')
file.write('\nfile  sum\n')




for x in range(10):
        naz = 'video-{}.avi'.format(x)


        [nizSlikeDodaj, nizSlikeOduzmi] = funkcija(naz)


        zaSabiranje = []
        zaOduzimanje = []
        for x in nizSlikeDodaj:
                if len(x) > 6 and len(x[1]) > 6:
                        promeniVelicinu = cv2.resize(x, (28, 28), interpolation = cv2.INTER_AREA)
                        invertovana = invert(promeniVelicinu)
                        resized = cv2.GaussianBlur(invertovana ,(5,5),0)  
                        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                        zaSabiranje.append(gray)
                        
                
        for x in nizSlikeOduzmi:
                if len(x) > 6 and len(x[1]) > 6:
                        promeniVelicinu = cv2.resize(x, (28, 28), interpolation = cv2.INTER_AREA)
                        invertovana = invert(promeniVelicinu)
                        resized = cv2.GaussianBlur(invertovana ,(5,5),0)  
                        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                        zaOduzimanje.append(gray)

        #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        #model5 = cnn_model(x_train,x_test,y_train,y_test)  
        model5 = load("modelConvo.h5") 

        plus = 0
        minus = 0


        for x in zaSabiranje:    
                pred = model5.predict(x.reshape(1, 28, 28, 1))
                plus += pred.argmax() 
                
        
        
        
        for x in zaOduzimanje:
                pred = model5.predict(x.reshape(1, 28, 28, 1))
                minus += pred.argmax()
                
        print(naz)
        print(plus, '-', minus)
        print(' =', plus - minus )
        print(*'**********')

        br = plus - minus
        ttt = naz + ' ' + str(br) + '\n'
        file.write(ttt)


file.close()