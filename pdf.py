# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:09:03 2019
@author: Tushar goel
"""
import tkinter as tk
from PIL import Image, ImageTk
def deifning_user():
    global idd
    global name
    root1=tk.Toplevel(root)
    root1.geometry("1500x1000")
    label2 = tk.Label(root1, text="Enter Your Id",bg="white", font=('arial', 22))
    label2.grid()
    label3 = tk.Label(root1, text="Enter Your Name",bg="white", font=('arial', 22))
    label3.grid(row=1, column=0)
    idd = tk.StringVar()
    entry1 = tk.Entry(root1, textvariable=idd,bg="black",fg="red", font=('arial', 16))
    entry1.grid(row=0, column=1)
    name = tk.StringVar()
    entry2 = tk.Entry(root1, textvariable=name,bg="black",fg="red", font=('arial', 16))
    entry2.grid(row=1, column=1)
    root1.configure(background="white")
    button4= tk.Button(root1, text="Add Face",font=('arial', 20),bg="red", activebackground="blue",command=second_fast)
    button4.grid(row=3, column=0)
    button5= tk.Button(root1, text="Test Your Face",font=('arial', 20),bg="red", activebackground="blue",command=face_camera)
    button5.grid(row=6, column=0)
def second_fast():
    global idddd
    global nameee
    global checker
    checker="True"
    idddd=idd.get()
    nameee=name.get()
    print(idddd)
    print(nameee)
    face_detect(idddd)

def face_detect(idddd):
    import cv2 
    import numpy as np
    face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)
    i=0
    name=str(idddd)
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=face.detectMultiScale(gray,1.2,5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        for(x,y,w,h) in faces:
            i=i+1
            cv2.imwrite("datasete/User."+name +'.'+ str(i) + ".jpg", gray[y-50:y+h+50,x-50:x+w+50])
            cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
            cv2.imshow('im',im[y-50:y+h+50,x-50:x+w+50])
            cv2.waitKey(100)
        if i>20:
            cam.release()
            cv2.destroyAllWindows()
            break
    train_dataset()
def train_dataset():
    import os
    import cv2
    import numpy as np
    from PIL import Image
    face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    datapath='datasete'
    def get_images(datapath):
         image_paths = [os.path.join(datapath, f) for f in os.listdir(datapath)]
         images = []
         labels = []
         for i in image_paths:
             image_pil = Image.open(i).convert('L')
             image = np.array(image_pil, 'uint8')
             nbr = int(os.path.split(i)[-1].split(".")[1])
             print(nbr)
             faces = face.detectMultiScale(image)
             for (x, y, w, h) in faces:
                 images.append(image[y: y + h, x: x + w])
                 labels.append(nbr)
                 cv2.waitKey(10)
         return images, labels
    
    
    images, labels = get_images(datapath)
    cv2.imshow('test',images[0])
    cv2.waitKey(1)
    
    recognizer.train(images, np.array(labels))
    recognizer.save('recognizer/trainningData11.yml')
    cv2.destroyAllWindows()
def email():
    import smtpd
    from email.message import EmailMessage
    import smtpd,smtplib
    import imghdr
    
    gmailaddress = "enter your email"
    gmailpassword = "enter your password"
    
    msg=EmailMessage()
    msg['subject']='hi dear'
    msg['from']=gmailaddress
    msg['to']='tushargoel1099@gmail.com'
    msg.set_content('sjnslnsg')
    msg.set_content('An Unknown Person is Spotted')
    
    with open('D:\\python_idle\\face_recognization project\\unKnown\\yellow.jpg','rb') as f:
        file_data=f.read()
        file_type=imghdr.what(f.name)
        file_name=f.name
    
    msg.add_attachment(file_data,maintype='image',subtype=file_type,filename=file_name)
    
    
    
    
    with smtplib.SMTP_SSL('smtp.gmail.com',465) as c:
        c.login(gmailaddress,gmailpassword)
    
        c.send_message(msg)
        c.quit()

def face_camera():
    import cv2 
    import numpy as np
    from tkinter import messagebox
    face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer/trainningData11.yml")
    cap = cv2.VideoCapture(0)
    id=0
    name="yellow"
    fontface=cv2.FONT_HERSHEY_SIMPLEX
    while(cap.isOpened()):
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray,2.4,3)
        if face is ():
           print("no face")
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(127,0,0),3)
            id,conf=rec.predict(gray[y:y+h,x:x+w])
            if(conf<60):
                if(id==1):
                  id="tushar"
                if(checker=="True"):
                  if(id==idddd):
                     id=nameee
                else:
                    pass
            else:
                 id='nt known'
                 cv2.imwrite("unKnown/"+name +".jpg", frame)
            cv2.putText(frame,str(id)+"--", (x,y+h),fontface, 1.1, (0,255,0))
          
        cv2.imshow("frame",frame) 
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    cap.release()
    if(id=="nt known"):
        print("an unknown person is spotted")
        messagebox.showinfo("Warning", "person not known")
        email()
    else:
      messagebox.showinfo("cam-secure", "succesfully verified") 
    cv2.destroyAllWindows()    

def face_cam():
    from tkinter import messagebox
    import cv2 
    import numpy as np
    face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer/trainningData11.yml")
    cap = cv2.VideoCapture(0)
    id=0
    name="yellow"
    fontface=cv2.FONT_HERSHEY_SIMPLEX
    while(cap.isOpened()):
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray,2.4,3)
        if face is ():
           print("no face")
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(127,0,0),3)
            id,conf=rec.predict(gray[y:y+h,x:x+w])
            if(conf<60):
                if(id==1):
                  id="tushar"  
                else:
                    pass
            else:
                 id='nt known'
                 cv2.imwrite("unKnown/"+name +".jpg", frame)
            cv2.putText(frame,str(id)+"--", (x,y+h),fontface, 1.1, (0,255,0))
          
        cv2.imshow("frame",frame) 
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    cap.release()
    if(id=="nt known"):
        print("an unknown person is spotted")
        messagebox.showinfo("Warning", "person not known")
        email()
    else:
      messagebox.showinfo("cam-secure", "succesfully verified") 
    cv2.destroyAllWindows()    
        
root=tk.Tk()
root.geometry("1500x1000")
root.configure(background="black")
img = ImageTk.PhotoImage(Image.open("D:\\python_idle\\face_recognization project\\download.png"))
panel = tk.Label(root, image = img)
panel.pack(side = "top", fill = "both", expand = "yes")

label1 = tk.Label(text = "Welcome To The Cam-Secure",fg="red",bg="white",font=('arial', 20),pady=2,padx=2)
label1.pack()

button1 = tk.Button(root, text="Add Trusted Face",width=50,height=3,pady=3,padx=2,bg="yellow",activebackground="blue",font=('arial', 23),command=deifning_user)
button1.pack()

button2 = tk.Button(root,text = "Start Service",width=50,height=3,pady=3,padx=2,bg="yellow",activebackground="blue",font=('arial', 23),command=face_cam)
button2.pack()
root.mainloop()
