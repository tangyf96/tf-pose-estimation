import cv2
import math
image1=cv2.imread('images/xahid_youya/input/moto_left1.jpg')
image2=cv2.imread('images/xahid_youya/input/moto_right1.jpg')
#image1=cv2.cvtColor(image11, cv2.COLOR_BGR2GRAY)
#image2=cv2.cvtColor(image22, cv2.COLOR_BGR2GRAY)
v1=image1[664,918]
r,g,b=image1[664,918]
print(v1)
def manhattan_distance1(v1,v2):
    error=abs(int(v1)-int(v2))
    return error
def manhattan_distance2(v1,v2):
    r,g,b=v1
    r1,g1,b1=v2
    error=abs(int(r)-int(r1))+abs(int(g)-int(g1))+abs(int(b)-int(b1))
    return error
x=950-20
y=662-20
error=100000
truex=x
truey=y
for i in range(0,25):
    for j in range (0,25):
        y2,x2=[y+i,x+j]
        #v2=image2[y2,x2]
        sum=0
        for m in range(-11,12):
            for n in range(-11,12):
                #error1=image2[y2]
                    #error1=manhattan_distance1(v1,v2)
                    r3,g3,b3=image1[664+n,918+m]
                    r2,g2,b2=image2[y2+n,x2+m]
                    if m==0 and n==0:
                        sum=sum+int(manhattan_distance2((r3,g3,b3),(r2,g2,b2)))
                    else:
                        sum=sum+0.5*int(manhattan_distance2((r3,g3,b3),(r2,g2,b2)))

                #error2=manhattan_distance2((r,g,b),(r2,g2,b2))
        if sum<error:
            truex=x2
            truey=y2
        error=min(error,sum)
cv2.circle(image1,(918,664),1,(0,250,0),1)
cv2.imwrite("image10.jpg",image1)
cv2.circle(image2,(truex,truey),1,(0,250,0),5)
cv2.imwrite("image11.jpg",image2)
#print(image2[588,994])
print(truex,truey)
print(error)
