import math
import numpy as np

class Camera_P:
    def __init__(self):
        self.exp_path = './handeye.txt'
        self.inp_path = './calibration.pckl'
        self.euler = np.zeros((4,4))
        self.read_text()
        self.cal_orn()
        self.cal_look()
        self.parameter = [self.thetax,self.thetay,self.thetaz,
                          self.look_x,self.look_y,self.dist,
                          self.x,self.y,self.z]
        np.save('params.npy',np.array(self.parameter))

    def read_text(self):
        with open(self.exp_path,'r') as f:
            for i,line in enumerate(f.readlines()):
                line = line.strip().split()
                for j,item in enumerate(line):
                    self.euler[i][j] = float(item)
        self.x = self.euler[0][3]
        self.y = -self.euler[1][3]
        self.z = self.euler[2][3]

    def cal_orn(self):
        self.thetax = math.atan2(self.euler[2][1],self.euler[2][2])*180/math.pi
        self.thetay = math.atan2(-self.euler[2][0],math.sqrt(self.euler[2][1]**2+self.euler[2][2]**2))*180/math.pi
        self.thetaz = math.atan2(self.euler[1][0],self.euler[0][0])*180/math.pi

    def cal_look(self):
        self.look_x = self.x + self.z*math.cos((-90-self.thetaz)*math.pi/180)/math.tan((-90-self.thetax)*math.pi/180)
        self.look_y = self.y + self.z*math.sin((-90-self.thetaz)*math.pi/180)/math.tan((-90-self.thetax)*math.pi/180)
        self.dist = math.sqrt((self.look_x-self.x)**2+(self.look_y-self.y)**2+self.z**2)

if __name__ == '__main__':
    camera = Camera_P()
    print(camera.thetax,camera.thetay,camera.thetaz)
    print(camera.look_x,camera.look_y,camera.dist)
    print(camera.x,camera.y,camera.z)
