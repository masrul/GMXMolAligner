import numpy as np
import math
from copy import deepcopy


class Aligner:
    def __init__(self, coordFile):
        self.coordFile = coordFile
        self._natoms = 0
        self._symbols = []
        self._resids = []
        self._atomids = []
        self._resnames = []
        self._x = []
        self._y = []
        self._z = []
        self._box = []
        self.read()

    def copy(self):
        return deepcopy(self)

    def read(self):
        coordFH = open(self.coordFile, "r")
        lines = coordFH.readlines()
        coordFH.close()

        self._natoms = int(lines[1])

        for line in lines[2:-1]:
            resid = int(line[0:5])
            resname = line[5:10]
            symbol = line[10:15]
            atomid = int(line[15:20])
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])

            self._resids.append(resid)
            self._resnames.append(resname)
            self._symbols.append(symbol)
            self._atomids.append(atomid)
            self._x.append(x)
            self._y.append(y)
            self._z.append(z)

        self._box = [float(Str) for Str in lines[-1].split()[0:3]]

    def align(self, iatom=None, jatom=None, target_dir=None):

        if not iatom or not jatom:
            iatom, jatom = self.findMoleculeAxis()

        if not target_dir:
            target_dir = [0.0, 0.0, 1.0]

        coords = np.zeros((3, self._natoms), dtype=float)
        coords[0, :] = self._x
        coords[1, :] = self._y
        coords[2, :] = self._z

        molAxis = [
            coords[0, iatom - 1] - coords[0, jatom - 1],
            coords[1, iatom - 1] - coords[1, jatom - 1],
            coords[2, iatom - 1] - coords[2, jatom - 1],
        ]
        coords = do_align(molAxis, target_dir, coords)

        self._x = list(coords[0, :])
        self._y = list(coords[1, :])
        self._z = list(coords[2, :])

    def rotate(self, iatom=None, jatom=None, angle=None):

        if not iatom or not jatom:
            iatom, jatom = self.findMoleculeAxis()

        if not angle:
            angle = 5.0

        coords = np.zeros((3, self._natoms), dtype=float)
        coords[0, :] = self._x
        coords[1, :] = self._y
        coords[2, :] = self._z

        molAxis = [
            coords[0, iatom - 1] - coords[0, jatom - 1],
            coords[1, iatom - 1] - coords[1, jatom - 1],
            coords[2, iatom - 1] - coords[2, jatom - 1],
        ]

        rotMat = getRotMatrix(molAxis, angle)
        coords = np.matmul(rotMat, coords)

        self._x = list(coords[0, :])
        self._y = list(coords[1, :])
        self._z = list(coords[2, :])

    def write(self, outFile, boxLen=None):
        if boxLen:
            lx, ly, lz = boxLen
        else:
            lx, ly, lz = self._box

        outFH = open(outFile, "w")
        outFH.write("%s\n" % ("Created by Masrul Huda"))
        outFH.write("%-10d\n" % (self.natoms))

        groFMT = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"
        for i in range(self.natoms):
            outFH.write(
                groFMT
                % (
                    self.resids[i],
                    self.resnames[i],
                    self.symbols[i],
                    self.atomids[i],
                    self.x[i],
                    self.y[i],
                    self.z[i],
                )
            )
        outFH.write("%10.5f%10.5f%10.5f\n" % (lx, ly, lz))
        outFH.close()

    def moveTo(self, transPos, atomID=None):

        """
        tranPos: coordinate of target point, where COM or atomID will
         be moved  
        """

        if atomID:
            xref = self.x[atomID - 1]
            yref = self.y[atomID - 1]
            zref = self.z[atomID - 1]
        else:
            xcom = sum(self._x) / self._natoms
            ycom = sum(self._y) / self._natoms
            zcom = sum(self._z) / self._natoms

            xref = xcom
            yref = ycom
            zref = zcom

        for i in range(self._natoms):
            self._x[i] = self._x[i] - xref + transPos[0]
            self._y[i] = self._y[i] - yref + transPos[1]
            self._z[i] = self._z[i] - zref + transPos[2]

    def moveBy(self, transVector):

        for i in range(self.natoms):
            self.x[i] += transVector[0]
            self.y[i] += transVector[1]
            self.z[i] += transVector[2]

    def findMoleculeAxis(self):

        pairs = []
        dist = []

        for i in range(self._natoms - 1):
            for j in range(i + 1, self._natoms):
                dx = self._x[i] - self._x[j]
                dy = self._y[i] - self._y[j]
                dz = self._z[i] - self._z[j]

                dist.append(dx ** 2 + dy ** 2 + dz ** 2)
                pairs.append((i, j))

        argmax = np.argmax(dist)

        return pairs[argmax][0] + 1, pairs[argmax][1] + 1

    def merge(self, other):

        self._natoms += other._natoms
        self._symbols += other._symbols
        self._resids += [resid + self._resids[-1] for resid in other._resids]
        self._atomids += [atomid + self._atomids[-1] for atomid in other._atomids]
        self._resnames += other._resnames
        self._x += other._x
        self._y += other._y
        self._z += other._z

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def natoms(self):
        return self._natoms

    @property
    def symbols(self):
        return self._symbols

    @property
    def resnames(self):
        return self._resnames

    @property
    def atomids(self):
        return self._atomids

    @property
    def resids(self):
        return self._resids

    @property
    def box(self):
        return self._box


def getAlignMatrix(v, c):

    alignMat = np.zeros((3, 3), dtype=float)
    iMat = np.zeros((3, 3), dtype=float)
    vx = np.zeros((3, 3), dtype=float)

    iMat[0, 0] = 1.0
    iMat[1, 1] = 1.0
    iMat[2, 2] = 1.0

    vx[0, 0] = 0.0
    vx[0, 1] = -v[2]
    vx[0, 2] = v[1]

    vx[1, 0] = v[2]
    vx[1, 1] = 0.0
    vx[1, 2] = -v[0]

    vx[2, 0] = -v[1]
    vx[2, 1] = v[0]
    vx[2, 2] = 0.0

    factor = 1.0 / (1.0 + c)

    alignMat = iMat + vx + np.matmul(vx, vx) * factor

    return alignMat


def getRotMatrix(rotAxis, angle):
    rotMat = np.zeros((3, 3), dtype=float)

    angle = angle * 0.0174533  # degree to radian

    cos = math.cos(angle)
    sin = math.sin(angle)
    _cos_ = 1 - cos

    u = rotAxis / np.linalg.norm(rotAxis)

    ux = u[0]
    uy = u[1]
    uz = u[2]

    uxy = ux * uy
    uyz = uy * uz
    uzx = uz * ux
    uxx = ux * ux
    uyy = uy * uy
    uzz = uz * uz

    rotMat[0, 0] = cos + (uxx * _cos_)
    rotMat[0, 1] = (uxy * _cos_) - (uz * sin)
    rotMat[0, 2] = (uzx * _cos_) + (uy * sin)

    rotMat[1, 0] = (uxy * _cos_) + (uz * sin)
    rotMat[1, 1] = cos + (uyy * _cos_)
    rotMat[1, 2] = (uyz * _cos_) - (ux * sin)

    rotMat[2, 0] = (uzx * _cos_) - (uy * sin)
    rotMat[2, 1] = (uyz * _cos_) + (ux * sin)
    rotMat[2, 2] = cos + (uzz * _cos_)

    return rotMat


def do_align(u, v, coords):

    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    normal = np.cross(u, v)
    c = np.dot(u, v)

    if abs(abs(c) - 1) < 10e-10:  # if vectors are antiparallel
        coords = coords * -1
        return coords

    alignMat = getAlignMatrix(normal, c)
    coords = np.matmul(alignMat, coords)

    return coords
