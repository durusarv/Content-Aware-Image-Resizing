# -*- coding: utf-8 -*-
"""
@author: durusarv and jdlips
"""

import numpy as np
import cv2


class SeamCarver:
    """
    init
    Initializes the SeamCarver object
    @params:    inputFilename: the path and filename of the input image
                outputFilename: the path and filename of the output image
                outputWidth: the final width of the output image
                outputHeight: the final height of the output image
                demo: boolean that controls if we output images at each step
                    (default is False)
    """
    def __init__(self, inputFilename, outputFilename, outputWidth, outputHeight, demo=False):
        # Setting input parameters
        self.inputFilename = inputFilename
        self.inputImg = cv2.imread(inputFilename)
        self.inputHeight = np.size(self.inputImg, 0)
        self.inputWidth = np.size(self.inputImg, 1)

        # Setting output parameters
        self.outputFilename = outputFilename
        self.outputImg = np.copy(self.inputImg)
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight

        # Setting other parameters
        self.count = 0
        self.stepImg = np.copy(self.inputImg)
        self.delta = (np.absolute(self.inputWidth - self.outputWidth) + np.absolute(self.inputHeight - self.outputHeight))
        self.percentDone = 0.00001
        self.prevPercentDone = self.percentDone
        self.demo = demo
        self.rotated = False

    """
    seamCarving
    Controls the workflow of the entire algorithm
    """
    def seamCarving(self):
        colSeams = self.inputWidth - self.outputWidth
        # Checking if we are removing seams or adding them to the width
        if colSeams != 0:
            if colSeams > 0:
                self.removeSeams(colSeams)
            elif colSeams < 0:
                self.addSeams(-1 * colSeams)

        rowSeams = self.inputHeight - self.outputHeight
        # Checking if we are removing seams or adding them to the height
        if rowSeams != 0:
            self.outputImg = cv2.rotate(self.outputImg, 0)
            self.outputHeight = np.size(self.outputImg, 0)
            self.outputWidth = np.size(self.outputImg, 1)
            self.stepImg = np.copy(self.outputImg)
            self.rotated = True
            
            if rowSeams > 0:
                self.removeSeams(rowSeams)
            elif rowSeams < 0:
                self.addSeams(-1 * rowSeams)
    """
    removeSeams
    Controls the workflow of removing pixel layers (seam) to the images
    @params:    seams: the number of seams to be removed
    """
    def removeSeams(self, seams):
        count = 0
        
        while count < seams:
            energyMap = self.getEnergyMap()
            energyValuesDown = self.getCumulativeMaps(energyMap)
            leastEnergySeam = self.getLeastEnergySeam(energyValuesDown[0])
            
            if (self.demo):
                self.demoSteps(leastEnergySeam)

            row, col = self.outputImg.shape[: 2]
            output = np.zeros((row, col - 1, 3))
            
            for currentRow in range(row):
                currentColumn = leastEnergySeam[currentRow]
                for i in range(3):
                    output[currentRow, :, i] = np.delete(self.outputImg[currentRow, :, i], [currentColumn])
                    
            self.outputImg = np.copy(output)
            self.stepImg = np.copy(self.outputImg)
            self.percentDone = (self.count/self.delta)
            self.printPercentDone()
            count += 1
    """
    addSeams
    Controls the workflow of adding pixel layers (seam) to the images
    @params:    seams: the number of seams to be added
    """
    def addSeams(self, seams):
        count = 0
        
        while count < seams:
            energyMap = self.getEnergyMap()
            energyValuesDown = self.getCumulativeMaps(energyMap)
            leastEnergySeam = self.getLeastEnergySeam(energyValuesDown[1])

            if (self.demo):
                self.demoSteps(leastEnergySeam)
                
            row, col = self.outputImg.shape[: 2]
            output = np.zeros((row, col + 1, 3))
            outputImg = self.outputImg
            
            for currentRow in range(row):
                currentColumn = leastEnergySeam[currentRow]
                
                for i in range(3):
                    if currentColumn == 0:
                        output[currentRow, currentColumn, i] = outputImg[currentRow, currentColumn, i]
                        output[currentRow, currentColumn + 1, i] = (outputImg[currentRow, currentColumn, i] + \
                                                                    outputImg[currentRow, currentColumn + 1, i] + \
                                                                    outputImg[currentRow, currentColumn + 2, i]) / 3
                        output[currentRow, currentColumn + 1:, i] = outputImg[currentRow, currentColumn:, i]
                    else:
                        output[currentRow, :currentColumn, i] = outputImg[currentRow, :currentColumn, i]
                        output[currentRow, currentColumn, i] = (outputImg[currentRow, currentColumn - 1, i] + \
                                                                    outputImg[currentRow, currentColumn, i] + \
                                                                    outputImg[currentRow, currentColumn + 1, i]) / 3
                        output[currentRow, currentColumn + 1:, i] = outputImg[currentRow, currentColumn:, i]

            self.outputImg = np.copy(output)
            self.percentDone = (self.count/self.delta)
            self.printPercentDone()
            count += 1

    """
    getEnergyMap
    Generates an energy map of the image based on the partial derivatives in the
    x and y channels for each pixel and each channel, then combines them
    @returns:   energyMap: The energy map of the image
    """
    def getEnergyMap(self):

        blue, green, red = self.split_channels()

        blueEnergy = np.absolute(cv2.Scharr(blue, -1, 1, 0)) + np.absolute(cv2.Scharr(blue, -1, 0, 1))
        greenEnergy = np.absolute(cv2.Scharr(green, -1, 1, 0)) + np.absolute(cv2.Scharr(green, -1, 0, 1))
        redEnergy = np.absolute(cv2.Scharr(red, -1, 1, 0)) + np.absolute(cv2.Scharr(red, -1, 0, 1))
        energyMap = blueEnergy + greenEnergy + redEnergy
        return energyMap

    """
    getCumulativeMaps
    Gets a cummulative energy map of the image from a non cummulative energy map
    of the image
    @params:    energyMap: a non cummulative energy map of the photo
    @returns:   mins:
                backtrack:
    """
    def getCumulativeMaps(self, energyMap):

        blue, green, red = self.split_channels()

        xKernel = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]])
        yLeftKernel = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
        yRightKernel = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]])

        xNeighbors = np.absolute(cv2.filter2D(blue, -1, kernel=xKernel)) + \
                 np.absolute(cv2.filter2D(green, -1, kernel=xKernel)) + \
                 np.absolute(cv2.filter2D(red, -1, kernel=xKernel))

        yLeft = np.absolute(cv2.filter2D(blue, -1, kernel=yLeftKernel)) + \
                 np.absolute(cv2.filter2D(green, -1, kernel=yLeftKernel)) + \
                 np.absolute(cv2.filter2D(red, -1, kernel=yLeftKernel))

        yRight = np.absolute(cv2.filter2D(blue, -1, kernel=yRightKernel)) + \
                 np.absolute(cv2.filter2D(green, -1, kernel=yRightKernel)) + \
                 np.absolute(cv2.filter2D(red, -1, kernel=yRightKernel))

        mins = np.copy(energyMap)
        backtrack = np.copy(energyMap)
        row, col = energyMap.shape

        for r in range(1, row):
            for c in range(0, col):
                if c == 0:
                    mins[r, c] = energyMap[r, c] + min(mins[r - 1, c + 1] + xNeighbors[r - 1, c + 1] + \
                                                        yRight[r - 1, c + 1],
                                                        mins[r - 1, c] + xNeighbors[r - 1, c])
                elif c == col - 1:
                    mins[r, c] = energyMap[r, c] + min(mins[r - 1, c - 1] + xNeighbors[r - 1, c - 1] + \
                                                        yLeft[r - 1, c - 1],
                                                        mins[r - 1, c] + xNeighbors[r - 1, c])
                else:
                    mins[r, c] = energyMap[r, c] + min(mins[r - 1, c - 1] + xNeighbors[r - 1, c - 1] + \
                                                        yLeft[r - 1, c - 1],
                                                        mins[r - 1, c + 1] + xNeighbors[r - 1, c + 1] + \
                                                        yRight[r - 1, c + 1],
                                                        mins[r - 1, c] + xNeighbors[r - 1, c])
                backtrack[r, c] = energyMap[r, c] + np.amin(backtrack[r - 1, max(c - 1, 0): min(c + 2, col - 1)])

        return mins, backtrack

    """
    getLeastEnergySeam
    Returns the least energy seam in an image from a map of all of the seams
    @params:    energyValuesDown: a map of cummulative energy values (seams) in
                the image
    @returns:   lis: The least energy value seam in the image
    """
    def getLeastEnergySeam(self, energyValuesDown):
        m, n = energyValuesDown.shape
        lis = np.zeros((m,), dtype=np.uint32)
        lis[-1] = np.argmin(energyValuesDown[-1])
        for row in range(m - 2, -1, -1):
            prv_x = lis[row + 1]
            if prv_x == 0:
                lis[row] = np.argmin(energyValuesDown[row, : 2])
            else:
                lis[row] = np.argmin(energyValuesDown[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return lis

    """
    split_channels
    Splits an image into red, blue, and green channels
    @returns:   blue: the blue channel of the image
                green: the green channel of the image
                red: the red channel of the image
    """
    def split_channels(self):

        blue = self.outputImg[:,:,0]
        green = self.outputImg[:,:,1]
        red = self.outputImg[:,:,2]

        return blue, green, red

    """
    outputImageToFile
    Outputs image to a file, and rotates it if necesary
    @params:    filename: the filename and path where the image will be stored
                img: the image itself
    """
    def outputImageToFile(self, filename, img):
        if (self.rotated):
            img = cv2.rotate(img, 2)
            cv2.imwrite(filename, img)
        else:
            cv2.imwrite(filename, img)

    """
    printPercentDone
    Prints what percent done the program is with resizing the image, in a
    minimum of 0.01% intervals
    """
    def printPercentDone(self):
        if (self.percentDone >= self.prevPercentDone + 0.0001):
            self.prevPercentDone = round(self.percentDone, 4)
            print(str(round((self.prevPercentDone * 100), 2)) + "%")

    """
    demoSteps
    Outputs all steps in the process of resizing an image as other .jpg images
    with a red line to represent the seam to be deleted or added
    @params:    leastEnergySeam: the seam that has the lowest energy in the
                photo
    """
    def demoSteps(self, leastEnergySeam):
        self.stepImg = np.copy(self.outputImg)
        row, col = self.outputImg.shape[: 2]
        outputStep = self.stepImg
        for r in range(row):
            c = leastEnergySeam[r]
            self.stepImg[r,c] = [0, 0, 255]
        self.outputImageToFile("output/steps/iceberg_" + str(self.outputWidth) + "x"+ str(self.outputHeight) + "/iceberg_" + str(self.outputWidth) + "x" + str(self.outputHeight) + "_" + str(self.count) + ".jpg", self.stepImg)
        self.count += 1
