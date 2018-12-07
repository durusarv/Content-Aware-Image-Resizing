"""
@authors: durusarv and jdlips
"""

import cv2
import numpy as np
from seam_carver import SeamCarver

def main():
    inputFilename = "input/iceberg.jpg"
    outputWidth = 500
    outputHeight = 300
    outputFilename = "output/iceberg_" + str(outputWidth) + "x" + str(outputHeight) + ".jpg"


    sC = SeamCarver(inputFilename, outputFilename, outputWidth, outputHeight, True)
    sC.seamCarving();
    sC.outputImageToFile(outputFilename, sC.outputImg)
    print("Done")

if __name__ == "__main__":
    main()
