
import csv
class myFile:
    def __init__(self, fileName = None, fileMode = None):
        self.__fileName = fileName
        self.__fileMode = fileMode
        self.__fileList = []
        if self.__fileMode == "r" and self.__fileName != None:
            self.file = open(self.__fileName, self.__fileMode)
            read_file = csv.reader(self.file)
            for line in read_file:
                self.__fileList.append(line)
            self.__fileList[1:] = sorted(self.__fileList[1:])
            
        elif self.__fileMode == "w" and self.__fileName != "":
            self.file = open(self.__fileName, self.__fileMode)


    def getStatus(self):

        if self.__fileList != [] or self.__fileMode == "w":
            return True
        else:
            print("파일이 열리지 않았음 status = false")
            return False


    def getBody(self):

        if self.getStatus() == True:
            return self.__fileList[1:]
        else:
            print("파일이 열리지 않아 body 출력이 불가합니다.")
            return False

    def setContentHead(self,fileHeader = None):
        if self.getStatus() == True and fileHeader != None:
            self.__fileHeader = fileHeader
            return True
        else:
            print("fileHeader 가 주어지지 않았거나, 파일이 열리지 않았습니다.")
            return False


    def setContentBody(self, fileContent = None):
        if self.getStatus() == True and fileContent != None:
            self.__fileContent = fileContent
            return True
        else:
            print("fileContent 가 주어지지 않았거나, 파일이 열리지 않았습니다.")
            return False

    def writeFile(self):
        if self.getStatus() == True:
            myWriter = csv.writer(self.file)
            myWriter.writerow(self.__fileHeader)
            for i in range(len(self.__fileContent)):
                myWriter.writerow(self.__fileContent[i])
            return True

        else:
            print("파일이 열리지 않았습니다.")
            return False

    def closeFile(self):
        if self.getStatus() == True:
            self.file.close()
            return True

        else:
            print("파일이 열리지 않았습니다.")
            return False

        
def mergeList(li1, li2):
    result = []
    temp = []
    
    for i in range(len(li1)):
        for j in range(len(li2)):
            if li1[i][0] == li2[j][0]:
                temp = []
                temp.append(li1[i][0])
                temp.append(li1[i][1])                
                temp.append(li2[j][1])
                temp.append(li2[j][2])
                temp.append(li2[j][3])                
                sum = 0
                for k in range(1,4):
                    sum = sum + int(li2[j][k])
            
                temp.append(str(int(sum/3)))

                result.append(temp)

    result = sorted(result)

    return result