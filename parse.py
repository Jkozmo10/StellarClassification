from body import *

def parse_file(file):
  bodies = {}
  count = 0
  inFile = open(file, 'r')

  for line in inFile:
    curLine = line.split(',')
    curBody = Body(curLine[0], curLine[1], curLine[2], curLine[3], curLine[4], curLine[5], curLine[6], curLine[7]
      ,curLine[8], curLine[9], curLine[10], curLine[11], curLine[12], curLine[13], curLine[14], curLine[15], curLine[16], curLine[17])
    bodies[count] = curBody
    count += 1

  return bodies


