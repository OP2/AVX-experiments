#! /usr/bin/python

""" 
Open the file, change line 7 containting the macro to desired vector pack size.
"""
def changeBlock(blockSize):
  # Open and read file
  source = open('rose_openmp_code.F90','r')
  lines = source.readlines()
  source.close()
  # Change line 7
  lines[7] = '#define BS ' + str(blockSize) + '\n'
  # Rewrite modified file
  source = open('rose_openmp_code.F90','w')
  source.writelines(lines)
  source.close()

if __name__ == '__main__':
  from sys import argv
  # Tell user that incorrect number of args given
  if not(len(argv) == 2):
    print 'I need to know the blocksize...'
  # Call changing method
  try:
    changeBlock(int(argv[1]))
  except:
    print 'Please input an integer as the blocksize'
