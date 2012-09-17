#!/usr/bin/python

from sys import argv, exit

if __name__ == "__main__":
  if len(argv) != 3:
    exit('Usage: Please input two file names')

  try:
    file1 = open(argv[1], 'r')
    file2 = open(argv[2], 'r')
    
    # Lines contained within the file in array form
    lines1 = file1.readlines()
    lines2 = file2.readlines()
  
    file1.close()
    file2.close()

    if len(lines1) != len(lines2):
      exit('The number of lines in the two files is different...')

    # Number of lines that do not match
    mismatched = 0
    for i in range(len(lines1)):
      val1 = float(lines1[i])
      val2 = float(lines2[i])
      if val1 != val2:
        mismatched += 1
        print "Mismatch at line %d, %s:%f, %s:%f" % (i, argv[1], val1, argv[2], val2)

    print "Complete! There were %d/%d mismatches" % (mismatched, len(lines1))

  except:
    exit('You did something wrong, and you should feel bad.')
