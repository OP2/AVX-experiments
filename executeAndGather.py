#! /usr/bin/python
from os import system
from sys import argv, stderr, exit
from changeBlockSize import changeBlock

def printParameterError():
  message = """Usage:
           executeAndGather.py -[o,v n]\n""" 
  stderr.write(message)
  exit()

"""
Given an option (original or vector) and in the case of vector, a blocksize
number, this function will fetch the specified version of the source,
compile it and store the execution times in a file.
"""
if __name__ == '__main__':

  try:
    # Parse user input and load correct file accordingly
    if argv[1] == '-v':
      system('rm rose_openmp_code.F90')
      system('cp generated_sources/rose_openmp_code_vector.F90 rose_openmp_code.F90')
      changeBlock(int(argv[2]))
    elif argv[1] == '-o':
      system('rm rose_openmp_code.F90')
      system('cp generated_sources/rose_openmp_code_original.F90 rose_openmp_code.F90')
    else:
      printParameterError()
  except:
      printParameterError()
    
  # Compile and execute
  system('make clean')
  system('make')
  system('./airfoil_openmp_128 > execution_results.txt')

  # Change results so to exclude anything we don't want
  results = open('execution_results.txt','r')
  lines = results.readlines()
  results.close()
  save_soln_lines = [line[4:] for line in lines if '### save_soln' in line]
  update_lines = [line[4:] for line in lines if '### update' in line]
  adt_calc_lines = [line[4:] for line in lines if '### adt_calc' in line]
  adt_gather_lines = [line[4:] for line in lines if '### adt_gather' in line]
  res_calc_lines = [line[4:] for line in lines if '### res_calc' in line]
  res_gather_lines = [line[4:] for line in lines if '### res_gather' in line]
  bres_calc_lines = [line[4:] for line in lines if '### bres_calc' in line]
  bres_gather_lines = [line[4:] for line in lines if '### bres_gather' in line]
  all_lines = [line[4:] for line in lines if '### ENTIRE' in line]
  not_counted_lines = [line[4:] for line in lines if '### not counted' in line]
  results = open('execution_results.txt','w')
  results.writelines(save_soln_lines)
  results.writelines(update_lines)
  results.writelines(adt_calc_lines)
  results.writelines(adt_gather_lines)
  results.writelines(res_calc_lines)
  results.writelines(res_gather_lines)
  results.writelines(bres_calc_lines)
  results.writelines(bres_gather_lines)
  results.writelines(all_lines)
  results.writelines(not_counted_lines)
  results.close()


