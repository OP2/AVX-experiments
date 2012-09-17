#!/usr/bin/python

from sys import argv

def printStats(name, data):
  total = 0
  for line in data:
    try:
      total += float(line)
    except:
      pass
  if not(len(data) == 0):
    print "%s: %f(%d iterations)" % (name, total, len(data))

if __name__ == "__main__":
  file = open(argv[1],'r')
  lines = file.readlines()
  save_soln_lines = [line[10:] for line in lines if 'save_soln' in line]
  update_lines = [line[10:] for line in lines if 'update' in line]
  adt_calc_lines = [line[10:] for line in lines if 'adt_calc' in line]
  adt_gather_lines = [line[11:] for line in lines if 'adt_gather' in line]
  res_calc_lines = [line[10:] for line in lines if 'res_calc' in line and not('b' in line)]
  res_gather_lines = [line[11:] for line in lines if 'res_gather' in line and not ('b' in line)]
  bres_calc_lines = [line[12:] for line in lines if 'bres_calc' in line]
  bres_gather_lines = [line[13:] for line in lines if 'bres_gather' in line]
  all_lines = [line[15:] for line in lines if 'ENTIRE' in line]
  not_counted = [line[12:] for line in lines if 'not counted' in line]
  printStats('save_soln', save_soln_lines)
  printStats('update', update_lines)
  printStats('adt_calc', adt_calc_lines)
  printStats('adt_gather', adt_gather_lines)
  printStats('res_calc', res_calc_lines)
  printStats('res_gather', res_gather_lines)
  printStats('bres_calc', bres_calc_lines)
  printStats('bres_gather', bres_gather_lines)
  printStats("All", all_lines)
  printStats("not counted", not_counted)
