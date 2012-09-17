IC = icc
FC = ifort

SUPPORT = /home/alex/OpenMP/support

MKL = /opt/intel/mkl
MKLINCLUDE = $(MKL)/include
MKLLIB = $(MKL)/lib/intel64
OP2 = /home/alex/OP2-Common/op2

OPENMP = -openmp
PREPROCESSOR = -fpp
PART_SIZE_ENV = 128

EXEC = airfoil_openmp_$(PART_SIZE_ENV)

CPPLINK = -lstdc++
OPT = -vec-report3 -xAVX -O3 -DOP_PART_SIZE_1=$(PART_SIZE_ENV)

F_OP2_MOD = $(OP2)/fortran/mod/intel
F_OP2_LIB = $(OP2)/fortran/lib

FLINK = -L$(F_OP2_LIB) -L$(MKLLIB) -lmkl_rt -lpthread -lm 
FMODS = -module $(F_OP2_MOD)
FINK = -I$(MKLINCLUDE)

all: cfunctions airfoil_seq link

cfunctions: $(SUPPORT)/debug.c
	$(IC) $(OPT) $(DEBUG) $(OPENMP) -c $(SUPPORT)/debug.c

airfoil_seq: $(SUPPORT)/OP2Profiling.f90 debug_int.f90 input.f90 constvars.f90 rose_openmp_code.F90 airfoil_seqfun.f90 rose_airfoil.F90
	$(FC) $(OPT) $(DEBUG) $(OPENMP) $(FINK) $(FMODS) -c  $(SUPPORT)/OP2Profiling.f90 debug_int.f90 input.f90 constvars.f90 rose_openmp_code.F90 airfoil_seqfun.f90 rose_airfoil.F90

link: debug.o OP2Profiling.o debug_int.o input.o constvars.o rose_openmp_code.o airfoil_seqfun.o rose_airfoil.o
	$(FC) $(OPT) $(OPENMP) $(FINK) $(FLINK) OP2Profiling.o debug.o debug_int.o input.o constvars.o rose_openmp_code.o airfoil_seqfun.o rose_airfoil.o -o $(EXEC) -lop2_for_openmp_rt_support -lop2_for_openmp 

clean:
	rm -f *.o
	rm -f *.mod
	rm -f $(EXEC)
	rm -f *~
	rm -f *.s
