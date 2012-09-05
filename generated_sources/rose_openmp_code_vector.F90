MODULE GENERATED_MODULE
USE OP2_FORTRAN_DECLARATIONS
USE OP2_FORTRAN_RT_SUPPORT
USE OP2_CONSTANTS
#ifdef _OPENMP 
USE OMP_LIB
#endif 
#define BS 4
REAL(kind=8) :: alpha_OP2_CONSTANT
REAL(kind=8) :: cfl_OP2_CONSTANT
REAL(kind=8) :: eps_OP2_CONSTANT
REAL(kind=8) :: gam_OP2_CONSTANT
REAL(kind=8) :: gm1_OP2_CONSTANT
REAL(kind=8) :: mach_OP2_CONSTANT
REAL(kind=8), DIMENSION(4) :: qinf_OP2_CONSTANT
LOGICAL :: firstTime_adt_calc = .TRUE.
TYPE ( c_ptr )  :: planRet_adt_calc
TYPE ( op_plan ) , POINTER :: actualPlan_adt_calc
TYPE ( c_ptr ) , POINTER, DIMENSION(:) :: ind_maps_adt_calc
TYPE ( c_ptr ) , POINTER, DIMENSION(:) :: mappingArray_adt_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_maps1_adt_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray1_adt_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray2_adt_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray3_adt_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray4_adt_calc
INTEGER(kind=4) :: mappingArray1Size_adt_calc
INTEGER(kind=4) :: mappingArray2Size_adt_calc
INTEGER(kind=4) :: mappingArray3Size_adt_calc
INTEGER(kind=4) :: mappingArray4Size_adt_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: blkmap_adt_calc
INTEGER(kind=4) :: blkmapSize_adt_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_offs_adt_calc
INTEGER(kind=4) :: ind_offsSize_adt_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_sizes_adt_calc
INTEGER(kind=4) :: ind_sizesSize_adt_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: nelems_adt_calc
INTEGER(kind=4) :: nelemsSize_adt_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: nthrcol_adt_calc
INTEGER(kind=4) :: nthrcolSize_adt_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: offset_adt_calc
INTEGER(kind=4) :: offsetSize_adt_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: thrcol_adt_calc
INTEGER(kind=4) :: thrcolSize_adt_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ncolblk_adt_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: pnindirect_adt_calc
LOGICAL :: firstTime_bres_calc = .TRUE.
TYPE ( c_ptr )  :: planRet_bres_calc
TYPE ( op_plan ) , POINTER :: actualPlan_bres_calc
TYPE ( c_ptr ) , POINTER, DIMENSION(:) :: ind_maps_bres_calc
TYPE ( c_ptr ) , POINTER, DIMENSION(:) :: mappingArray_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_maps1_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_maps3_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_maps4_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_maps5_bres_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray1_bres_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray2_bres_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray3_bres_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray4_bres_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray5_bres_calc
INTEGER(kind=4) :: mappingArray1Size_bres_calc
INTEGER(kind=4) :: mappingArray2Size_bres_calc
INTEGER(kind=4) :: mappingArray3Size_bres_calc
INTEGER(kind=4) :: mappingArray4Size_bres_calc
INTEGER(kind=4) :: mappingArray5Size_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: blkmap_bres_calc
INTEGER(kind=4) :: blkmapSize_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_offs_bres_calc
INTEGER(kind=4) :: ind_offsSize_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_sizes_bres_calc
INTEGER(kind=4) :: ind_sizesSize_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: nelems_bres_calc
INTEGER(kind=4) :: nelemsSize_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: nthrcol_bres_calc
INTEGER(kind=4) :: nthrcolSize_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: offset_bres_calc
INTEGER(kind=4) :: offsetSize_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: thrcol_bres_calc
INTEGER(kind=4) :: thrcolSize_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ncolblk_bres_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: pnindirect_bres_calc
LOGICAL :: firstTime_res_calc = .TRUE.
TYPE ( c_ptr )  :: planRet_res_calc
TYPE ( op_plan ) , POINTER :: actualPlan_res_calc
TYPE ( c_ptr ) , POINTER, DIMENSION(:) :: ind_maps_res_calc
TYPE ( c_ptr ) , POINTER, DIMENSION(:) :: mappingArray_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_maps1_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_maps3_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_maps5_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_maps7_res_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray1_res_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray2_res_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray3_res_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray4_res_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray5_res_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray6_res_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray7_res_calc
INTEGER(kind=2), POINTER, DIMENSION(:) :: mappingArray8_res_calc
INTEGER(kind=4) :: mappingArray1Size_res_calc
INTEGER(kind=4) :: mappingArray2Size_res_calc
INTEGER(kind=4) :: mappingArray3Size_res_calc
INTEGER(kind=4) :: mappingArray4Size_res_calc
INTEGER(kind=4) :: mappingArray5Size_res_calc
INTEGER(kind=4) :: mappingArray6Size_res_calc
INTEGER(kind=4) :: mappingArray7Size_res_calc
INTEGER(kind=4) :: mappingArray8Size_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: blkmap_res_calc
INTEGER(kind=4) :: blkmapSize_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_offs_res_calc
INTEGER(kind=4) :: ind_offsSize_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ind_sizes_res_calc
INTEGER(kind=4) :: ind_sizesSize_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: nelems_res_calc
INTEGER(kind=4) :: nelemsSize_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: nthrcol_res_calc
INTEGER(kind=4) :: nthrcolSize_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: offset_res_calc
INTEGER(kind=4) :: offsetSize_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: thrcol_res_calc
INTEGER(kind=4) :: thrcolSize_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: ncolblk_res_calc
INTEGER(kind=4), POINTER, DIMENSION(:) :: pnindirect_res_calc
CONTAINS
SUBROUTINE initOP2Constants(alpha,cfl,eps,gam,gm1,mach,qinf)
IMPLICIT NONE
REAL(kind=8) :: alpha
REAL(kind=8) :: cfl
REAL(kind=8) :: eps
REAL(kind=8) :: gam
REAL(kind=8) :: gm1
REAL(kind=8) :: mach
REAL(kind=8), DIMENSION(4) :: qinf
INTEGER(kind=4) :: i1
alpha_OP2_CONSTANT = alpha
cfl_OP2_CONSTANT = cfl
eps_OP2_CONSTANT = eps
gam_OP2_CONSTANT = gam
gm1_OP2_CONSTANT = gm1
mach_OP2_CONSTANT = mach
qinf_OP2_CONSTANT = qinf
END SUBROUTINE 

SUBROUTINE adt_calc_modified(x1,x2,x3,x4,q,adt)
IMPLICIT NONE
REAL(kind=8), DIMENSION(*) :: x1
REAL(kind=8), DIMENSION(*) :: x2
REAL(kind=8), DIMENSION(*) :: x3
REAL(kind=8), DIMENSION(*) :: x4
REAL(kind=8), DIMENSION(*) :: q
REAL(kind=8), DIMENSION(1) :: adt
reAL(kind=8) :: dx,dy,ri,u,v,c
ri = 1.0 / q(1)
u = ri * q(2)
v = ri * q(3)
c = sqrt(gam_OP2_CONSTANT * gm1_OP2_CONSTANT * (ri * q(4) - 0.5 * (u * u + v * v)))
dx = x2(1) - x1(1)
dy = x2(2) - x1(2)
adt(1) = abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)
dx = x3(1) - x2(1)
dy = x3(2) - x2(2)
adt(1) = adt(1) + abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)
dx = x4(1) - x3(1)
dy = x4(2) - x3(2)
adt(1) = adt(1) + abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)
dx = x1(1) - x4(1)
dy = x1(2) - x4(2)
adt(1) = adt(1) + abs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy)
adt(1) = adt(1) / cfl_OP2_CONSTANT
END SUBROUTINE 

SUBROUTINE adt_calc_kernel(opDat1,opDat5,opDat6,ind_maps1,mappingArray1,mappingArray2,mappingArray3,mappingArray4,ind_sizes,ind_offs,blkmap,offset,nelems,nthrcol,thrcol,blockOffset,blockID)
IMPLICIT NONE
REAL(kind=8), DIMENSION(0:*) :: opDat1
REAL(kind=8), DIMENSION(0:*) :: opDat5
REAL(kind=8), DIMENSION(0:*) :: opDat6
INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps1
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray1
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray2
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray3
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray4
INTEGER(kind=4), DIMENSION(0:*) :: ind_sizes
INTEGER(kind=4), DIMENSION(0:*) :: ind_offs
INTEGER(kind=4), DIMENSION(0:*) :: blkmap
INTEGER(kind=4), DIMENSION(0:*) :: offset
INTEGER(kind=4), DIMENSION(0:*) :: nelems
INTEGER(kind=4), DIMENSION(0:*) :: nthrcol
INTEGER(kind=4), DIMENSION(0:*) :: thrcol
INTEGER(kind=4) :: blockOffset
INTEGER(kind=4) :: blockID
INTEGER(kind=4) :: threadBlockOffset
INTEGER(kind=4) :: threadBlockID
INTEGER(kind=4) :: numberOfActiveThreads
INTEGER(kind=4) :: i1
INTEGER(kind=4) :: i2
INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat1IndirectionMap
REAL(kind=8), POINTER, DIMENSION(:) :: opDat1SharedIndirection
REAL(kind=8), DIMENSION(0:8000 - 1), TARGET :: sharedFloat8
INTEGER(kind=4) :: opDat1nBytes
INTEGER(kind=4) :: opDat1RoundUp
INTEGER(kind=4) :: opDat1SharedIndirectionSize
integer(8) :: time1, time2, count_rate, count_max
integer(8) :: time1g, time2g, count_rateg, count_maxg
real(kind=8), allocatable, dimension(:) :: arg1, arg2, arg3, arg4, arg5
threadBlockID = blkmap(blockID + blockOffset)
numberOfActiveThreads = nelems(threadBlockID)
call system_clock(time1, count_rate, count_max)
allocate (arg1(numberOfActiveThreads*2))
allocate (arg2(numberOfActiveThreads*2))
allocate (arg3(numberOfActiveThreads*2))
allocate (arg4(numberOfActiveThreads*2))
allocate (arg5(numberOfActiveThreads*4))
call system_clock(time2, count_rate, count_max)
print *, "### not counted"
threadBlockOffset = offset(threadBlockID)
opDat1SharedIndirectionSize = ind_sizes(0 + threadBlockID * 1)
opDat1IndirectionMap => ind_maps1(ind_offs(0 + threadBlockID * 1):)
opDat1nBytes = 0
opDat1SharedIndirection => sharedFloat8(opDat1nBytes:)
DO i1 = 0, opDat1SharedIndirectionSize - 1, 1
DO i2 = 0, 2 - 1, 1
opDat1SharedIndirection(i2 + i1 * 2 + 1) = opDat1(i2 + opDat1IndirectionMap(i1 + 1) * 2)
END DO
END DO
call system_clock(time1, count_rate, count_max)
call mkl_domatcopy('r', 't', numberOfActiveThreads, 4, 1.d0, opDat5(4*threadBlockOffset:4*(numberOfActiveThreads-1+threadBlockOffset) + 3), 4, arg5, numberOfActiveThreads)
call system_clock(time2, count_rate, count_max)
print *, "### not counted"
call system_clock(time1g, count_rateg, count_maxg)
DO i1 = 0, numberOfActiveThreads - 1, 1
arg1(i1 + 1) = opDat1SharedIndirection(1 + mappingArray1(i1 + threadBlockOffset) * 2)
arg1(i1 + numberOfActiveThreads + 1) = opDat1SharedIndirection(1 + mappingArray1(i1 + threadBlockOffset) * 2 + 1)
arg2(i1 + 1) = opDat1SharedIndirection(1 + mappingArray2(i1 + threadBlockOffset) * 2)
arg2(i1 + numberOfActiveThreads + 1) = opDat1SharedIndirection(1 + mappingArray2(i1 + threadBlockOffset) * 2 + 1)
arg3(i1 + 1) = opDat1SharedIndirection(1 + mappingArray3(i1 + threadBlockOffset) * 2)
arg3(i1 + numberOfActiveThreads + 1) = opDat1SharedIndirection(1 + mappingArray3(i1 + threadBlockOffset) * 2 + 1)
arg4(i1 + 1) = opDat1SharedIndirection(1 + mappingArray4(i1 + threadBlockOffset) * 2)
arg4(i1 + numberOfActiveThreads + 1) = opDat1SharedIndirection(1 + mappingArray4(i1 + threadBlockOffset) * 2 + 1)
END DO
call system_clock(time1, count_rate, count_max)
call adt_calc_kernel_caller(arg1,arg2,arg3,arg4,arg5,opDat6(threadBlockOffset:threadBlockOffset+numberOfActiveThreads-1), numberOfActiveThreads)
call system_clock(time2, count_rate, count_max)
print *, "### adt_calc", time2 - time1
DO i1 = 0, numberOfActiveThreads - 1, 1
opDat1SharedIndirection(1 + mappingArray1(i1 + threadBlockOffset) * 2) = arg1(i1 + 1) 
opDat1SharedIndirection(1 + mappingArray1(i1 + threadBlockOffset) * 2 + 1) = arg1(i1 + numberOfActiveThreads + 1)
opDat1SharedIndirection(1 + mappingArray2(i1 + threadBlockOffset) * 2) = arg2(i1 + 1)
opDat1SharedIndirection(1 + mappingArray2(i1 + threadBlockOffset) * 2 + 1) = arg2(i1 + numberOfActiveThreads + 1)
opDat1SharedIndirection(1 + mappingArray3(i1 + threadBlockOffset) * 2) = arg3(i1 + 1) 
opDat1SharedIndirection(1 + mappingArray3(i1 + threadBlockOffset) * 2 + 1) = arg3(i1 + numberOfActiveThreads + 1)
opDat1SharedIndirection(1 + mappingArray4(i1 + threadBlockOffset) * 2) = arg4(i1 + 1)
opDat1SharedIndirection(1 + mappingArray4(i1 + threadBlockOffset) * 2 + 1) = arg4(i1 + numberOfActiveThreads + 1) 
END DO
call system_clock(time2g, count_rateg, count_maxg)
print *, "### adt_gather", time2g - time1g
call system_clock(time1, count_rate, count_max)
call mkl_domatcopy('r', 't', 4, numberOfActiveThreads, 1.d0, arg5, numberOfActiveThreads, opDat5(4*threadBlockOffset:4*(numberOfActiveThreads-1+threadBlockOffset) + 3), 4)
call system_clock(time2, count_rate, count_max)
print *, "### not counted", time2 - time1
deallocate (arg1)
deallocate (arg2)
deallocate (arg3)
deallocate (arg4)
deallocate (arg5)
END SUBROUTINE 

subroutine adt_calc_kernel_caller(x1, x2, x3, x4, q, adt, iterations)

  implicit none
  real(kind=8), dimension(BS, iterations/BS, 2) :: x1, x2, x3, x4
  real(kind=8), dimension(BS, iterations/BS, 4) :: q
  real(kind=8), dimension(BS, iterations/BS) :: adt
  integer :: iterations
  integer :: i
  REAL(kind=8), dimension(BS) :: dx,dy,ri,u,v,c
  do i = 1, iterations/BS
    ri(:) = 1.0 / q(:, i, 1)
    u(:) = ri(:) * q(:, i, 2)
    v(:) = ri(:) * q(:, i, 3)
    c(:) = sqrt(gam_OP2_CONSTANT * gm1_OP2_CONSTANT * (ri(:) * q(:, i, 4) - 0.5 * (u(:) * u(:) + v(:) * v(:))))
    dx(:) = x2(:, i, 1) - x1(:, i, 1)
    dy(:) = x2(:, i, 2) - x1(:, i, 2)
    adt(:,i) = abs(u(:) * dy(:) - v(:) * dx(:)) + c(:) * sqrt(dx(:) * dx(:) + dy(:) * dy(:))
    dx(:) = x3(:, i, 1) - x2(:, i, 1)
    dy(:) = x3(:, i, 2) - x2(:, i, 2)
    adt(:,i) = adt(:,i) + abs(u(:) * dy(:) - v(:) * dx(:)) + c(:) * sqrt(dx(:) * dx(:) + dy(:) * dy(:))
    dx(:) = x4(:, i, 1) - x3(:, i, 1)
    dy(:) = x4(:, i, 2) - x3(:, i, 2)
    adt(:,i) = adt(:,i) + abs(u(:) * dy(:) - v(:) * dx(:)) + c(:) * sqrt(dx(:) * dx(:) + dy(:) * dy(:))
    dx(:) = x1(:, i, 1) - x4(:, i, 1)
    dy(:) = x1(:, i, 2) - x4(:, i, 2)
    adt(:,i) = adt(:,i) + abs(u(:) * dy(:) - v(:) * dx(:)) + c(:) * sqrt(dx(:) * dx(:) + dy(:) * dy(:))
    adt(:,i) = adt(:,i) / cfl_OP2_CONSTANT
  end do
end subroutine

SUBROUTINE adt_calc_host(userSubroutine,set,opDat1,opIndirection1,opMap1,opAccess1,opDat2,opIndirection2,opMap2,opAccess2,opDat3,opIndirection3,opMap3,opAccess3,opDat4,opIndirection4,opMap4,opAccess4,opDat5,opIndirection5,opMap5,opAccess5,opDat6,opIndirection6,opMap6,opAccess6)
IMPLICIT NONE
character(len=9), INTENT(IN) :: userSubroutine
TYPE ( op_set ) , INTENT(IN) :: set
TYPE ( op_dat ) , INTENT(IN) :: opDat1
INTEGER(kind=4), INTENT(IN) :: opIndirection1
TYPE ( op_map ) , INTENT(IN) :: opMap1
INTEGER(kind=4), INTENT(IN) :: opAccess1
TYPE ( op_dat ) , INTENT(IN) :: opDat2
INTEGER(kind=4), INTENT(IN) :: opIndirection2
TYPE ( op_map ) , INTENT(IN) :: opMap2
INTEGER(kind=4), INTENT(IN) :: opAccess2
TYPE ( op_dat ) , INTENT(IN) :: opDat3
INTEGER(kind=4), INTENT(IN) :: opIndirection3
TYPE ( op_map ) , INTENT(IN) :: opMap3
INTEGER(kind=4), INTENT(IN) :: opAccess3
TYPE ( op_dat ) , INTENT(IN) :: opDat4
INTEGER(kind=4), INTENT(IN) :: opIndirection4
TYPE ( op_map ) , INTENT(IN) :: opMap4
INTEGER(kind=4), INTENT(IN) :: opAccess4
TYPE ( op_dat ) , INTENT(IN) :: opDat5
INTEGER(kind=4), INTENT(IN) :: opIndirection5
TYPE ( op_map ) , INTENT(IN) :: opMap5
INTEGER(kind=4), INTENT(IN) :: opAccess5
TYPE ( op_dat ) , INTENT(IN) :: opDat6
INTEGER(kind=4), INTENT(IN) :: opIndirection6
TYPE ( op_map ) , INTENT(IN) :: opMap6
INTEGER(kind=4), INTENT(IN) :: opAccess6
TYPE ( op_set_core ) , POINTER :: opSetCore
TYPE ( op_dat_core ) , POINTER :: opDat1Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat1Local
INTEGER(kind=4) :: opDat1Cardinality
TYPE ( op_set_core ) , POINTER :: opSet1Core
TYPE ( op_map_core ) , POINTER :: opMap1Core
TYPE ( op_dat_core ) , POINTER :: opDat2Core
TYPE ( op_map_core ) , POINTER :: opMap2Core
TYPE ( op_dat_core ) , POINTER :: opDat3Core
TYPE ( op_map_core ) , POINTER :: opMap3Core
TYPE ( op_dat_core ) , POINTER :: opDat4Core
TYPE ( op_map_core ) , POINTER :: opMap4Core
TYPE ( op_dat_core ) , POINTER :: opDat5Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat5Local
INTEGER(kind=4) :: opDat5Cardinality
TYPE ( op_set_core ) , POINTER :: opSet5Core
TYPE ( op_map_core ) , POINTER :: opMap5Core
TYPE ( op_dat_core ) , POINTER :: opDat6Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat6Local
INTEGER(kind=4) :: opDat6Cardinality
TYPE ( op_set_core ) , POINTER :: opSet6Core
TYPE ( op_map_core ) , POINTER :: opMap6Core
INTEGER(kind=4) :: i1
INTEGER(kind=4) :: numberOfThreads
INTEGER(kind=4), DIMENSION(1:6) :: opDatArray
INTEGER(kind=4), DIMENSION(1:6) :: mappingIndicesArray
INTEGER(kind=4), DIMENSION(1:6) :: mappingArray
INTEGER(kind=4), DIMENSION(1:6) :: accessDescriptorArray
INTEGER(kind=4), DIMENSION(1:6) :: indirectionDescriptorArray
INTEGER(kind=4), DIMENSION(1:6) :: opDatTypesArray
INTEGER(kind=4) :: numberOfOpDats
INTEGER(kind=4) :: numberOfIndirectOpDats
INTEGER(kind=4) :: blockOffset
INTEGER(kind=4) :: nblocks
INTEGER(kind=4) :: i2
IF (set%setPtr%size .EQ. 0) THEN
RETURN
END IF
#ifdef _OPENMP 
numberOfThreads = omp_get_max_threads()
#else 
numberOfThreads = 1
#endif 
opSetCore => set%setPtr
opMap1Core => opMap1%mapPtr
opMap2Core => opMap2%mapPtr
opMap3Core => opMap3%mapPtr
opMap4Core => opMap4%mapPtr
opMap5Core => opMap5%mapPtr
opMap6Core => opMap6%mapPtr
opDat1Core => opDat1%dataPtr
opDat2Core => opDat2%dataPtr
opDat3Core => opDat3%dataPtr
opDat4Core => opDat4%dataPtr
opDat5Core => opDat5%dataPtr
opDat6Core => opDat6%dataPtr
CALL c_f_pointer(opDat1Core%set,opSet1Core)
CALL c_f_pointer(opDat5Core%set,opSet5Core)
CALL c_f_pointer(opDat6Core%set,opSet6Core)
opDat1Cardinality = opDat1Core%dim * opSet1Core%size
opDat5Cardinality = opDat5Core%dim * opSet5Core%size
opDat6Cardinality = opDat6Core%dim * opSet6Core%size
CALL c_f_pointer(opDat1Core%dat,opDat1Local,(/opDat1Cardinality/))
CALL c_f_pointer(opDat5Core%dat,opDat5Local,(/opDat5Cardinality/))
CALL c_f_pointer(opDat6Core%dat,opDat6Local,(/opDat6Cardinality/))
IF (firstTime_adt_calc .EQV. .TRUE.) THEN
firstTime_adt_calc = .FALSE.
opDatArray(1) = opDat1Core%index
opDatArray(2) = opDat2Core%index
opDatArray(3) = opDat3Core%index
opDatArray(4) = opDat4Core%index
opDatArray(5) = opDat5Core%index
opDatArray(6) = opDat6Core%index
mappingIndicesArray(1) = opIndirection1
mappingIndicesArray(2) = opIndirection2
mappingIndicesArray(3) = opIndirection3
mappingIndicesArray(4) = opIndirection4
mappingIndicesArray(5) = opIndirection5
mappingIndicesArray(6) = opIndirection6
DO i1 = 1, 6, 1
IF (mappingIndicesArray(i1) .NE. -1) THEN
mappingIndicesArray(i1) = mappingIndicesArray(i1) - 1
END IF
END DO
mappingArray(1) = opMap1Core%index
mappingArray(2) = opMap2Core%index
mappingArray(3) = opMap3Core%index
mappingArray(4) = opMap4Core%index
mappingArray(5) = opMap5Core%index
mappingArray(6) = opMap6Core%index
accessDescriptorArray(1) = opAccess1
accessDescriptorArray(2) = opAccess2
accessDescriptorArray(3) = opAccess3
accessDescriptorArray(4) = opAccess4
accessDescriptorArray(5) = opAccess5
accessDescriptorArray(6) = opAccess6
indirectionDescriptorArray(1) = 0
indirectionDescriptorArray(2) = 0
indirectionDescriptorArray(3) = 0
indirectionDescriptorArray(4) = 0
indirectionDescriptorArray(5) = -1
indirectionDescriptorArray(6) = -1
numberOfOpDats = 6
numberOfIndirectOpDats = 1
IF (opMap1Core%dim .EQ. -1) THEN
opDatTypesArray(1) = F_OP_ARG_GBL
ELSE
opDatTypesArray(1) = F_OP_ARG_DAT
END IF
IF (opMap2Core%dim .EQ. -1) THEN
opDatTypesArray(2) = F_OP_ARG_GBL
ELSE
opDatTypesArray(2) = F_OP_ARG_DAT
END IF
IF (opMap3Core%dim .EQ. -1) THEN
opDatTypesArray(3) = F_OP_ARG_GBL
ELSE
opDatTypesArray(3) = F_OP_ARG_DAT
END IF
IF (opMap4Core%dim .EQ. -1) THEN
opDatTypesArray(4) = F_OP_ARG_GBL
ELSE
opDatTypesArray(4) = F_OP_ARG_DAT
END IF
IF (opMap5Core%dim .EQ. -1) THEN
opDatTypesArray(5) = F_OP_ARG_GBL
ELSE
opDatTypesArray(5) = F_OP_ARG_DAT
END IF
IF (opMap6Core%dim .EQ. -1) THEN
opDatTypesArray(6) = F_OP_ARG_GBL
ELSE
opDatTypesArray(6) = F_OP_ARG_DAT
END IF
planRet_adt_calc = cplan_OpenMP(userSubroutine,opSetCore%index,numberOfOpDats,opDatArray,mappingIndicesArray,mappingArray,accessDescriptorArray,numberOfIndirectOpDats,indirectionDescriptorArray,opDatTypesArray,0)
CALL c_f_pointer(planRet_adt_calc,actualPlan_adt_calc)
CALL c_f_pointer(actualPlan_adt_calc%nindirect,pnindirect_adt_calc,(/numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_adt_calc%ind_maps,ind_maps_adt_calc,(/numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_adt_calc%maps,mappingArray_adt_calc,(/numberOfOpDats/))
CALL c_f_pointer(actualPlan_adt_calc%ncolblk,ncolblk_adt_calc,(/opSetCore%size/))
CALL c_f_pointer(actualPlan_adt_calc%ind_sizes,ind_sizes_adt_calc,(/actualPlan_adt_calc%nblocks * numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_adt_calc%ind_offs,ind_offs_adt_calc,(/actualPlan_adt_calc%nblocks * numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_adt_calc%blkmap,blkmap_adt_calc,(/actualPlan_adt_calc%nblocks/))
CALL c_f_pointer(actualPlan_adt_calc%offset,offset_adt_calc,(/actualPlan_adt_calc%nblocks/))
CALL c_f_pointer(actualPlan_adt_calc%nelems,nelems_adt_calc,(/actualPlan_adt_calc%nblocks/))
CALL c_f_pointer(actualPlan_adt_calc%nthrcol,nthrcol_adt_calc,(/actualPlan_adt_calc%nblocks/))
CALL c_f_pointer(actualPlan_adt_calc%thrcol,thrcol_adt_calc,(/opSetCore%size/))
CALL c_f_pointer(ind_maps_adt_calc(1),ind_maps1_adt_calc,(/pnindirect_adt_calc(1)/))
IF (indirectionDescriptorArray(1) >= 0) THEN
CALL c_f_pointer(mappingArray_adt_calc(1),mappingArray1_adt_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(2) >= 0) THEN
CALL c_f_pointer(mappingArray_adt_calc(2),mappingArray2_adt_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(3) >= 0) THEN
CALL c_f_pointer(mappingArray_adt_calc(3),mappingArray3_adt_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(4) >= 0) THEN
CALL c_f_pointer(mappingArray_adt_calc(4),mappingArray4_adt_calc,(/opSetCore%size/))
END IF
END IF
blockOffset = 0
DO i1 = 0, actualPlan_adt_calc%ncolors - 1, 1
nblocks = ncolblk_adt_calc(i1 + 1)
!$OMP PARALLEL DO 
DO i2 = 0, nblocks - 1, 1
CALL adt_calc_kernel(opDat1Local,opDat5Local,opDat6Local,ind_maps1_adt_calc,mappingArray1_adt_calc,mappingArray2_adt_calc,mappingArray3_adt_calc,mappingArray4_adt_calc,ind_sizes_adt_calc,ind_offs_adt_calc,blkmap_adt_calc,offset_adt_calc,nelems_adt_calc,nthrcol_adt_calc,thrcol_adt_calc,blockOffset,i2)
END DO
!$OMP END PARALLEL DO
blockOffset = blockOffset + nblocks
END DO
END SUBROUTINE 

SUBROUTINE bres_calc_modified(x1,x2,q1,adt1,res1,bound)
IMPLICIT NONE
REAL(kind=8), DIMENSION(*) :: x1
REAL(kind=8), DIMENSION(*) :: x2
REAL(kind=8), DIMENSION(*) :: q1
REAL(kind=8), DIMENSION(*) :: adt1
REAL(kind=8), DIMENSION(*) :: res1
INTEGER(kind=4), DIMENSION(*) :: bound
REAL(kind=8) :: dx,dy,mu,ri,p1,vol1,p2,vol2,f
dx = x1(1) - x2(1)
dy = x1(2) - x2(2)
ri = 1.0 / q1(1)
p1 = gm1_OP2_CONSTANT * (q1(4) - 0.5 * ri * (q1(2) * q1(2) + q1(3) * q1(3)))
IF (bound(1) .EQ. 1) THEN
res1(2) = res1(2) + p1 * dy
res1(3) = res1(3) + -(p1 * dx)
ELSE
vol1 = ri * (q1(2) * dy - q1(3) * dx)
ri = 1.0 / qinf_OP2_CONSTANT(1)
p2 = gm1_OP2_CONSTANT * (qinf_OP2_CONSTANT(4) - 0.5 * ri * (qinf_OP2_CONSTANT(2) * qinf_OP2_CONSTANT(2) + qinf_OP2_CONSTANT(3) * qinf_OP2_CONSTANT(3)))
vol2 = ri * (qinf_OP2_CONSTANT(2) * dy - qinf_OP2_CONSTANT(3) * dx)
mu = adt1(1) * eps_OP2_CONSTANT
f = 0.5 * (vol1 * q1(1) + vol2 * qinf_OP2_CONSTANT(1)) + mu * (q1(1) - qinf_OP2_CONSTANT(1))
res1(1) = res1(1) + f
f = 0.5 * (vol1 * q1(2) + p1 * dy + vol2 * qinf_OP2_CONSTANT(2) + p2 * dy) + mu * (q1(2) - qinf_OP2_CONSTANT(2))
res1(2) = res1(2) + f
f = 0.5 * (vol1 * q1(3) - p1 * dx + vol2 * qinf_OP2_CONSTANT(3) - p2 * dx) + mu * (q1(3) - qinf_OP2_CONSTANT(3))
res1(3) = res1(3) + f
f = 0.5 * (vol1 * (q1(4) + p1) + vol2 * (qinf_OP2_CONSTANT(4) + p2)) + mu * (q1(4) - qinf_OP2_CONSTANT(4))
res1(4) = res1(4) + f
END IF
END SUBROUTINE 

SUBROUTINE bres_calc_kernel_caller(x1,x2,q1,adt1,res1,bound,iterations)
IMPLICIT NONE
integer :: iterations
REAL(kind=8), DIMENSION(BS, iterations/BS, 2) :: x1, x2
REAL(kind=8), DIMENSION(BS, iterations/BS, 4) :: q1
REAL(kind=8), DIMENSION(BS, iterations/BS) :: adt1
REAL(kind=8), DIMENSION(BS, iterations/BS, 4) :: res1
INTEGER(kind=4), DIMENSION(BS, iterations/BS) :: bound
REAL(kind=8), DIMENSION(BS) :: dx,dy,mu,ri,p1,vol1,p2,vol2,f
integer :: i,j
do i = 1, iterations/BS
  do j = 1, BS
    dx(j) = x1(j, i, 1) - x2(j, i, 1)
    dy(j) = x1(j, i, 2) - x2(j, i, 2)
    ri(j) = 1.0 / q1(j, i, 1)
    p1(j) = gm1_OP2_CONSTANT * (q1(j, i, 4) - 0.5 * ri(j) * (q1(j, i, 2) * q1(j, i, 2) + q1(j, i, 3) * q1(j, i, 3)))
    IF (bound(j, i) .EQ. 1) THEN
      res1(j, i, 2) = res1(j, i, 2) + p1(j) * dy(j)
      res1(j, i, 3) = res1(j, i, 3) + -(p1(j) * dx(j))
    ELSE
      vol1(j) = ri(j) * (q1(j, i, 2) * dy(j) - q1(j, i, 3) * dx(j))
      ri(j) = 1.0 / qinf_OP2_CONSTANT(1)
      p2(j) = gm1_OP2_CONSTANT * (qinf_OP2_CONSTANT(4) - 0.5 * ri(j) * (qinf_OP2_CONSTANT(2) * qinf_OP2_CONSTANT(2) + qinf_OP2_CONSTANT(3) * qinf_OP2_CONSTANT(3)))
      vol2(j) = ri(j) * (qinf_OP2_CONSTANT(2) * dy(j) - qinf_OP2_CONSTANT(3) * dx(j))
      mu(j) = adt1(j, i) * eps_OP2_CONSTANT
      f(j) = 0.5 * (vol1(j) * q1(j, i, 1) + vol2(j) * qinf_OP2_CONSTANT(1)) + mu(j) * (q1(j, i, 1) - qinf_OP2_CONSTANT(1))
      res1(j, i, 1) = res1(j, i, 1) + f(j)
      f(j) = 0.5 * (vol1(j) * q1(j, i, 2) + p1(j) * dy(j) + vol2(j) * qinf_OP2_CONSTANT(2) + p2(j) * dy(j)) + mu(j) * (q1(j, i, 2) - qinf_OP2_CONSTANT(2))
      res1(j, i, 2) = res1(j, i, 2) + f(j)
      f(j) = 0.5 * (vol1(j) * q1(j, i, 3) - p1(j) * dx(j) + vol2(j) * qinf_OP2_CONSTANT(3) - p2(j) * dx(j)) + mu(j) * (q1(j, i, 3) - qinf_OP2_CONSTANT(3))
      res1(j, i, 3) = res1(j, i, 3) + f(j)
      f(j) = 0.5 * (vol1(j) * (q1(j, i, 4) + p1(j)) + vol2(j) * (qinf_OP2_CONSTANT(4) + p2(j))) + mu(j) * (q1(j, i, 4) - qinf_OP2_CONSTANT(4))
      res1(j, i, 4) = res1(j, i, 4) + f(j)
    END IF
  end do
end do
END SUBROUTINE 

!SUBROUTINE bres_calc_kernel(opDat1,opDat3,opDat4,opDat5,opDat6,ind_maps1,ind_maps3,ind_maps4,ind_maps5,mappingArray1,mappingArray2,mappingArray3,mappingArray4,mappingArray5,ind_sizes,ind_offs,blkmap,offset,nelems,nthrcol,thrcol,blockOffset,blockID)
!IMPLICIT NONE
!REAL(kind=8), DIMENSION(0:*) :: opDat1
!REAL(kind=8), DIMENSION(0:*) :: opDat3
!REAL(kind=8), DIMENSION(0:*) :: opDat4
!REAL(kind=8), DIMENSION(0:*) :: opDat5
!INTEGER(kind=4), DIMENSION(0:*) :: opDat6
!INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps1
!INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps3
!INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps4
!INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps5
!INTEGER(kind=2), DIMENSION(0:*) :: mappingArray1
!INTEGER(kind=2), DIMENSION(0:*) :: mappingArray2
!INTEGER(kind=2), DIMENSION(0:*) :: mappingArray3
!INTEGER(kind=2), DIMENSION(0:*) :: mappingArray4
!INTEGER(kind=2), DIMENSION(0:*) :: mappingArray5
!INTEGER(kind=4), DIMENSION(0:*) :: ind_sizes
!INTEGER(kind=4), DIMENSION(0:*) :: ind_offs
!INTEGER(kind=4), DIMENSION(0:*) :: blkmap
!INTEGER(kind=4), DIMENSION(0:*) :: offset
!INTEGER(kind=4), DIMENSION(0:*) :: nelems
!INTEGER(kind=4), DIMENSION(0:*) :: nthrcol
!INTEGER(kind=4), DIMENSION(0:*) :: thrcol
!INTEGER(kind=4) :: blockOffset
!INTEGER(kind=4) :: blockID
!INTEGER(kind=4) :: threadBlockOffset
!INTEGER(kind=4) :: threadBlockID
!INTEGER(kind=4) :: numberOfActiveThreads
!INTEGER(kind=4) :: i1
!INTEGER(kind=4) :: i2
!INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat1IndirectionMap
!REAL(kind=8), POINTER, DIMENSION(:) :: opDat1SharedIndirection
!REAL(kind=8), DIMENSION(0:8000 - 1), TARGET :: sharedFloat8
!INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat3IndirectionMap
!REAL(kind=8), POINTER, DIMENSION(:) :: opDat3SharedIndirection
!INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat4IndirectionMap
!REAL(kind=8), POINTER, DIMENSION(:) :: opDat4SharedIndirection
!INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat5IndirectionMap
!REAL(kind=8), POINTER, DIMENSION(:) :: opDat5SharedIndirection
!INTEGER(kind=4) :: opDat1nBytes
!INTEGER(kind=4) :: opDat3nBytes
!INTEGER(kind=4) :: opDat4nBytes
!INTEGER(kind=4) :: opDat5nBytes
!INTEGER(kind=4) :: opDat1RoundUp
!INTEGER(kind=4) :: opDat3RoundUp
!INTEGER(kind=4) :: opDat4RoundUp
!INTEGER(kind=4) :: opDat5RoundUp
!INTEGER(kind=4) :: opDat1SharedIndirectionSize
!INTEGER(kind=4) :: opDat3SharedIndirectionSize
!INTEGER(kind=4) :: opDat4SharedIndirectionSize
!INTEGER(kind=4) :: opDat5SharedIndirectionSize
!REAL(kind=8), DIMENSION(0:3) :: opDat5Local
!INTEGER(kind=4) :: opDat5Map
!INTEGER(kind=4) :: numOfColours
!INTEGER(kind=4) :: numberOfActiveThreadsCeiling
!INTEGER(kind=4) :: colour1
!INTEGER(kind=4) :: colour2
!threadBlockID = blkmap(blockID + blockOffset)
!numberOfActiveThreads = nelems(threadBlockID)
!threadBlockOffset = offset(threadBlockID)
!numberOfActiveThreadsCeiling = numberOfActiveThreads
!numOfColours = nthrcol(threadBlockID)
!opDat1SharedIndirectionSize = ind_sizes(0 + threadBlockID * 4)
!opDat3SharedIndirectionSize = ind_sizes(1 + threadBlockID * 4)
!opDat4SharedIndirectionSize = ind_sizes(2 + threadBlockID * 4)
!opDat5SharedIndirectionSize = ind_sizes(3 + threadBlockID * 4)
!opDat1IndirectionMap => ind_maps1(ind_offs(0 + threadBlockID * 4):)
!opDat3IndirectionMap => ind_maps3(ind_offs(1 + threadBlockID * 4):)
!opDat4IndirectionMap => ind_maps4(ind_offs(2 + threadBlockID * 4):)
!opDat5IndirectionMap => ind_maps5(ind_offs(3 + threadBlockID * 4):)
!opDat3RoundUp = opDat1SharedIndirectionSize * 2
!opDat4RoundUp = opDat3SharedIndirectionSize * 4
!opDat5RoundUp = opDat4SharedIndirectionSize * 1
!opDat1nBytes = 0
!opDat3nBytes = opDat1nBytes + opDat3RoundUp
!opDat4nBytes = opDat3nBytes + opDat4RoundUp
!opDat5nBytes = opDat4nBytes + opDat5RoundUp
!opDat1SharedIndirection => sharedFloat8(opDat1nBytes:)
!opDat3SharedIndirection => sharedFloat8(opDat3nBytes:)
!opDat4SharedIndirection => sharedFloat8(opDat4nBytes:)
!opDat5SharedIndirection => sharedFloat8(opDat5nBytes:)
!DO i1 = 0, opDat1SharedIndirectionSize - 1, 1
!DO i2 = 0, 2 - 1, 1
!opDat1SharedIndirection(i2 + i1 * 2 + 1) = opDat1(i2 + opDat1IndirectionMap(i1 + 1) * 2)
!END DO
!END DO
!DO i1 = 0, opDat3SharedIndirectionSize - 1, 1
!DO i2 = 0, 4 - 1, 1
!opDat3SharedIndirection(i2 + i1 * 4 + 1) = opDat3(i2 + opDat3IndirectionMap(i1 + 1) * 4)
!END DO
!END DO
!DO i1 = 0, opDat4SharedIndirectionSize - 1, 1
!DO i2 = 0, 1 - 1, 1
!opDat4SharedIndirection(i2 + i1 * 1 + 1) = opDat4(i2 + opDat4IndirectionMap(i1 + 1) * 1)
!END DO
!END DO
!DO i1 = 0, opDat5SharedIndirectionSize - 1, 1
!DO i2 = 0, 4 - 1, 1
!opDat5SharedIndirection(i2 + i1 * 4 + 1) = 0
!END DO
!END DO
!DO i1 = 0, numberOfActiveThreadsCeiling - 1, 1
!colour2 = -1
!IF (i1 < numberOfActiveThreads) THEN
!DO i2 = 0, 4 - 1, 1
!opDat5Local(i2) = 0
!END DO
!CALL bres_calc_modified(opDat1SharedIndirection(1 + mappingArray1(i1 + threadBlockOffset) * 2:1 + mappingArray1(i1 + threadBlockOffset) * 2 + 2 - 1),opDat1SharedIndirection(1 + mappingArray2(i1 + threadBlockOffset) * 2:1 + mappingArray2(i1 + threadBlockOffset) * 2 + 2 - 1),opDat3SharedIndirection(1 + mappingArray3(i1 + threadBlockOffset) * 4:1 + mappingArray3(i1 + threadBlockOffset) * 4 + 4 - 1),opDat4SharedIndirection(1 + mappingArray4(i1 + threadBlockOffset) * 1:1 + mappingArray4(i1 + threadBlockOffset) * 1 + 1 - 1),opDat5Local,opDat6((i1 + threadBlockOffset) * 1:(i1 + threadBlockOffset) * 1 + 1 - 1))
!colour2 = thrcol(i1 + threadBlockOffset)
!END IF
!opDat5Map = mappingArray5(i1 + threadBlockOffset)
!DO colour1 = 0, numOfColours - 1, 1
!IF (colour2 .EQ. colour1) THEN
!DO i2 = 0, 4 - 1, 1
!opDat5SharedIndirection(1 + (i2 + opDat5Map * 4)) = opDat5SharedIndirection(1 + (i2 + opDat5Map * 4)) + opDat5Local(i2)
!END DO
!END IF
!END DO
!END DO
!DO i1 = 0, opDat5SharedIndirectionSize - 1, 1
!DO i2 = 0, 4 - 1, 1
!opDat5(i2 + opDat5IndirectionMap(i1 + 1) * 4) = opDat5(i2 + opDat5IndirectionMap(i1 + 1) * 4) + opDat5SharedIndirection(1 + (i2 + i1 * 4))
!END DO
!END DO
!END SUBROUTINE 

SUBROUTINE bres_calc_kernel(opDat1,opDat3,opDat4,opDat5,opDat6,ind_maps1,ind_maps3,ind_maps4,ind_maps5,mappingArray1,mappingArray2,mappingArray3,mappingArray4,mappingArray5,ind_sizes,ind_offs,blkmap,offset,nelems,nthrcol,thrcol,blockOffset,blockID)
IMPLICIT NONE
REAL(kind=8), DIMENSION(0:*) :: opDat1
REAL(kind=8), DIMENSION(0:*) :: opDat3
REAL(kind=8), DIMENSION(0:*) :: opDat4
REAL(kind=8), DIMENSION(0:*) :: opDat5
INTEGER(kind=4), DIMENSION(0:*) :: opDat6
INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps1
INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps3
INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps4
INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps5
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray1
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray2
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray3
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray4
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray5
INTEGER(kind=4), DIMENSION(0:*) :: ind_sizes
INTEGER(kind=4), DIMENSION(0:*) :: ind_offs
INTEGER(kind=4), DIMENSION(0:*) :: blkmap
INTEGER(kind=4), DIMENSION(0:*) :: offset
INTEGER(kind=4), DIMENSION(0:*) :: nelems
INTEGER(kind=4), DIMENSION(0:*) :: nthrcol
INTEGER(kind=4), DIMENSION(0:*) :: thrcol
INTEGER(kind=4) :: blockOffset
INTEGER(kind=4) :: blockID
INTEGER(kind=4) :: threadBlockOffset
INTEGER(kind=4) :: threadBlockID
INTEGER(kind=4) :: numberOfActiveThreads
INTEGER(kind=4) :: i1
INTEGER(kind=4) :: i2
INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat1IndirectionMap
REAL(kind=8), POINTER, DIMENSION(:) :: opDat1SharedIndirection
REAL(kind=8), DIMENSION(0:8000 - 1), TARGET :: sharedFloat8
INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat3IndirectionMap
REAL(kind=8), POINTER, DIMENSION(:) :: opDat3SharedIndirection
INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat4IndirectionMap
REAL(kind=8), POINTER, DIMENSION(:) :: opDat4SharedIndirection
INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat5IndirectionMap
REAL(kind=8), POINTER, DIMENSION(:) :: opDat5SharedIndirection
INTEGER(kind=4) :: opDat1nBytes
INTEGER(kind=4) :: opDat3nBytes
INTEGER(kind=4) :: opDat4nBytes
INTEGER(kind=4) :: opDat5nBytes
INTEGER(kind=4) :: opDat1RoundUp
INTEGER(kind=4) :: opDat3RoundUp
INTEGER(kind=4) :: opDat4RoundUp
INTEGER(kind=4) :: opDat5RoundUp
INTEGER(kind=4) :: opDat1SharedIndirectionSize
INTEGER(kind=4) :: opDat3SharedIndirectionSize
INTEGER(kind=4) :: opDat4SharedIndirectionSize
INTEGER(kind=4) :: opDat5SharedIndirectionSize
REAL(kind=8), allocatable, DIMENSION(:) :: opDat5Local
INTEGER(kind=4) :: opDat5Map
INTEGER(kind=4) :: numOfColours
INTEGER(kind=4) :: numberOfActiveThreadsCeiling
INTEGER(kind=4) :: colour1
INTEGER(kind=4) :: colour2
integer(8) :: time1, time2, count_rate, count_max
integer(8) :: time1g, time2g, count_rateg, count_maxg
real(kind=8), allocatable, dimension(:) :: arg1, arg2, arg3, arg4, arg5
threadBlockID = blkmap(blockID + blockOffset)
numberOfActiveThreads = nelems(threadBlockID)
threadBlockOffset = offset(threadBlockID)
numberOfActiveThreadsCeiling = numberOfActiveThreads
allocate (arg1(numberOfActiveThreads*2))
allocate (arg2(numberOfActiveThreads*2))
allocate (arg3(numberOfActiveThreads*4))
allocate (arg4(numberOfActiveThreads))
allocate (arg5(numberOfActiveThreads*4))
numOfColours = nthrcol(threadBlockID)
opDat1SharedIndirectionSize = ind_sizes(0 + threadBlockID * 4)
opDat3SharedIndirectionSize = ind_sizes(1 + threadBlockID * 4)
opDat4SharedIndirectionSize = ind_sizes(2 + threadBlockID * 4)
opDat5SharedIndirectionSize = ind_sizes(3 + threadBlockID * 4)
opDat1IndirectionMap => ind_maps1(ind_offs(0 + threadBlockID * 4):)
opDat3IndirectionMap => ind_maps3(ind_offs(1 + threadBlockID * 4):)
opDat4IndirectionMap => ind_maps4(ind_offs(2 + threadBlockID * 4):)
opDat5IndirectionMap => ind_maps5(ind_offs(3 + threadBlockID * 4):)
opDat3RoundUp = opDat1SharedIndirectionSize * 2
opDat4RoundUp = opDat3SharedIndirectionSize * 4
opDat5RoundUp = opDat4SharedIndirectionSize * 1
opDat1nBytes = 0
opDat3nBytes = opDat1nBytes + opDat3RoundUp
opDat4nBytes = opDat3nBytes + opDat4RoundUp
opDat5nBytes = opDat4nBytes + opDat5RoundUp
opDat1SharedIndirection => sharedFloat8(opDat1nBytes:)
opDat3SharedIndirection => sharedFloat8(opDat3nBytes:)
opDat4SharedIndirection => sharedFloat8(opDat4nBytes:)
opDat5SharedIndirection => sharedFloat8(opDat5nBytes:)
DO i1 = 0, opDat1SharedIndirectionSize - 1, 1
  DO i2 = 0, 2 - 1, 1
    opDat1SharedIndirection(i2 + i1 * 2 + 1) = opDat1(i2 + opDat1IndirectionMap(i1 + 1) * 2)
  END DO
END DO
DO i1 = 0, opDat3SharedIndirectionSize - 1, 1
  DO i2 = 0, 4 - 1, 1
    opDat3SharedIndirection(i2 + i1 * 4 + 1) = opDat3(i2 + opDat3IndirectionMap(i1 + 1) * 4)
  END DO
END DO
DO i1 = 0, opDat4SharedIndirectionSize - 1, 1
  DO i2 = 0, 1 - 1, 1
    opDat4SharedIndirection(i2 + i1 * 1 + 1) = opDat4(i2 + opDat4IndirectionMap(i1 + 1) * 1)
  END DO
END DO
DO i1 = 0, opDat5SharedIndirectionSize - 1, 1
  DO i2 = 0, 4 - 1, 1
    opDat5SharedIndirection(i2 + i1 * 4 + 1) = 0
  END DO
END DO
call system_clock(time1g, count_rateg, count_maxg)
do i1 = 0, numberOfActiveThreads - 1, 1
  arg1(i1 + 1) = opDat1SharedIndirection(1 + mappingArray1(i1 + threadBlockOffset)*2)
  arg1(i1 + numberOfActiveThreads + 1) = opDat1SharedIndirection(1 + mappingArray1(i1 + threadBlockOffset) * 2 + 1)
  arg2(i1 + 1) = opDat1SharedIndirection(1 + mappingArray2(i1 + threadBlockOffset) * 2)
  arg2(i1 + numberOfActiveThreads + 1) = opDat1SharedIndirection(1 + mappingArray2(i1 + threadBlockOffset) * 2 + 1)
  arg3(i1 + 1) = opDat3SharedIndirection(1 + mappingArray3(i1 + threadBlockOffset) * 4)
  arg3(i1 + numberOfActiveThreads + 1) = opDat3SharedIndirection(1 + mappingArray3(i1 + threadBlockOffset) * 4 + 1)
  arg3(i1 + 2*numberOfActiveThreads + 1) = opDat3SharedIndirection(1 + mappingArray3(i1 + threadBlockOffset) * 4 + 2)
  arg3(i1 + 3*numberOfActiveThreads + 1) = opDat3SharedIndirection(1 + mappingArray3(i1 + threadBlockOffset) * 4 + 3)
  arg4(i1 + 1) = opDat4SharedIndirection(1 + mappingArray4(i1 + threadBlockOffset))
  arg5(i1+1) = 0.d0
  arg5(i1+numberOfActiveThreads+1)=0.d0
  arg5(i1+2*numberOfActiveThreads+1)=0.d0
  arg5(i1+3*numberOfActiveThreads+1)=0.d0
end do
call system_clock(time1,count_rate,count_max)
call bres_calc_kernel_caller(arg1, arg2, arg3, arg4, arg5, opDat6(threadBlockOffset: threadBlockOffset+numberOfActiveThreads -1), numberOfActiveThreads)
call system_clock(time2, count_rate,count_max)
print *, "### bres_calc", time2 - time1
do i1 = 0, numberOfActiveThreadsCeiling - 1, 1
opDat1SharedIndirection(1+mappingArray1(i1+threadBlockOffset)*2)=arg1(i1+1)
opDat1SharedIndirection(1+mappingArray1(i1+threadBlockOffset)*2+1)=arg1(i1+numberOfActiveThreads+1)
opDat1SharedIndirection(1+mappingArray2(i1+threadBlockOffset)*2)=arg2(i1+1)
opDat1SharedIndirection(1+mappingArray2(i1+threadBlockOffset)*2+1)=arg2(i1+numberOfActiveThreads+1)
opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4)=arg3(i1+1)
opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4+1)=arg3(i1+numberOfActiveThreads+1)
opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4+2)=arg3(i1+2*numberOfActiveThreads+1)
opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4+3)=arg3(i1+3*numberOfActiveThreads+1)
opDat4SharedIndirection(1+mappingArray4(i1+threadBlockOffset))=arg4(i1+1)
end do
call system_clock(time2g, count_rateg, count_maxg)
print *, "### bres_gather", time2g - time1g

DO i1 = 0, numberOfActiveThreadsCeiling - 1, 1
  colour2 = -1
   colour2 = thrcol(i1 + threadBlockOffset)
  opDat5Map = mappingArray5(i1 + threadBlockOffset)
  DO colour1 = 0, numOfColours - 1, 1
    IF (colour2 .EQ. colour1) THEN
      DO i2 = 0, 4 - 1, 1
        opDat5SharedIndirection(1 + (i2 + opDat5Map * 4)) = opDat5SharedIndirection(1 + (i2 + opDat5Map * 4)) + arg5(i2*numberOfActiveThreadsCeiling + i1 + 1)
      END DO
    END IF
  END DO
END DO

DO i1 = 0, opDat5SharedIndirectionSize - 1, 1
DO i2 = 0, 4 - 1, 1
opDat5(i2 + opDat5IndirectionMap(i1 + 1) * 4) = opDat5(i2 + opDat5IndirectionMap(i1 + 1) * 4) + opDat5SharedIndirection(1 + (i2 + i1 * 4))
END DO
END DO
deallocate (arg1)
deallocate (arg2)
deallocate (arg3)
deallocate (arg4)
deallocate (arg5)
END SUBROUTINE 

SUBROUTINE bres_calc_host(userSubroutine,set,opDat1,opIndirection1,opMap1,opAccess1,opDat2,opIndirection2,opMap2,opAccess2,opDat3,opIndirection3,opMap3,opAccess3,opDat4,opIndirection4,opMap4,opAccess4,opDat5,opIndirection5,opMap5,opAccess5,opDat6,opIndirection6,opMap6,opAccess6)
IMPLICIT NONE
character(len=10), INTENT(IN) :: userSubroutine
TYPE ( op_set ) , INTENT(IN) :: set
TYPE ( op_dat ) , INTENT(IN) :: opDat1
INTEGER(kind=4), INTENT(IN) :: opIndirection1
TYPE ( op_map ) , INTENT(IN) :: opMap1
INTEGER(kind=4), INTENT(IN) :: opAccess1
TYPE ( op_dat ) , INTENT(IN) :: opDat2
INTEGER(kind=4), INTENT(IN) :: opIndirection2
TYPE ( op_map ) , INTENT(IN) :: opMap2
INTEGER(kind=4), INTENT(IN) :: opAccess2
TYPE ( op_dat ) , INTENT(IN) :: opDat3
INTEGER(kind=4), INTENT(IN) :: opIndirection3
TYPE ( op_map ) , INTENT(IN) :: opMap3
INTEGER(kind=4), INTENT(IN) :: opAccess3
TYPE ( op_dat ) , INTENT(IN) :: opDat4
INTEGER(kind=4), INTENT(IN) :: opIndirection4
TYPE ( op_map ) , INTENT(IN) :: opMap4
INTEGER(kind=4), INTENT(IN) :: opAccess4
TYPE ( op_dat ) , INTENT(IN) :: opDat5
INTEGER(kind=4), INTENT(IN) :: opIndirection5
TYPE ( op_map ) , INTENT(IN) :: opMap5
INTEGER(kind=4), INTENT(IN) :: opAccess5
TYPE ( op_dat ) , INTENT(IN) :: opDat6
INTEGER(kind=4), INTENT(IN) :: opIndirection6
TYPE ( op_map ) , INTENT(IN) :: opMap6
INTEGER(kind=4), INTENT(IN) :: opAccess6
TYPE ( op_set_core ) , POINTER :: opSetCore
TYPE ( op_dat_core ) , POINTER :: opDat1Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat1Local
INTEGER(kind=4) :: opDat1Cardinality
TYPE ( op_set_core ) , POINTER :: opSet1Core
TYPE ( op_map_core ) , POINTER :: opMap1Core
TYPE ( op_dat_core ) , POINTER :: opDat2Core
TYPE ( op_map_core ) , POINTER :: opMap2Core
TYPE ( op_dat_core ) , POINTER :: opDat3Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat3Local
INTEGER(kind=4) :: opDat3Cardinality
TYPE ( op_set_core ) , POINTER :: opSet3Core
TYPE ( op_map_core ) , POINTER :: opMap3Core
TYPE ( op_dat_core ) , POINTER :: opDat4Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat4Local
INTEGER(kind=4) :: opDat4Cardinality
TYPE ( op_set_core ) , POINTER :: opSet4Core
TYPE ( op_map_core ) , POINTER :: opMap4Core
TYPE ( op_dat_core ) , POINTER :: opDat5Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat5Local
INTEGER(kind=4) :: opDat5Cardinality
TYPE ( op_set_core ) , POINTER :: opSet5Core
TYPE ( op_map_core ) , POINTER :: opMap5Core
TYPE ( op_dat_core ) , POINTER :: opDat6Core
INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat6Local
INTEGER(kind=4) :: opDat6Cardinality
TYPE ( op_set_core ) , POINTER :: opSet6Core
TYPE ( op_map_core ) , POINTER :: opMap6Core
INTEGER(kind=4) :: i1
INTEGER(kind=4) :: numberOfThreads
INTEGER(kind=4), DIMENSION(1:6) :: opDatArray
INTEGER(kind=4), DIMENSION(1:6) :: mappingIndicesArray
INTEGER(kind=4), DIMENSION(1:6) :: mappingArray
INTEGER(kind=4), DIMENSION(1:6) :: accessDescriptorArray
INTEGER(kind=4), DIMENSION(1:6) :: indirectionDescriptorArray
INTEGER(kind=4), DIMENSION(1:6) :: opDatTypesArray
INTEGER(kind=4) :: numberOfOpDats
INTEGER(kind=4) :: numberOfIndirectOpDats
INTEGER(kind=4) :: blockOffset
INTEGER(kind=4) :: nblocks
INTEGER(kind=4) :: i2
IF (set%setPtr%size .EQ. 0) THEN
RETURN
END IF
#ifdef _OPENMP 
numberOfThreads = omp_get_max_threads()
#else 
numberOfThreads = 1
#endif 
opSetCore => set%setPtr
opMap1Core => opMap1%mapPtr
opMap2Core => opMap2%mapPtr
opMap3Core => opMap3%mapPtr
opMap4Core => opMap4%mapPtr
opMap5Core => opMap5%mapPtr
opMap6Core => opMap6%mapPtr
opDat1Core => opDat1%dataPtr
opDat2Core => opDat2%dataPtr
opDat3Core => opDat3%dataPtr
opDat4Core => opDat4%dataPtr
opDat5Core => opDat5%dataPtr
opDat6Core => opDat6%dataPtr
CALL c_f_pointer(opDat1Core%set,opSet1Core)
CALL c_f_pointer(opDat3Core%set,opSet3Core)
CALL c_f_pointer(opDat4Core%set,opSet4Core)
CALL c_f_pointer(opDat5Core%set,opSet5Core)
CALL c_f_pointer(opDat6Core%set,opSet6Core)
opDat1Cardinality = opDat1Core%dim * opSet1Core%size
opDat3Cardinality = opDat3Core%dim * opSet3Core%size
opDat4Cardinality = opDat4Core%dim * opSet4Core%size
opDat5Cardinality = opDat5Core%dim * opSet5Core%size
opDat6Cardinality = opDat6Core%dim * opSet6Core%size
CALL c_f_pointer(opDat1Core%dat,opDat1Local,(/opDat1Cardinality/))
CALL c_f_pointer(opDat3Core%dat,opDat3Local,(/opDat3Cardinality/))
CALL c_f_pointer(opDat4Core%dat,opDat4Local,(/opDat4Cardinality/))
CALL c_f_pointer(opDat5Core%dat,opDat5Local,(/opDat5Cardinality/))
CALL c_f_pointer(opDat6Core%dat,opDat6Local,(/opDat6Cardinality/))
IF (firstTime_bres_calc .EQV. .TRUE.) THEN
firstTime_bres_calc = .FALSE.
opDatArray(1) = opDat1Core%index
opDatArray(2) = opDat2Core%index
opDatArray(3) = opDat3Core%index
opDatArray(4) = opDat4Core%index
opDatArray(5) = opDat5Core%index
opDatArray(6) = opDat6Core%index
mappingIndicesArray(1) = opIndirection1
mappingIndicesArray(2) = opIndirection2
mappingIndicesArray(3) = opIndirection3
mappingIndicesArray(4) = opIndirection4
mappingIndicesArray(5) = opIndirection5
mappingIndicesArray(6) = opIndirection6
DO i1 = 1, 6, 1
IF (mappingIndicesArray(i1) .NE. -1) THEN
mappingIndicesArray(i1) = mappingIndicesArray(i1) - 1
END IF
END DO
mappingArray(1) = opMap1Core%index
mappingArray(2) = opMap2Core%index
mappingArray(3) = opMap3Core%index
mappingArray(4) = opMap4Core%index
mappingArray(5) = opMap5Core%index
mappingArray(6) = opMap6Core%index
accessDescriptorArray(1) = opAccess1
accessDescriptorArray(2) = opAccess2
accessDescriptorArray(3) = opAccess3
accessDescriptorArray(4) = opAccess4
accessDescriptorArray(5) = opAccess5
accessDescriptorArray(6) = opAccess6
indirectionDescriptorArray(1) = 0
indirectionDescriptorArray(2) = 0
indirectionDescriptorArray(3) = 1
indirectionDescriptorArray(4) = 2
indirectionDescriptorArray(5) = 3
indirectionDescriptorArray(6) = -1
numberOfOpDats = 6
numberOfIndirectOpDats = 4
IF (opMap1Core%dim .EQ. -1) THEN
opDatTypesArray(1) = F_OP_ARG_GBL
ELSE
opDatTypesArray(1) = F_OP_ARG_DAT
END IF
IF (opMap2Core%dim .EQ. -1) THEN
opDatTypesArray(2) = F_OP_ARG_GBL
ELSE
opDatTypesArray(2) = F_OP_ARG_DAT
END IF
IF (opMap3Core%dim .EQ. -1) THEN
opDatTypesArray(3) = F_OP_ARG_GBL
ELSE
opDatTypesArray(3) = F_OP_ARG_DAT
END IF
IF (opMap4Core%dim .EQ. -1) THEN
opDatTypesArray(4) = F_OP_ARG_GBL
ELSE
opDatTypesArray(4) = F_OP_ARG_DAT
END IF
IF (opMap5Core%dim .EQ. -1) THEN
opDatTypesArray(5) = F_OP_ARG_GBL
ELSE
opDatTypesArray(5) = F_OP_ARG_DAT
END IF
IF (opMap6Core%dim .EQ. -1) THEN
opDatTypesArray(6) = F_OP_ARG_GBL
ELSE
opDatTypesArray(6) = F_OP_ARG_DAT
END IF
planRet_bres_calc = cplan_OpenMP(userSubroutine,opSetCore%index,numberOfOpDats,opDatArray,mappingIndicesArray,mappingArray,accessDescriptorArray,numberOfIndirectOpDats,indirectionDescriptorArray,opDatTypesArray,0)
CALL c_f_pointer(planRet_bres_calc,actualPlan_bres_calc)
CALL c_f_pointer(actualPlan_bres_calc%nindirect,pnindirect_bres_calc,(/numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_bres_calc%ind_maps,ind_maps_bres_calc,(/numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_bres_calc%maps,mappingArray_bres_calc,(/numberOfOpDats/))
CALL c_f_pointer(actualPlan_bres_calc%ncolblk,ncolblk_bres_calc,(/opSetCore%size/))
CALL c_f_pointer(actualPlan_bres_calc%ind_sizes,ind_sizes_bres_calc,(/actualPlan_bres_calc%nblocks * numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_bres_calc%ind_offs,ind_offs_bres_calc,(/actualPlan_bres_calc%nblocks * numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_bres_calc%blkmap,blkmap_bres_calc,(/actualPlan_bres_calc%nblocks/))
CALL c_f_pointer(actualPlan_bres_calc%offset,offset_bres_calc,(/actualPlan_bres_calc%nblocks/))
CALL c_f_pointer(actualPlan_bres_calc%nelems,nelems_bres_calc,(/actualPlan_bres_calc%nblocks/))
CALL c_f_pointer(actualPlan_bres_calc%nthrcol,nthrcol_bres_calc,(/actualPlan_bres_calc%nblocks/))
CALL c_f_pointer(actualPlan_bres_calc%thrcol,thrcol_bres_calc,(/opSetCore%size/))
CALL c_f_pointer(ind_maps_bres_calc(1),ind_maps1_bres_calc,(/pnindirect_bres_calc(1)/))
CALL c_f_pointer(ind_maps_bres_calc(2),ind_maps3_bres_calc,(/pnindirect_bres_calc(2)/))
CALL c_f_pointer(ind_maps_bres_calc(3),ind_maps4_bres_calc,(/pnindirect_bres_calc(3)/))
CALL c_f_pointer(ind_maps_bres_calc(4),ind_maps5_bres_calc,(/pnindirect_bres_calc(4)/))
IF (indirectionDescriptorArray(1) >= 0) THEN
CALL c_f_pointer(mappingArray_bres_calc(1),mappingArray1_bres_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(2) >= 0) THEN
CALL c_f_pointer(mappingArray_bres_calc(2),mappingArray2_bres_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(3) >= 0) THEN
CALL c_f_pointer(mappingArray_bres_calc(3),mappingArray3_bres_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(4) >= 0) THEN
CALL c_f_pointer(mappingArray_bres_calc(4),mappingArray4_bres_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(5) >= 0) THEN
CALL c_f_pointer(mappingArray_bres_calc(5),mappingArray5_bres_calc,(/opSetCore%size/))
END IF
END IF
blockOffset = 0
DO i1 = 0, actualPlan_bres_calc%ncolors - 1, 1
nblocks = ncolblk_bres_calc(i1 + 1)
!$OMP PARALLEL DO 
DO i2 = 0, nblocks - 1, 1
CALL bres_calc_kernel(opDat1Local,opDat3Local,opDat4Local,opDat5Local,opDat6Local,ind_maps1_bres_calc,ind_maps3_bres_calc,ind_maps4_bres_calc,ind_maps5_bres_calc,mappingArray1_bres_calc,mappingArray2_bres_calc,mappingArray3_bres_calc,mappingArray4_bres_calc,mappingArray5_bres_calc,ind_sizes_bres_calc,ind_offs_bres_calc,blkmap_bres_calc,offset_bres_calc,nelems_bres_calc,nthrcol_bres_calc,thrcol_bres_calc,blockOffset,i2)
END DO
!$OMP END PARALLEL DO
blockOffset = blockOffset + nblocks
END DO
END SUBROUTINE 

SUBROUTINE res_calc_modified(x1,x2,q1,q2,adt1,adt2,res1,res2)
IMPLICIT NONE
REAL(kind=8), DIMENSION(*) :: x1
REAL(kind=8), DIMENSION(*) :: x2
REAL(kind=8), DIMENSION(*) :: q1
REAL(kind=8), DIMENSION(*) :: q2
REAL(kind=8), DIMENSION(*) :: adt1
REAL(kind=8), DIMENSION(*) :: adt2
REAL(kind=8), DIMENSION(*) :: res1
REAL(kind=8), DIMENSION(*) :: res2
REAL(kind=8) :: dx,dy,mu,ri,p1,vol1,p2,vol2,f
dx = x1(1) - x2(1)
dy = x1(2) - x2(2)
ri = 1.0 / q1(1)
p1 = gm1_OP2_CONSTANT * (q1(4) - 0.5 * ri * (q1(2) * q1(2) + q1(3) * q1(3)))
vol1 = ri * (q1(2) * dy - q1(3) * dx)
ri = 1.0 / q2(1)
p2 = gm1_OP2_CONSTANT * (q2(4) - 0.5 * ri * (q2(2) * q2(2) + q2(3) * q2(3)))
vol2 = ri * (q2(2) * dy - q2(3) * dx)
mu = 0.5 * (adt1(1) + adt2(1)) * eps_OP2_CONSTANT
f = 0.5 * (vol1 * q1(1) + vol2 * q2(1)) + mu * (q1(1) - q2(1))
res1(1) = res1(1) + f
res2(1) = res2(1) - f
f = 0.5 * (vol1 * q1(2) + p1 * dy + vol2 * q2(2) + p2 * dy) + mu * (q1(2) - q2(2))
res1(2) = res1(2) + f
res2(2) = res2(2) - f
f = 0.5 * (vol1 * q1(3) - p1 * dx + vol2 * q2(3) - p2 * dx) + mu * (q1(3) - q2(3))
res1(3) = res1(3) + f
res2(3) = res2(3) - f
f = 0.5 * (vol1 * (q1(4) + p1) + vol2 * (q2(4) + p2)) + mu * (q1(4) - q2(4))
res1(4) = res1(4) + f
res2(4) = res2(4) - f
END SUBROUTINE 

subroutine res_calc_kernel_caller(x1,x2,q1,q2,adt1,adt2,res1,res2,iterations)
  implicit none
  real(kind=8), dimension(BS, iterations/BS, 2) :: x1,x2
  real(kind=8), dimension(BS, iterations/BS, 4) :: q1,q2
  real(kind=8), dimension(BS, iterations/BS) :: adt1,adt2
  real(kind=8), dimension(BS, iterations/BS, 4) :: res1,res2
  integer :: iterations
  integer :: i
  real(kind=8), dimension(BS) :: dx,dy,mu,ri,p1,vol1,p2,vol2,f
  do i = 1, iterations/BS
    dx(:) = x1(:, i, 1) - x2(:, i, 1)
    dy(:) = x1(:, i, 2) - x2(:, i, 2)
    ri(:) = 1.0 / q1(:, i, 1)
    p1(:) = gm1_OP2_CONSTANT * (q1(:, i, 4) - 0.5 * ri(:) * (q1(:, i, 2) * q1(:, i, 2) + q1(:, i, 3) * q1(:, i, 3)))
    vol1(:) = ri(:) * (q1(:, i, 2) * dy(:) - q1(:, i, 3) * dx(:))
    ri(:) = 1.0 / q2(:, i, 1)
    p2(:) = gm1_OP2_CONSTANT * (q2(:, i, 4) - 0.5 * ri(:) * (q2(:, i, 2) * q2(:, i, 2) + q2(:, i, 3) * q2(:, i, 3)))
    vol2(:) = ri(:) * (q2(:, i, 2) * dy(:) - q2(:, i, 3) * dx(:))
    mu(:) = 0.5 * (adt1(:, i) + adt2(:, i)) * eps_OP2_CONSTANT
    f(:) = 0.5 * (vol1(:) * q1(:, i, 1) + vol2(:) * q2(:, i, 1)) + mu(:) * (q1(:, i, 1) - q2(:, i, 1))
    res1(:, i, 1) = res1(:, i, 1) + f(:)
    res2(:, i, 1) = res2(:, i, 1) - f(:)
    f(:) = 0.5 * (vol1(:) * q1(:, i, 2) + p1(:) * dy(:) + vol2(:) * q2(:, i, 2) + p2(:) * dy(:)) + mu(:) * (q1(:, i, 2) - q2(:, i, 2))
    res1(:, i, 2) = res1(:, i, 2) + f(:)
    res2(:, i, 2) = res2(:, i, 2) - f(:)
    f(:) = 0.5 * (vol1(:) * q1(:, i, 3) - p1(:) * dx(:) + vol2(:) * q2(:, i, 3) - p2(:) * dx(:)) + mu(:) * (q1(:, i, 3) - q2(:, i, 3))
    res1(:, i, 3) = res1(:, i, 3) + f(:)
    res2(:, i, 3) = res2(:, i, 3) - f(:)
    f(:) = 0.5 * (vol1(:) * (q1(:, i, 4) + p1(:)) + vol2(:) * (q2(:, i, 4) + p2(:))) + mu(:) * (q1(:, i, 4) - q2(:, i, 4))
    res1(:, i, 4) = res1(:, i, 4) + f(:)
    res2(:, i, 4) = res2(:, i, 4) - f(:)
  end do
end subroutine

SUBROUTINE res_calc_kernel(opDat1,opDat3,opDat5,opDat7,ind_maps1,ind_maps3,ind_maps5,ind_maps7,mappingArray1,mappingArray2,mappingArray3,mappingArray4,mappingArray5,mappingArray6,mappingArray7,mappingArray8,ind_sizes,ind_offs,blkmap,offset,nelems,nthrcol,thrcol,blockOffset,blockID)
IMPLICIT NONE
REAL(kind=8), DIMENSION(0:*) :: opDat1
REAL(kind=8), DIMENSION(0:*) :: opDat3
REAL(kind=8), DIMENSION(0:*) :: opDat5
REAL(kind=8), DIMENSION(0:*) :: opDat7
INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps1
INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps3
INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps5
INTEGER(kind=4), DIMENSION(0:), TARGET :: ind_maps7
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray1
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray2
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray3
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray4
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray5
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray6
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray7
INTEGER(kind=2), DIMENSION(0:*) :: mappingArray8
INTEGER(kind=4), DIMENSION(0:*) :: ind_sizes
INTEGER(kind=4), DIMENSION(0:*) :: ind_offs
INTEGER(kind=4), DIMENSION(0:*) :: blkmap
INTEGER(kind=4), DIMENSION(0:*) :: offset
INTEGER(kind=4), DIMENSION(0:*) :: nelems
INTEGER(kind=4), DIMENSION(0:*) :: nthrcol
INTEGER(kind=4), DIMENSION(0:*) :: thrcol
INTEGER(kind=4) :: blockOffset
INTEGER(kind=4) :: blockID
INTEGER(kind=4) :: threadBlockOffset
INTEGER(kind=4) :: threadBlockID
INTEGER(kind=4) :: numberOfActiveThreads
INTEGER(kind=4) :: i1
INTEGER(kind=4) :: i2
INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat1IndirectionMap
REAL(kind=8), POINTER, DIMENSION(:) :: opDat1SharedIndirection
REAL(kind=8), DIMENSION(0:8000 - 1), TARGET :: sharedFloat8
INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat3IndirectionMap
REAL(kind=8), POINTER, DIMENSION(:) :: opDat3SharedIndirection
INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat5IndirectionMap
REAL(kind=8), POINTER, DIMENSION(:) :: opDat5SharedIndirection
INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat7IndirectionMap
REAL(kind=8), POINTER, DIMENSION(:) :: opDat7SharedIndirection
INTEGER(kind=4) :: opDat1nBytes
INTEGER(kind=4) :: opDat3nBytes
INTEGER(kind=4) :: opDat5nBytes
INTEGER(kind=4) :: opDat7nBytes
INTEGER(kind=4) :: opDat1RoundUp
INTEGER(kind=4) :: opDat3RoundUp
INTEGER(kind=4) :: opDat5RoundUp
INTEGER(kind=4) :: opDat7RoundUp
INTEGER(kind=4) :: opDat1SharedIndirectionSize
INTEGER(kind=4) :: opDat3SharedIndirectionSize
INTEGER(kind=4) :: opDat5SharedIndirectionSize
INTEGER(kind=4) :: opDat7SharedIndirectionSize
REAL(kind=8), DIMENSION(0:3) :: opDat7Local
INTEGER(kind=4) :: opDat7Map
REAL(kind=8), DIMENSION(0:3) :: opDat8Local
INTEGER(kind=4) :: opDat8Map
INTEGER(kind=4) :: numOfColours
INTEGER(kind=4) :: numberOfActiveThreadsCeiling
INTEGER(kind=4) :: colour1
INTEGER(kind=4) :: colour2
integer(8) :: time1, time2, count_rate, count_max
integer(8) :: time1g, time2g, count_rateg, count_maxg
real(kind=8), allocatable, dimension(:) :: arg1, arg2, arg3
real(kind=8), allocatable, dimension(:) :: arg4, arg5, arg6
real(kind=8), allocatable, dimension(:) :: arg7, arg8
threadBlockID = blkmap(blockID + blockOffset)
numberOfActiveThreads = nelems(threadBlockID)
threadBlockOffset = offset(threadBlockID)
numberOfActiveThreadsCeiling = numberOfActiveThreads
allocate (arg1(numberOfActiveThreads*2))
allocate (arg2(numberOfActiveThreads*2))
allocate (arg3(numberOfActiveThreads*4))
allocate (arg4(numberOfActiveThreads*4))
allocate (arg5(numberOfActiveThreads))
allocate (arg6(numberOfActiveThreads))
allocate (arg7(numberOfActiveThreads*4))
allocate (arg8(numberOfActiveThreads*4))
numOfColours = nthrcol(threadBlockID)
opDat1SharedIndirectionSize = ind_sizes(0 + threadBlockID * 4)
opDat3SharedIndirectionSize = ind_sizes(1 + threadBlockID * 4)
opDat5SharedIndirectionSize = ind_sizes(2 + threadBlockID * 4)
opDat7SharedIndirectionSize = ind_sizes(3 + threadBlockID * 4)
opDat1IndirectionMap => ind_maps1(ind_offs(0 + threadBlockID * 4):)
opDat3IndirectionMap => ind_maps3(ind_offs(1 + threadBlockID * 4):)
opDat5IndirectionMap => ind_maps5(ind_offs(2 + threadBlockID * 4):)
opDat7IndirectionMap => ind_maps7(ind_offs(3 + threadBlockID * 4):)
opDat3RoundUp = opDat1SharedIndirectionSize * 2
opDat5RoundUp = opDat3SharedIndirectionSize * 4
opDat7RoundUp = opDat5SharedIndirectionSize * 1
opDat1nBytes = 0
opDat3nBytes = opDat1nBytes + opDat3RoundUp
opDat5nBytes = opDat3nBytes + opDat5RoundUp
opDat7nBytes = opDat5nBytes + opDat7RoundUp
opDat1SharedIndirection => sharedFloat8(opDat1nBytes:)
opDat3SharedIndirection => sharedFloat8(opDat3nBytes:)
opDat5SharedIndirection => sharedFloat8(opDat5nBytes:)
opDat7SharedIndirection => sharedFloat8(opDat7nBytes:)
DO i1 = 0, opDat1SharedIndirectionSize - 1, 1
DO i2 = 0, 2 - 1, 1
opDat1SharedIndirection(i2 + i1 * 2 + 1) = opDat1(i2 + opDat1IndirectionMap(i1 + 1) * 2)
END DO
END DO
DO i1 = 0, opDat3SharedIndirectionSize - 1, 1
DO i2 = 0, 4 - 1, 1
opDat3SharedIndirection(i2 + i1 * 4 + 1) = opDat3(i2 + opDat3IndirectionMap(i1 + 1) * 4)
END DO
END DO
DO i1 = 0, opDat5SharedIndirectionSize - 1, 1
DO i2 = 0, 1 - 1, 1
opDat5SharedIndirection(i2 + i1 * 1 + 1) = opDat5(i2 + opDat5IndirectionMap(i1 + 1) * 1)
END DO
END DO
DO i1 = 0, opDat7SharedIndirectionSize - 1, 1
DO i2 = 0, 4 - 1, 1
opDat7SharedIndirection(i2 + i1 * 4 + 1) = 0
END DO
END DO

call system_clock(time1g, count_rateg, count_maxg)
do i1=0, numberOfActiveThreads -1, 1
arg1(i1+1)=opDat1SharedIndirection(1+mappingArray1(i1+threadBlockOffset)*2)
arg1(i1+numberOfActiveThreads+1)=opDat1SharedIndirection(1+mappingArray1(i1+threadBlockOffset)*2+1)
arg2(i1+1)=opDat1SharedIndirection(1+mappingArray2(i1+threadBlockOffset)*2)
arg2(i1+numberOfActiveThreads+1)=opDat1SharedIndirection(1+mappingArray2(i1+threadBlockOffset)*2+1)
arg3(i1+1)=opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4)
arg3(i1+numberOfActiveThreads+1)=opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4+1)
arg3(i1+2*numberOfActiveThreads+1)=opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4+2)
arg3(i1+3*numberOfActiveThreads+1)=opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4+3)
arg4(i1+1)=opDat3SharedIndirection(1+mappingArray4(i1+threadBlockOffset)*4)
arg4(i1+numberOfActiveThreads+1)=opDat3SharedIndirection(1+mappingArray4(i1+threadBlockOffset)*4+1)
arg4(i1+2*numberOfActiveThreads+1)=opDat3SharedIndirection(1+mappingArray4(i1+threadBlockOffset)*4+2)
arg4(i1+3*numberOfActiveThreads+1)=opDat3SharedIndirection(1+mappingArray4(i1+threadBlockOffset)*4+3)
arg5(i1+1)=opDat5SharedIndirection(1+mappingArray5(i1+threadBlockOffset))
arg6(i1+1)=opDat5SharedIndirection(1+mappingArray6(i1+threadBlockOffset))
arg7(i1+1) = 0.d0
arg7(i1+numberOfActiveThreads+1)=0.d0
arg7(i1+2*numberOfActiveThreads+1)=0.d0
arg7(i1+3*numberOfActiveThreads+1)=0.d0
arg8(i1+1) = 0.d0
arg8(i1+numberOfActiveThreads+1)=0.d0
arg8(i1+2*numberOfActiveThreads+1)=0.d0
arg8(i1+3*numberOfActiveThreads+1)=0.d0
end do
call system_clock(time1,count_rate, count_max)

call res_calc_kernel_caller(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,numberOfActiveThreads)


call system_clock(time2,count_rate, count_max)
print *, "### res_calc", time2 - time1

do i1=0, numberOfActiveThreads -1, 1
opDat1SharedIndirection(1+mappingArray1(i1+threadBlockOffset)*2)=arg1(i1+1)
opDat1SharedIndirection(1+mappingArray1(i1+threadBlockOffset)*2+1)=arg1(i1+numberOfActiveThreads+1)
opDat1SharedIndirection(1+mappingArray2(i1+threadBlockOffset)*2)=arg2(i1+1)
opDat1SharedIndirection(1+mappingArray2(i1+threadBlockOffset)*2+1)=arg2(i1+numberOfActiveThreads+1)
opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4)=arg3(i1+1)
opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4+1)=arg3(i1+numberOfActiveThreads+1)
opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4+2)=arg3(i1+2*numberOfActiveThreads+1)
opDat3SharedIndirection(1+mappingArray3(i1+threadBlockOffset)*4+3)=arg3(i1+3*numberOfActiveThreads+1)
opDat3SharedIndirection(1+mappingArray4(i1+threadBlockOffset)*4)=arg4(i1+1)
opDat3SharedIndirection(1+mappingArray4(i1+threadBlockOffset)*4+1)=arg4(i1+numberOfActiveThreads+1)
opDat3SharedIndirection(1+mappingArray4(i1+threadBlockOffset)*4+2)=arg4(i1+2*numberOfActiveThreads+1)
opDat3SharedIndirection(1+mappingArray4(i1+threadBlockOffset)*4+3)=arg4(i1+3*numberOfActiveThreads+1)
opDat5SharedIndirection(1+mappingArray5(i1+threadBlockOffset))=arg5(i1+1)
opDat5SharedIndirection(1+mappingArray6(i1+threadBlockOffset))=arg6(i1+1)
end do

call system_clock(time2g, count_rateg, count_maxg)
print *, "### res_gather", time2g - time1g


DO i1 = 0, numberOfActiveThreadsCeiling - 1, 1
colour2 = -1
colour2 = thrcol(i1 + threadBlockOffset)
opDat7Map = mappingArray7(i1 + threadBlockOffset)
opDat8Map = mappingArray8(i1 + threadBlockOffset)
DO colour1 = 0, numOfColours - 1, 1
IF (colour2 .EQ. colour1) THEN
DO i2 = 0, 4 - 1, 1
opDat7SharedIndirection(1 + (i2 + opDat7Map * 4)) = opDat7SharedIndirection(1 + (i2 + opDat7Map * 4)) + arg7(i2*numberOfActiveThreadsCeiling + i1 + 1)
END DO
DO i2 = 0, 4 - 1, 1
opDat7SharedIndirection(1 + (i2 + opDat8Map * 4)) = opDat7SharedIndirection(1 + (i2 + opDat8Map * 4)) + arg8(i2*numberOfActiveThreadsCeiling + i1 + 1)
END DO
END IF
END DO
END DO

DO i1 = 0, opDat7SharedIndirectionSize - 1, 1
DO i2 = 0, 4 - 1, 1
opDat7(i2 + opDat7IndirectionMap(i1 + 1) * 4) = opDat7(i2 + opDat7IndirectionMap(i1 + 1) * 4) + opDat7SharedIndirection(1 + (i2 + i1 * 4))
END DO
END DO
deallocate (arg1)
deallocate (arg2)
deallocate (arg3)
deallocate (arg4)
deallocate (arg5)
deallocate (arg6)
deallocate (arg7)
deallocate (arg8)
END SUBROUTINE 

SUBROUTINE res_calc_host(userSubroutine,set,opDat1,opIndirection1,opMap1,opAccess1,opDat2,opIndirection2,opMap2,opAccess2,opDat3,opIndirection3,opMap3,opAccess3,opDat4,opIndirection4,opMap4,opAccess4,opDat5,opIndirection5,opMap5,opAccess5,opDat6,opIndirection6,opMap6,opAccess6,opDat7,opIndirection7,opMap7,opAccess7,opDat8,opIndirection8,opMap8,opAccess8)
IMPLICIT NONE
character(len=9), INTENT(IN) :: userSubroutine
TYPE ( op_set ) , INTENT(IN) :: set
TYPE ( op_dat ) , INTENT(IN) :: opDat1
INTEGER(kind=4), INTENT(IN) :: opIndirection1
TYPE ( op_map ) , INTENT(IN) :: opMap1
INTEGER(kind=4), INTENT(IN) :: opAccess1
TYPE ( op_dat ) , INTENT(IN) :: opDat2
INTEGER(kind=4), INTENT(IN) :: opIndirection2
TYPE ( op_map ) , INTENT(IN) :: opMap2
INTEGER(kind=4), INTENT(IN) :: opAccess2
TYPE ( op_dat ) , INTENT(IN) :: opDat3
INTEGER(kind=4), INTENT(IN) :: opIndirection3
TYPE ( op_map ) , INTENT(IN) :: opMap3
INTEGER(kind=4), INTENT(IN) :: opAccess3
TYPE ( op_dat ) , INTENT(IN) :: opDat4
INTEGER(kind=4), INTENT(IN) :: opIndirection4
TYPE ( op_map ) , INTENT(IN) :: opMap4
INTEGER(kind=4), INTENT(IN) :: opAccess4
TYPE ( op_dat ) , INTENT(IN) :: opDat5
INTEGER(kind=4), INTENT(IN) :: opIndirection5
TYPE ( op_map ) , INTENT(IN) :: opMap5
INTEGER(kind=4), INTENT(IN) :: opAccess5
TYPE ( op_dat ) , INTENT(IN) :: opDat6
INTEGER(kind=4), INTENT(IN) :: opIndirection6
TYPE ( op_map ) , INTENT(IN) :: opMap6
INTEGER(kind=4), INTENT(IN) :: opAccess6
TYPE ( op_dat ) , INTENT(IN) :: opDat7
INTEGER(kind=4), INTENT(IN) :: opIndirection7
TYPE ( op_map ) , INTENT(IN) :: opMap7
INTEGER(kind=4), INTENT(IN) :: opAccess7
TYPE ( op_dat ) , INTENT(IN) :: opDat8
INTEGER(kind=4), INTENT(IN) :: opIndirection8
TYPE ( op_map ) , INTENT(IN) :: opMap8
INTEGER(kind=4), INTENT(IN) :: opAccess8
TYPE ( op_set_core ) , POINTER :: opSetCore
TYPE ( op_dat_core ) , POINTER :: opDat1Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat1Local
INTEGER(kind=4) :: opDat1Cardinality
TYPE ( op_set_core ) , POINTER :: opSet1Core
TYPE ( op_map_core ) , POINTER :: opMap1Core
TYPE ( op_dat_core ) , POINTER :: opDat2Core
TYPE ( op_map_core ) , POINTER :: opMap2Core
TYPE ( op_dat_core ) , POINTER :: opDat3Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat3Local
INTEGER(kind=4) :: opDat3Cardinality
TYPE ( op_set_core ) , POINTER :: opSet3Core
TYPE ( op_map_core ) , POINTER :: opMap3Core
TYPE ( op_dat_core ) , POINTER :: opDat4Core
TYPE ( op_map_core ) , POINTER :: opMap4Core
TYPE ( op_dat_core ) , POINTER :: opDat5Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat5Local
INTEGER(kind=4) :: opDat5Cardinality
TYPE ( op_set_core ) , POINTER :: opSet5Core
TYPE ( op_map_core ) , POINTER :: opMap5Core
TYPE ( op_dat_core ) , POINTER :: opDat6Core
TYPE ( op_map_core ) , POINTER :: opMap6Core
TYPE ( op_dat_core ) , POINTER :: opDat7Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat7Local
INTEGER(kind=4) :: opDat7Cardinality
TYPE ( op_set_core ) , POINTER :: opSet7Core
TYPE ( op_map_core ) , POINTER :: opMap7Core
TYPE ( op_dat_core ) , POINTER :: opDat8Core
TYPE ( op_map_core ) , POINTER :: opMap8Core
INTEGER(kind=4) :: i1
INTEGER(kind=4) :: numberOfThreads
INTEGER(kind=4), DIMENSION(1:8) :: opDatArray
INTEGER(kind=4), DIMENSION(1:8) :: mappingIndicesArray
INTEGER(kind=4), DIMENSION(1:8) :: mappingArray
INTEGER(kind=4), DIMENSION(1:8) :: accessDescriptorArray
INTEGER(kind=4), DIMENSION(1:8) :: indirectionDescriptorArray
INTEGER(kind=4), DIMENSION(1:8) :: opDatTypesArray
INTEGER(kind=4) :: numberOfOpDats
INTEGER(kind=4) :: numberOfIndirectOpDats
INTEGER(kind=4) :: blockOffset
INTEGER(kind=4) :: nblocks
INTEGER(kind=4) :: i2
IF (set%setPtr%size .EQ. 0) THEN
RETURN
END IF
#ifdef _OPENMP 
numberOfThreads = omp_get_max_threads()
#else 
numberOfThreads = 1
#endif 
opSetCore => set%setPtr
opMap1Core => opMap1%mapPtr
opMap2Core => opMap2%mapPtr
opMap3Core => opMap3%mapPtr
opMap4Core => opMap4%mapPtr
opMap5Core => opMap5%mapPtr
opMap6Core => opMap6%mapPtr
opMap7Core => opMap7%mapPtr
opMap8Core => opMap8%mapPtr
opDat1Core => opDat1%dataPtr
opDat2Core => opDat2%dataPtr
opDat3Core => opDat3%dataPtr
opDat4Core => opDat4%dataPtr
opDat5Core => opDat5%dataPtr
opDat6Core => opDat6%dataPtr
opDat7Core => opDat7%dataPtr
opDat8Core => opDat8%dataPtr
CALL c_f_pointer(opDat1Core%set,opSet1Core)
CALL c_f_pointer(opDat3Core%set,opSet3Core)
CALL c_f_pointer(opDat5Core%set,opSet5Core)
CALL c_f_pointer(opDat7Core%set,opSet7Core)
opDat1Cardinality = opDat1Core%dim * opSet1Core%size
opDat3Cardinality = opDat3Core%dim * opSet3Core%size
opDat5Cardinality = opDat5Core%dim * opSet5Core%size
opDat7Cardinality = opDat7Core%dim * opSet7Core%size
CALL c_f_pointer(opDat1Core%dat,opDat1Local,(/opDat1Cardinality/))
CALL c_f_pointer(opDat3Core%dat,opDat3Local,(/opDat3Cardinality/))
CALL c_f_pointer(opDat5Core%dat,opDat5Local,(/opDat5Cardinality/))
CALL c_f_pointer(opDat7Core%dat,opDat7Local,(/opDat7Cardinality/))
IF (firstTime_res_calc .EQV. .TRUE.) THEN
firstTime_res_calc = .FALSE.
opDatArray(1) = opDat1Core%index
opDatArray(2) = opDat2Core%index
opDatArray(3) = opDat3Core%index
opDatArray(4) = opDat4Core%index
opDatArray(5) = opDat5Core%index
opDatArray(6) = opDat6Core%index
opDatArray(7) = opDat7Core%index
opDatArray(8) = opDat8Core%index
mappingIndicesArray(1) = opIndirection1
mappingIndicesArray(2) = opIndirection2
mappingIndicesArray(3) = opIndirection3
mappingIndicesArray(4) = opIndirection4
mappingIndicesArray(5) = opIndirection5
mappingIndicesArray(6) = opIndirection6
mappingIndicesArray(7) = opIndirection7
mappingIndicesArray(8) = opIndirection8
DO i1 = 1, 8, 1
IF (mappingIndicesArray(i1) .NE. -1) THEN
mappingIndicesArray(i1) = mappingIndicesArray(i1) - 1
END IF
END DO
mappingArray(1) = opMap1Core%index
mappingArray(2) = opMap2Core%index
mappingArray(3) = opMap3Core%index
mappingArray(4) = opMap4Core%index
mappingArray(5) = opMap5Core%index
mappingArray(6) = opMap6Core%index
mappingArray(7) = opMap7Core%index
mappingArray(8) = opMap8Core%index
accessDescriptorArray(1) = opAccess1
accessDescriptorArray(2) = opAccess2
accessDescriptorArray(3) = opAccess3
accessDescriptorArray(4) = opAccess4
accessDescriptorArray(5) = opAccess5
accessDescriptorArray(6) = opAccess6
accessDescriptorArray(7) = opAccess7
accessDescriptorArray(8) = opAccess8
indirectionDescriptorArray(1) = 0
indirectionDescriptorArray(2) = 0
indirectionDescriptorArray(3) = 1
indirectionDescriptorArray(4) = 1
indirectionDescriptorArray(5) = 2
indirectionDescriptorArray(6) = 2
indirectionDescriptorArray(7) = 3
indirectionDescriptorArray(8) = 3
numberOfOpDats = 8
numberOfIndirectOpDats = 4
IF (opMap1Core%dim .EQ. -1) THEN
opDatTypesArray(1) = F_OP_ARG_GBL
ELSE
opDatTypesArray(1) = F_OP_ARG_DAT
END IF
IF (opMap2Core%dim .EQ. -1) THEN
opDatTypesArray(2) = F_OP_ARG_GBL
ELSE
opDatTypesArray(2) = F_OP_ARG_DAT
END IF
IF (opMap3Core%dim .EQ. -1) THEN
opDatTypesArray(3) = F_OP_ARG_GBL
ELSE
opDatTypesArray(3) = F_OP_ARG_DAT
END IF
IF (opMap4Core%dim .EQ. -1) THEN
opDatTypesArray(4) = F_OP_ARG_GBL
ELSE
opDatTypesArray(4) = F_OP_ARG_DAT
END IF
IF (opMap5Core%dim .EQ. -1) THEN
opDatTypesArray(5) = F_OP_ARG_GBL
ELSE
opDatTypesArray(5) = F_OP_ARG_DAT
END IF
IF (opMap6Core%dim .EQ. -1) THEN
opDatTypesArray(6) = F_OP_ARG_GBL
ELSE
opDatTypesArray(6) = F_OP_ARG_DAT
END IF
IF (opMap7Core%dim .EQ. -1) THEN
opDatTypesArray(7) = F_OP_ARG_GBL
ELSE
opDatTypesArray(7) = F_OP_ARG_DAT
END IF
IF (opMap8Core%dim .EQ. -1) THEN
opDatTypesArray(8) = F_OP_ARG_GBL
ELSE
opDatTypesArray(8) = F_OP_ARG_DAT
END IF
planRet_res_calc = cplan_OpenMP(userSubroutine,opSetCore%index,numberOfOpDats,opDatArray,mappingIndicesArray,mappingArray,accessDescriptorArray,numberOfIndirectOpDats,indirectionDescriptorArray,opDatTypesArray,0)
CALL c_f_pointer(planRet_res_calc,actualPlan_res_calc)
CALL c_f_pointer(actualPlan_res_calc%nindirect,pnindirect_res_calc,(/numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_res_calc%ind_maps,ind_maps_res_calc,(/numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_res_calc%maps,mappingArray_res_calc,(/numberOfOpDats/))
CALL c_f_pointer(actualPlan_res_calc%ncolblk,ncolblk_res_calc,(/opSetCore%size/))
CALL c_f_pointer(actualPlan_res_calc%ind_sizes,ind_sizes_res_calc,(/actualPlan_res_calc%nblocks * numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_res_calc%ind_offs,ind_offs_res_calc,(/actualPlan_res_calc%nblocks * numberOfIndirectOpDats/))
CALL c_f_pointer(actualPlan_res_calc%blkmap,blkmap_res_calc,(/actualPlan_res_calc%nblocks/))
CALL c_f_pointer(actualPlan_res_calc%offset,offset_res_calc,(/actualPlan_res_calc%nblocks/))
CALL c_f_pointer(actualPlan_res_calc%nelems,nelems_res_calc,(/actualPlan_res_calc%nblocks/))
CALL c_f_pointer(actualPlan_res_calc%nthrcol,nthrcol_res_calc,(/actualPlan_res_calc%nblocks/))
CALL c_f_pointer(actualPlan_res_calc%thrcol,thrcol_res_calc,(/opSetCore%size/))
CALL c_f_pointer(ind_maps_res_calc(1),ind_maps1_res_calc,(/pnindirect_res_calc(1)/))
CALL c_f_pointer(ind_maps_res_calc(2),ind_maps3_res_calc,(/pnindirect_res_calc(2)/))
CALL c_f_pointer(ind_maps_res_calc(3),ind_maps5_res_calc,(/pnindirect_res_calc(3)/))
CALL c_f_pointer(ind_maps_res_calc(4),ind_maps7_res_calc,(/pnindirect_res_calc(4)/))
IF (indirectionDescriptorArray(1) >= 0) THEN
CALL c_f_pointer(mappingArray_res_calc(1),mappingArray1_res_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(2) >= 0) THEN
CALL c_f_pointer(mappingArray_res_calc(2),mappingArray2_res_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(3) >= 0) THEN
CALL c_f_pointer(mappingArray_res_calc(3),mappingArray3_res_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(4) >= 0) THEN
CALL c_f_pointer(mappingArray_res_calc(4),mappingArray4_res_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(5) >= 0) THEN
CALL c_f_pointer(mappingArray_res_calc(5),mappingArray5_res_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(6) >= 0) THEN
CALL c_f_pointer(mappingArray_res_calc(6),mappingArray6_res_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(7) >= 0) THEN
CALL c_f_pointer(mappingArray_res_calc(7),mappingArray7_res_calc,(/opSetCore%size/))
END IF
IF (indirectionDescriptorArray(8) >= 0) THEN
CALL c_f_pointer(mappingArray_res_calc(8),mappingArray8_res_calc,(/opSetCore%size/))
END IF
END IF
blockOffset = 0
DO i1 = 0, actualPlan_res_calc%ncolors - 1, 1
nblocks = ncolblk_res_calc(i1 + 1)
!$OMP PARALLEL DO 
DO i2 = 0, nblocks - 1, 1
CALL res_calc_kernel(opDat1Local,opDat3Local,opDat5Local,opDat7Local,ind_maps1_res_calc,ind_maps3_res_calc,ind_maps5_res_calc,ind_maps7_res_calc,mappingArray1_res_calc,mappingArray2_res_calc,mappingArray3_res_calc,mappingArray4_res_calc,mappingArray5_res_calc,mappingArray6_res_calc,mappingArray7_res_calc,mappingArray8_res_calc,ind_sizes_res_calc,ind_offs_res_calc,blkmap_res_calc,offset_res_calc,nelems_res_calc,nthrcol_res_calc,thrcol_res_calc,blockOffset,i2)
END DO
!$OMP END PARALLEL DO
blockOffset = blockOffset + nblocks
END DO
END SUBROUTINE 

SUBROUTINE save_soln_modified(q,qold)
  IMPLICIT NONE
  REAL(kind=8), DIMENSION(4) :: q
  REAL(kind=8), DIMENSION(4) :: qold
  INTEGER(kind=4) :: i
  ! call levelTwo (qold
  DO i = 1, 4
  qold(i) = q(i)
  END DO
END SUBROUTINE 

SUBROUTINE save_soln_kernel(opDat1,opDat2,sliceStart,sliceEnd)
IMPLICIT NONE
REAL(kind=8), DIMENSION(0:*) :: opDat1
REAL(kind=8), DIMENSION(0:*) :: opDat2
INTEGER(kind=4) :: sliceStart
INTEGER(kind=4) :: sliceEnd
INTEGER(kind=4) :: i1, diff
integer(8) :: time1, time2, count_rate, count_max
real(kind=8), allocatable, dimension(:) :: topDat1, topDat2
diff = sliceEnd - sliceStart
call system_clock(time1, count_rate, count_max)
allocate (topDat1(diff*4))
allocate (topDat2(diff*4))
call mkl_domatcopy('r','t',diff,4,1.d0,opDat1(4*sliceStart:4*sliceEnd-1),4,topDat1,diff)
call mkl_domatcopy('r','t',diff,4,1.d0,opDat2(4*sliceStart:4*sliceEnd-1),4,topDat2,diff)
call system_clock(time2, count_rate, count_max)
print *, "### not counted", time2 - time1
call system_clock(time1, count_rate, count_max)
CALL save_soln_caller(topDat1, topDat2, diff)
call system_clock(time2, count_rate, count_max)
print *, "### save_soln", time2 - time1
call system_clock(time1, count_rate, count_max)
call mkl_domatcopy('r','t',4,diff,1.d0,topDat1,diff,opDat1(4*sliceStart:4*sliceEnd-1),4)
call mkl_domatcopy('r','t',4,diff,1.d0,topDat2,diff,opDat2(4*sliceStart:4*sliceEnd-1),4)
deallocate (topDat1)
deallocate (topDat2)
call system_clock(time2, count_rate, count_max)
print *, "### not counted", time2 - time1
END SUBROUTINE 

subroutine save_soln_caller(q, qold, diff)
  implicit none
  integer :: diff
  real(kind=8), dimension(BS,diff/BS,4) :: qold, q
  integer :: i, j
  do j = 1, diff/BS
    do i = 1, 4
      !DIR$ SIMD
      qold(:,j,i) = q(:,j,i)
    end do
  end do
end subroutine

SUBROUTINE save_soln_host(userSubroutine,set,opDat1,opIndirection1,opMap1,opAccess1,opDat2,opIndirection2,opMap2,opAccess2)
IMPLICIT NONE
character(len=10), INTENT(IN) :: userSubroutine
TYPE ( op_set ) , INTENT(IN) :: set
TYPE ( op_dat ) , INTENT(IN) :: opDat1
INTEGER(kind=4), INTENT(IN) :: opIndirection1
TYPE ( op_map ) , INTENT(IN) :: opMap1
INTEGER(kind=4), INTENT(IN) :: opAccess1
TYPE ( op_dat ) , INTENT(IN) :: opDat2
INTEGER(kind=4), INTENT(IN) :: opIndirection2
TYPE ( op_map ) , INTENT(IN) :: opMap2
INTEGER(kind=4), INTENT(IN) :: opAccess2
TYPE ( op_set_core ) , POINTER :: opSetCore
TYPE ( op_dat_core ) , POINTER :: opDat1Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat1Local
INTEGER(kind=4) :: opDat1Cardinality
TYPE ( op_set_core ) , POINTER :: opSet1Core
TYPE ( op_dat_core ) , POINTER :: opDat2Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat2Local
INTEGER(kind=4) :: opDat2Cardinality
TYPE ( op_set_core ) , POINTER :: opSet2Core
INTEGER(kind=4) :: i1
INTEGER(kind=4) :: numberOfThreads
INTEGER(kind=4) :: sliceStart
INTEGER(kind=4) :: sliceEnd
IF (set%setPtr%size .EQ. 0) THEN
RETURN
END IF
#ifdef _OPENMP 
numberOfThreads = omp_get_max_threads()
#else 
numberOfThreads = 1
#endif 
opSetCore => set%setPtr
opDat1Core => opDat1%dataPtr
opDat2Core => opDat2%dataPtr
CALL c_f_pointer(opDat1Core%set,opSet1Core)
CALL c_f_pointer(opDat2Core%set,opSet2Core)
opDat1Cardinality = opDat1Core%dim * opSet1Core%size
opDat2Cardinality = opDat2Core%dim * opSet2Core%size
CALL c_f_pointer(opDat1Core%dat,opDat1Local,(/opDat1Cardinality/))
CALL c_f_pointer(opDat2Core%dat,opDat2Local,(/opDat2Cardinality/))
!$OMP PARALLEL DO private (sliceStart,sliceEnd,i1)
DO i1 = 0, numberOfThreads - 1, 1
sliceStart = opSetCore%size * i1 / numberOfThreads
sliceEnd = opSetCore%size * (i1 + 1) / numberOfThreads
CALL save_soln_kernel(opDat1Local,opDat2Local,sliceStart,sliceEnd)
END DO
!$OMP END PARALLEL DO
END SUBROUTINE 

SUBROUTINE update_modified(qold,q,res,adt,rms)
IMPLICIT NONE
  real(kind=8),dimension(BS,4) :: qold
  real(kind=8),dimension(BS,4) :: q
  real(kind=8),dimension(BS,4) :: res
  real(kind=8),dimension(BS) :: adt
  real(kind=8), dimension(1) :: rms
  real(kind=8),dimension(BS) :: adti
  real(kind=8),dimension(BS) :: del
  integer(kind=4) :: i
  adti(:) = 1.0 / adt(:)
  do i = 1, 4
    del(:) = adti(:) * res(:,i)
    q(:,i) = qold(:,i) - del(:)
    res(:,i) = 0.0
    !rms(1) = rms(1) + sum(del(:) * del(:))
  end do
END SUBROUTINE 

SUBROUTINE update_kernel(opDat1,opDat2,opDat3,opDat4,opDat5,sliceStart,sliceEnd)
IMPLICIT NONE
REAL(kind=8), DIMENSION(0:*) :: opDat1
REAL(kind=8), DIMENSION(0:*) :: opDat2
REAL(kind=8), DIMENSION(0:*) :: opDat3
REAL(kind=8), DIMENSION(0:*) :: opDat4
REAL(kind=8), DIMENSION(0:*) :: opDat5
INTEGER(kind=4) :: sliceStart
INTEGER(kind=4) :: sliceEnd
INTEGER(kind=4) :: i1, diff
integer(8) :: time1, time2, count_rate, count_max
real(kind=8), allocatable, dimension(:) :: topDat1, topDat2, topDat3
diff = sliceEnd - sliceStart
call system_clock(time1, count_rate, count_max)
allocate (topDat1(diff*4))
allocate (topDat2(diff*4))
allocate (topDat3(diff*4))
call mkl_domatcopy('r', 't', diff, 4, 1.d0, opDat1(4*sliceStart:4*sliceEnd-1), 4, topDat1, diff)
call mkl_domatcopy('r', 't', diff, 4, 1.d0, opDat2(4*sliceStart:4*sliceEnd-1), 4, topDat2, diff)
call mkl_domatcopy('r', 't', diff, 4, 1.d0, opDat3(4*sliceStart:4*sliceEnd-1), 4, topDat3, diff)
call system_clock(time2, count_rate, count_max)
print *, "### not counted", time2 - time1
call system_clock(time1, count_rate, count_max)
call update_kernel_caller(topDat1, topDat2, topDat3, opDat4(sliceStart:sliceStart+diff-1), opDat5, diff)
call system_clock(time2, count_rate, count_max)
print *,"### update" ,time2 - time1
call system_clock(time1, count_rate, count_max)
call mkl_domatcopy('r', 't', 4, diff, 1.d0, topDat1, diff, opDat1(4*sliceStart:4*sliceEnd-1), 4)
call mkl_domatcopy('r', 't', 4, diff, 1.d0, topDat2, diff, opDat2(4*sliceStart:4*sliceEnd-1), 4)
call mkl_domatcopy('r', 't', 4, diff, 1.d0, topDat3, diff, opDat3(4*sliceStart:4*sliceEnd-1), 4)
deallocate (topDat1)
deallocate (topDat2)
deallocate (topDat3)
call system_clock(time2, count_rate, count_max)
print *, "### not counted", time2 - time1
END SUBROUTINE

subroutine update_kernel_caller(qold, q, res, adt, rms, diff)
  implicit none
  integer                               :: diff
  real(kind=8), dimension(BS,diff/BS,4) :: qold, q, res
  real(kind=8), dimension(BS,*)         :: adt
  real(kind=8), dimension(1)            :: rms
  real(kind=8), dimension(BS)            :: adti, del
  real(kind=8), dimension(BS) :: acc
  integer :: i,j
  do j=1,diff/BS
    adti(:) = 1.0 / adt(:,j)
    do i=1,4
      del(:) = adti(:) * res(:,j,i)
      !DIR$ SIMD
      q(:,j,i) = qold(:,j,i) - del(:)
      !DIR$ SIMD
      res(:,j,i) = 0.0
      acc(:) = del(:) * del(:) 
      rms(1) = rms(1) + sum(acc)
    end do
  end do
end subroutine

SUBROUTINE update_host(userSubroutine,set,opDat1,opIndirection1,opMap1,opAccess1,opDat2,opIndirection2,opMap2,opAccess2,opDat3,opIndirection3,opMap3,opAccess3,opDat4,opIndirection4,opMap4,opAccess4,opDat5,opIndirection5,opMap5,opAccess5)
IMPLICIT NONE
character(len=7), INTENT(IN) :: userSubroutine
TYPE ( op_set ) , INTENT(IN) :: set
TYPE ( op_dat ) , INTENT(IN) :: opDat1
INTEGER(kind=4), INTENT(IN) :: opIndirection1
TYPE ( op_map ) , INTENT(IN) :: opMap1
INTEGER(kind=4), INTENT(IN) :: opAccess1
TYPE ( op_dat ) , INTENT(IN) :: opDat2
INTEGER(kind=4), INTENT(IN) :: opIndirection2
TYPE ( op_map ) , INTENT(IN) :: opMap2
INTEGER(kind=4), INTENT(IN) :: opAccess2
TYPE ( op_dat ) , INTENT(IN) :: opDat3
INTEGER(kind=4), INTENT(IN) :: opIndirection3
TYPE ( op_map ) , INTENT(IN) :: opMap3
INTEGER(kind=4), INTENT(IN) :: opAccess3
TYPE ( op_dat ) , INTENT(IN) :: opDat4
INTEGER(kind=4), INTENT(IN) :: opIndirection4
TYPE ( op_map ) , INTENT(IN) :: opMap4
INTEGER(kind=4), INTENT(IN) :: opAccess4
TYPE ( op_dat ) , INTENT(IN) :: opDat5
INTEGER(kind=4), INTENT(IN) :: opIndirection5
TYPE ( op_map ) , INTENT(IN) :: opMap5
INTEGER(kind=4), INTENT(IN) :: opAccess5
TYPE ( op_set_core ) , POINTER :: opSetCore
TYPE ( op_dat_core ) , POINTER :: opDat1Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat1Local
INTEGER(kind=4) :: opDat1Cardinality
TYPE ( op_set_core ) , POINTER :: opSet1Core
TYPE ( op_dat_core ) , POINTER :: opDat2Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat2Local
INTEGER(kind=4) :: opDat2Cardinality
TYPE ( op_set_core ) , POINTER :: opSet2Core
TYPE ( op_dat_core ) , POINTER :: opDat3Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat3Local
INTEGER(kind=4) :: opDat3Cardinality
TYPE ( op_set_core ) , POINTER :: opSet3Core
TYPE ( op_dat_core ) , POINTER :: opDat4Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat4Local
INTEGER(kind=4) :: opDat4Cardinality
TYPE ( op_set_core ) , POINTER :: opSet4Core
TYPE ( op_dat_core ) , POINTER :: opDat5Core
REAL(kind=8), POINTER, DIMENSION(:) :: opDat5Local
INTEGER(kind=4) :: opDat5Cardinality
INTEGER(kind=4) :: i1
INTEGER(kind=4) :: numberOfThreads
INTEGER(kind=4) :: sliceStart
INTEGER(kind=4) :: sliceEnd
INTEGER(kind=4) :: i10
INTEGER(kind=4) :: i11
REAL(kind=8), DIMENSION(0:1 + 64 * 64 - 1) :: reductionArrayHost5
IF (set%setPtr%size .EQ. 0) THEN
RETURN
END IF
#ifdef _OPENMP 
numberOfThreads = omp_get_max_threads()
#else 
numberOfThreads = 1
#endif 
opSetCore => set%setPtr
opDat1Core => opDat1%dataPtr
opDat2Core => opDat2%dataPtr
opDat3Core => opDat3%dataPtr
opDat4Core => opDat4%dataPtr
opDat5Core => opDat5%dataPtr
CALL c_f_pointer(opDat1Core%set,opSet1Core)
CALL c_f_pointer(opDat2Core%set,opSet2Core)
CALL c_f_pointer(opDat3Core%set,opSet3Core)
CALL c_f_pointer(opDat4Core%set,opSet4Core)
opDat1Cardinality = opDat1Core%dim * opSet1Core%size
opDat2Cardinality = opDat2Core%dim * opSet2Core%size
opDat3Cardinality = opDat3Core%dim * opSet3Core%size
opDat4Cardinality = opDat4Core%dim * opSet4Core%size
opDat5Cardinality = opDat5Core%dim
CALL c_f_pointer(opDat1Core%dat,opDat1Local,(/opDat1Cardinality/))
CALL c_f_pointer(opDat2Core%dat,opDat2Local,(/opDat2Cardinality/))
CALL c_f_pointer(opDat3Core%dat,opDat3Local,(/opDat3Cardinality/))
CALL c_f_pointer(opDat4Core%dat,opDat4Local,(/opDat4Cardinality/))
CALL c_f_pointer(opDat5Core%dat,opDat5Local,(/opDat5Cardinality/))
DO i10 = 0, numberOfThreads - 1, 1
DO i11 = 0, 1 - 1, 1
reductionArrayHost5(i11 + i10 * 64) = 0
END DO
END DO
!$OMP PARALLEL DO private (sliceStart,sliceEnd,i1)
DO i1 = 0, numberOfThreads - 1, 1
sliceStart = opSetCore%size * i1 / numberOfThreads
sliceEnd = opSetCore%size * (i1 + 1) / numberOfThreads
CALL update_kernel(opDat1Local,opDat2Local,opDat3Local,opDat4Local,reductionArrayHost5(i1 * 64:),sliceStart,sliceEnd)
END DO
!$OMP END PARALLEL DO
DO i10 = 0, numberOfThreads - 1, 1
DO i11 = 0, 1 - 1, 1
opDat5Local(1 + i11) = reductionArrayHost5(i11 + i10 * 64) + opDat5Local(1 + i11)
END DO
END DO
END SUBROUTINE 

END MODULE GENERATED_MODULE

