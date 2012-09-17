# 1 "/data/carlo/FortranTranslator/OP2_ROSE_Fortran/tests/Fortran/airfoil-fused/airfoBtCZME.F90"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/data/carlo/FortranTranslator/OP2_ROSE_Fortran/tests/Fortran/airfoil-fused/airfoBtCZME.F90"
PROGRAM AIRFOIL
USE GENERATED_MODULE
USE OP2_FORTRAN_DECLARATIONS
USE OP2_CONSTANTS
USE AIRFOIL_SEQ
USE ISO_C_BINDING
use DebugInterface
IMPLICIT NONE
intrinsic sqrt, real
INTEGER(kind=4) :: iter,k,i
INTEGER(kind=4), PARAMETER :: maxnode = 9900
INTEGER(kind=4), PARAMETER :: maxcell = 9702 + 1
INTEGER(kind=4), PARAMETER :: maxedge = 19502
INTEGER(kind=4), PARAMETER :: iterationNumber = 1000
INTEGER(kind=4) :: nnode,ncell,nbedge,nedge,niter
REAL(kind=8) :: ncellr
  ! integer references (valid inside the OP2 library) for op_set
TYPE ( op_set )  :: nodes,edges,bedges,cells
  ! integer references (valid inside the OP2 library) for pointers between data sets
TYPE ( op_map )  :: pedge,pecell,pcell,pbedge,pbecell
  ! integer reference (valid inside the OP2 library) for op_data 
TYPE ( op_dat )  :: p_bound,p_x,p_q,p_qold,p_adt,p_res,p_rms
  ! arrays used in data
INTEGER(kind=4), DIMENSION(:), ALLOCATABLE, TARGET :: ecell,bound,edge,bedge,becell,cell
REAL(kind=8), DIMENSION(:), ALLOCATABLE, TARGET :: x,q,qold,adt,res,rms
character(len=10,kind=c_char) :: savesolnName = 'save_soln' // C_NULL_CHAR
character(len=9,kind=c_char) :: adtcalcName = 'adt_calc' // C_NULL_CHAR
character(len=9,kind=c_char) :: rescalcName = 'res_calc' // C_NULL_CHAR
character(len=10,kind=c_char) :: brescalcName = 'bres_calc' // C_NULL_CHAR
character(len=7,kind=c_char) :: updateName = 'update' // C_NULL_CHAR
character(len=6,kind=c_char) :: nodesName = 'nodes' // C_NULL_CHAR
character(len=6,kind=c_char) :: edgesName = 'edges' // C_NULL_CHAR
character(len=7,kind=c_char) :: bedgesName = 'bedges' // C_NULL_CHAR
character(len=6,kind=c_char) :: cellsName = 'cells' // C_NULL_CHAR
character(len=6,kind=c_char) :: pedgeName = 'pedge' // C_NULL_CHAR
character(len=7,kind=c_char) :: pecellName = 'pecell' // C_NULL_CHAR
character(len=6,kind=c_char) :: pcellName = 'pcell' // C_NULL_CHAR
character(len=7,kind=c_char) :: pbedgeName = 'pbedge' // C_NULL_CHAR
character(len=8,kind=c_char) :: pbecellName = 'pbecell' // C_NULL_CHAR
character(len=6,kind=c_char) :: boundName = 'bound' // C_NULL_CHAR
character(len=2,kind=c_char) :: xName = 'x' // C_NULL_CHAR
character(len=2,kind=c_char) :: qName = 'q' // C_NULL_CHAR
character(len=5,kind=c_char) :: qoldName = 'qold' // C_NULL_CHAR
character(len=4,kind=c_char) :: adtName = 'adt' // C_NULL_CHAR
character(len=4,kind=c_char) :: resName = 'res' // C_NULL_CHAR
INTEGER(kind=4) :: debugiter,retDebug
REAL(kind=8) :: datad

real(8) start_time, finish_time, totalExecutionTime
integer time_array_0(8), time_array_1(8)
integer(8) :: time1, time2, count_rate, count_max
  ! read set sizes from input file (input is subdivided in two routines as we cannot allocate arrays in subroutines in
  ! fortran 90)
PRINT *, "Getting set sizes"
CALL getSetSizes(nnode,ncell,nedge,nbedge)
PRINT *, ncell
  ! allocate sets (cannot allocate in subroutine in F90)
allocate( cell(4 * ncell) )
allocate( edge(2 * nedge) )
allocate( ecell(2 * nedge) )
allocate( bedge(2 * nbedge) )
allocate( becell(nbedge) )
allocate( bound(nbedge) )
allocate( x(2 * nnode) )
allocate( q(4 * ncell) )
allocate( qold(4 * ncell) )
allocate( res(4 * ncell) )
allocate( adt(ncell) )
allocate( rms(1) )
PRINT *, "Getting data"
CALL getSetInfo(nnode,ncell,nedge,nbedge,cell,edge,ecell,bedge,becell,bound,x,q,qold,res,adt)
PRINT *, "Initialising constants"
CALL initialise_flow_field(ncell,q,res)
DO iter = 1, 4 * ncell
res(iter) = 0.0
END DO
  ! OP initialisation
PRINT *, "Initialising OP2"
CALL op_init(7)
  ! declare sets, pointers, datasets and global constants (for now, no new partition info)
PRINT *, "Declaring OP2 sets"
CALL op_decl_set(nnode,nodes,nodesName)
CALL op_decl_set(nedge,edges,edgesName)
CALL op_decl_set(nbedge,bedges,bedgesName)
CALL op_decl_set(ncell,cells,cellsName)
PRINT *, "Declaring OP2 maps"
CALL op_decl_map(edges,nodes,2,edge,pedge,pedgeName)
CALL op_decl_map(edges,cells,2,ecell,pecell,pecellName)
CALL op_decl_map(bedges,nodes,2,bedge,pbedge,pbedgeName)
CALL op_decl_map(bedges,cells,1,becell,pbecell,pecellName)
CALL op_decl_map(cells,nodes,4,cell,pcell,pcellName)
PRINT *, "Declaring OP2 data"
CALL op_decl_dat(bedges,1,bound,p_bound,boundName)
CALL op_decl_dat(nodes,2,x,p_x,xName)
CALL op_decl_dat(cells,4,q,p_q,qName)
CALL op_decl_dat(cells,4,qold,p_qold,qoldName)
CALL op_decl_dat(cells,1,adt,p_adt,adtName)
CALL op_decl_dat(cells,4,res,p_res,resName)
PRINT *, "Declaring OP2 globals"
CALL op_decl_gbl(rms,p_rms,1)
CALL op_decl_const(gam,1)
CALL op_decl_const(gm1,1)
CALL op_decl_const(cfl,1)
CALL op_decl_const(eps,1)
CALL op_decl_const(mach,1)
CALL op_decl_const(alpha,1)
CALL op_decl_const(qinf,4)

call initOP2Constants(alpha,cfl,eps,gam,gm1,mach,qinf)

call date_and_time(values=time_array_0)
start_time = time_array_0 (5) * 3600 + time_array_0 (6) * 60 &
           + time_array_0 (7) + 0.001 * time_array_0 (8)
call system_clock(time1, count_rate, count_max)
    ! main time-marching loop CARLO!!!
DO niter = 1, iterationNumber

!print *, 'save_soln'

CALL save_soln_host("save_soln" // CHAR(0),cells,p_q,-1,OP_ID,OP_READ,p_qold,-1,OP_ID,OP_WRITE)
    ! predictor/corrector update loop

!print *, 'adt_calc'


      ! calculate area/timstep
CALL adt_calc_host("adt_calc" // CHAR(0),cells,p_x,1,pcell,OP_READ,p_x,2,pcell,OP_READ,p_x,3,pcell,OP_READ,p_x,4,pcell,OP_READ,p_q,-1,OP_ID,OP_READ,p_adt,-1,OP_ID,OP_WRITE)


!print *, 'res_calc'

      ! calculate flux residual
CALL res_calc_host("res_calc" // CHAR(0),edges,p_x,1,pedge,OP_READ,p_x,2,pedge,OP_READ,p_q,1,pecell,OP_READ,p_q,2,pecell,OP_READ,p_adt,1,pecell,OP_READ,p_adt,2,pecell,OP_READ,p_res,1,pecell,OP_INC,p_res,2,pecell,OP_INC)


CALL bres_calc_host("bres_calc" // CHAR(0),bedges,p_x,1,pbedge,OP_READ,p_x,2,pbedge,OP_READ,p_q,1,pbecell,OP_READ,p_adt,1,pbecell,OP_READ,p_res,1,pbecell,OP_INC,p_bound,-1,OP_ID,OP_READ)
      ! update flow field
rms(1) = 0.0


CALL update_host("update" // CHAR(0),cells,p_qold,-1,OP_ID,OP_READ,p_q,-1,OP_ID,OP_WRITE,p_res,-1,OP_ID,OP_RW,p_adt,-1,OP_ID,OP_READ,p_rms,-1,OP_GBL,OP_INC)


CALL adt_calc_host("adt_calc" // CHAR(0),cells,p_x,1,pcell,OP_READ,p_x,2,pcell,OP_READ,p_x,3,pcell,OP_READ,p_x,4,pcell,OP_READ,p_q,-1,OP_ID,OP_READ,p_adt,-1,OP_ID,OP_WRITE)

      ! calculate flux residual
CALL res_calc_host("res_calc" // CHAR(0),edges,p_x,1,pedge,OP_READ,p_x,2,pedge,OP_READ,p_q,1,pecell,OP_READ,p_q,2,pecell,OP_READ,p_adt,1,pecell,OP_READ,p_adt,2,pecell,OP_READ,p_res,1,pecell,OP_INC,p_res,2,pecell,OP_INC)

CALL bres_calc_host("bres_calc" // CHAR(0),bedges,p_x,1,pbedge,OP_READ,p_x,2,pbedge,OP_READ,p_q,1,pbecell,OP_READ,p_adt,1,pbecell,OP_READ,p_res,1,pbecell,OP_INC,p_bound,-1,OP_ID,OP_READ)


      ! update flow field
rms(1) = 0.0


CALL update_host("update" // CHAR(0),cells,p_qold,-1,OP_ID,OP_READ,p_q,-1,OP_ID,OP_WRITE,p_res,-1,OP_ID,OP_RW,p_adt,-1,OP_ID,OP_READ,p_rms,-1,OP_GBL,OP_INC)

!!     call op_get_dat ( p_q )
!      retDebug = openfile ( C_CHAR_"q_avx.txt"//C_NULL_CHAR )
!      do debugiter = 1, 4*ncell
!              datad = q(debugiter)
!              retDebug = writeRealToFile ( datad )
!      end do
!      retDebug = closefile ()
!
!stop

ncellr = real(ncell)
rms(1) = sqrt(rms(1) / ncellr)
IF (mod(niter,100) .EQ. 0) PRINT *, "=====> Iteration result ",rms(1)
END DO !CARLO!!!
call system_clock(time2, count_rate, count_max)
print *, "### ENTIRE TIME", time2 - time1
      call date_and_time(values=time_array_1)
      finish_time = time_array_1 (5) * 3600 + time_array_1 (6) * 60 &
           + time_array_1 (7) + 0.001 * time_array_1 (8)
totalExecutionTime = finish_time - start_time

!retDebug = openfile ( C_CHAR_"/work/cbertoll/AirfoilFortran/OpenMP/airfoil-mesh-indip-fusion/time_nofusion.txt"//C_NULL_CHAR )
!   datad = q(debugiter)
!   retDebug = writeRealToFile ( totalExecutionTime )
! end do
! retDebug = closefile ()
 write(*,*) 'Time total execution (ms): ', totalExecutionTime
print *, 'Finished..'

! uncomment the following statements to get the result of the airfoil written in a file
! modify the path as convenient
!       call op_get_dat ( p_q )
!        retDebug = openfile ( C_CHAR_"/work/cbertoll/AirfoilFortran/CUDA/FuseTests/q.txt"//C_NULL_CHAR )
!        do debugiter = 1, 4*ncell
!                datad = q(debugiter)
!                retDebug = writeRealToFile ( datad )
!        end do
!        retDebug = closefile ()
END PROGRAM AIRFOIL

