

module DebugInterface

  use ISO_C_BINDING

  interface

     ! debug C functions (to obtain similar output file that can be diff-ed
     integer(KIND=C_INT) function openfile ( filename ) BIND(C)

       use, intrinsic :: ISO_C_BINDING
       character(kind=c_char), dimension(1) :: filename

     end function openfile

     integer(KIND=C_INT) function closefile ( ) BIND(C)

       use, intrinsic :: ISO_C_BINDING

     end function closefile


     integer(KIND=C_INT) function writerealtofile ( dataw ) BIND(C)

       use, intrinsic :: ISO_C_BINDING

       real(c_double) :: dataw


     end function writerealtofile

     integer(KIND=C_INT) function writeinttofile ( dataw ) BIND(C)

       use, intrinsic :: ISO_C_BINDING

       integer(c_int) :: dataw

     end function writeinttofile

  end interface

end module DebugInterface
