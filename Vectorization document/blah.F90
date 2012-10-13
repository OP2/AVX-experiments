  do i2 = 0, BS - 1
    arg1(i2, :) = opDat1SharedIndirection(1 + mappingArray1(i1 + i2 + threadBlockOffset) * 2 : 1 + mappingArray1(i1 + i2 + threadBlockOffset) * 2 + 1)
    arg2(i2, :) = opDat1SharedIndirection(1 + mappingArray2(i1 + i2 + threadBlockOffset) * 2 : 1 + mappingArray2(i1 + i2 + threadBlockOffset) * 2 + 1)
    arg3(i2, :) = opDat1SharedIndirection(1 + mappingArray3(i1 + i2 + threadBlockOffset) * 2 : 1 + mappingArray3(i1 + i2 + threadBlockOffset) * 2 + 1)
    arg4(i2, :) = opDat1SharedIndirection(1 + mappingArray4(i1 + i2 + threadBlockOffset) * 2 : 1 + mappingArray4(i1 + i2 + threadBlockOffset) * 2 + 1)
    arg5(i2, :) = opDat5(4 * (i1 + i2 + threadBlockOffset) : 4 * (i1 + i2 + threadBlockOffset) + 3)
  end do