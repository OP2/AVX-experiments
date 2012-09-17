Playing around with the source code to see in what way the code needs to be organized in order for the intel compiler to vectorize it in the way we want.

The file of interest is rose_openmp_vector.f90 located in the generated_sources file.

Within the results directory, there is a times directory which contains the execution times of the various different sources.
original.txt is the non-vectorized times
newvector.txt is the vectorized times which gather, calculate and scatter with the minimum vector size
vector.txt is the vectorized times which gather everything, calculate everything and scatter everything all at once.

The other files in the results directory are just files containing the execution results of the various runs, this is used to see whether the modifications still yield the correct results.
