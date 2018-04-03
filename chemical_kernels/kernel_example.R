library("kernlab")
library("readr")
library("Rchemcpp")
library("MASS")


input_path = "~/Documents/research/fall2017/data/cl_smi.csv"
length = 4
output_path = 'Documents/research/fall2017/data/spectrum_4.txt'

spectrum = function(input_path, length, output_path){
  cl = read_csv(input_path, col_names = FALSE)
  N = nrow(cl)
  sk = stringdot(type="string", length=length)
  result = array(rep(0, N*N), dim=c(N,N))

  for (i in 1:N){
    for (j in 1:N){
      result[i,j] = kernelMatrix(sk, cl[i,1], cl[j,1])
    }
  }
  write.matrix(result, output_path)
}

# random walk kernel
input_path = 'Documents/research/fall2017/data/cl.sdf'
output_path = 'Documents/research/fall2017/data/cl_random_walk0.1.txt'
stopP = 0.1
mat = sd2gram(input_path,stopP=stopP)
write.matrix(mat,output_path)

# subtree kernel
input_path = 'Documents/research/fall2017/data/cl.sdf'
output_path = 'Documents/research/fall2017/data/cl_subtree_3_0.01.txt'
depth = 3
lambda = 0.01
mat = sd2gramSubtree(input_path, depthMax = depth, lambda = lambda)
write.matrix(mat,output_path)


