nvcc common.cu matrix.cu vector.cu -o libmt.dll -O3 --shared -Xcompiler=/wd4819
copy libmt.dll ..\..\go\matrics\example\
