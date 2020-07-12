nvcc common.cu tensor.cu vector.cu matrix.cu neural.cu -O3 --shared -Xcompiler=/wd4819 -o libmatrics.dll
copy libmatrics.dll C:\Windows\
