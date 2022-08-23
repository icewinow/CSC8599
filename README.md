# CSC8599
W,A,S,D ------------ Move camera      
Z,X ---------------- Up/down camera    
0 ------------------ display mouse cunsor    
1 ------------------ disabled mouse cunsor    

cmake:
mkdir -p out    
cd out    
cmake -G"Visual Studio 16 2019" ${COMMON_CMAKE_CONFIG_PARAMS} ../    
cmake --build . --config Debug    
