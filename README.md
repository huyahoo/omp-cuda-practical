# practical-work-2

## Image Processing by OPEN-OMP
The results will be saved in the folder named as "output".

Move to main directory:
```bash
cd OpenCV-OMP
```

### Exercise 2.1.1

The program requires two arguments to run correctly. The first argument is the path to the stereo image, and the second argument is the type of anaglyph to generate. The anaglyph type should be an integer between 0 and 4, each representing a different type of anaglyph.

The anaglyph types are as follows:
- O: None Anaglyphs (Default)
- 1: True Anaglyphs
- 2: Gray Anaglyphs
- 3: Color Anaglyphs
- 4: Half Color Anaglyphs
- 5: Optimized Anaglyphs

Usage:
```bash
./2.1.1-omp <image_path> <anaglyph_type>
```

Example:
```bash
g++ 2.1.1-omp.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ 2.1.1-omp.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o 2.1.1-omp
./2.1.1-omp stereo.jpg 2
```

### Exercise 2.1.2

- Anaglyph type is 0 as default
- Input kernel size in range odd numbers from 3 to 21
- Input sigma in range odd numbers from 0.1 to 10
  
Usage:
```bash
./2.1.1-omp <image_path> <anaglyph_type> <kernel_size> <sigma>
```

Example:
```bash
g++ 2.1.2-omp.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ 2.1.2-omp.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o 2.1.2-omp
./2.1.2-omp garden-stereo.jpg 0 7 5
```

### Exercise 2.1.3

Usage:
```bash
./2.1.1-omp <image_path> <neighborhood_size> <factor_ratio>
```

Example:
```bash
g++ 2.1.3-omp.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ 2.1.3-omp.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o 2.1.3-omp
./2.1.3-omp noise.png 0 5 1
```

## Image Processing by CUDA
The results will be saved in the folder named as "output".

Move to main directory:
```bash
cd OpenCV-CUDA
```

### Exercise 2.1.1

The program requires two arguments to run correctly. The first argument is the path to the stereo image, and the second argument is the type of anaglyph to generate. The anaglyph type should be an integer between 0 and 4, each representing a different type of anaglyph.

The anaglyph types are as follows:
- O: None Anaglyphs (Default)
- 1: True Anaglyphs
- 2: Gray Anaglyphs
- 3: Color Anaglyphs
- 4: Half Color Anaglyphs
- 5: Optimized Anaglyphs

Usage:
```bash
./2.1.1-cuda <image_path> <anaglyph_type>
```

Example:
```bash
/usr/local/cuda-11.6/bin/nvcc -O3 2.1.1-cuda.cu `pkg-config opencv4 --cflags --libs` -o 2.1.1-cuda
/usr/local/cuda-11.6/bin/nvcc -O3 2.1.1-cuda.cu `pkg-config opencv4 --cflags --libs` -o 2.1.1-cuda
./2.1.1-cuda garden-stereo.jpg 0
```