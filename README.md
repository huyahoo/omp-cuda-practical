# practical-work-2

## Exercise 2.1.1
### OMP

The program requires two arguments to run correctly. The first argument is the path to the stereo image, and the second argument is the type of anaglyph to generate. The anaglyph type should be an integer between 0 and 4, each representing a different type of anaglyph.

The anaglyph types are as follows:
0: True Anaglyphs
1: Gray Anaglyphs
2: Color Anaglyphs
3: Half Color Anaglyphs
4: Optimized Anaglyphs

Here is the usage:
```bash
./2.1.1-omp <image_path> <anaglyph_type>
```

Example:
```bash
cd Open-OMP
g++ 2.1.1-omp.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ 2.1.1-omp.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o 2.1.1-omp
./2.1.1-omp stereo.jpg 2
```