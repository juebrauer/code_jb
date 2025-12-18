#include <stdio.h>
#include <stdlib.h> // malloc & friends

#define MAP_3D_TO_1D_INDEX(dimX,dimY,x,y,z) (z*(dimY*dimX)+y*dimX+x)
typedef unsigned int uint;

int main() {
   uint dimX, dimY, dimZ;
   printf("Size of Z-dimension? ");   scanf("%d", &dimZ);
   printf("Size of Y-dimension? ");   scanf("%d", &dimY);
   printf("Size of X-dimension? ");   scanf("%d", &dimX);

   int* A = malloc(dimZ*dimY*dimX * sizeof(int));

   int counter = 0;
   for (uint z = 0; z < dimZ; z++)
      for (uint y = 0; y < dimY; y++)
         for (uint x = 0; x < dimX; x++)
         {
            uint index = z*(dimY*dimX) + y*dimX + x;
            A[index] = counter++;
         }

   printf("\n");
   for (uint z = 0; z < dimZ; z++) {
      printf("\n\nz=%d:\n", z);
      for (uint y = 0; y < dimY; y++) {
         for (uint x = 0; x < dimX; x++)
            printf("%03d ", A[MAP_3D_TO_1D_INDEX(dimX, dimY, x, y, z)]);
         printf("\n");
      }
   }
   free(A);
}