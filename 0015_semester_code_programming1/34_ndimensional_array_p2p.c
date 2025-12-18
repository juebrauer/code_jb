#include <stdio.h>
#include <stdlib.h> // malloc & friends
typedef unsigned int uint;

int main() {
   uint dimX, dimY;
   printf("Size of y dimension? ");
   scanf("%d", &dimY);
   printf("Size of x dimension? ");
   scanf("%d", &dimX);

   printf("Press a key to allocate the 2D array!\n");
   int** myArray2D = malloc(dimY * sizeof(int*));
   for (uint y = 0; y < dimY; y++)
   {
      myArray2D[y] = malloc(dimX * sizeof(int));
      printf("%d - th row is at memory address %p\n", y, myArray2D[y]);
   }

   int counter = 0;
   for (uint y = 0; y < dimY; y++)
      for (uint x = 0; x < dimX; x++)
         myArray2D[y][x] = counter++;

   printf("Press a key to free the 2D array!\n");  
   for (uint y = 0; y < dimY; y++)
      free (myArray2D[y]);
   free(myArray2D);
   printf("That's it! We are finished!\n");
}