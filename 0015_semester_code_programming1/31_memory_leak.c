#include <stdio.h>
//#include <stdlib.h> // malloc

int main() {
   while (1) {
      printf("\nHow much MB do you want? ");
      unsigned int mb;
      scanf("%d", &mb);

      if (mb == 0)
         break;

      unsigned int  number_of_bytes_to_reserve = mb * 1024 * 1024;
      unsigned char* A = malloc(number_of_bytes_to_reserve);

      if (A == NULL) {
         printf("Out of memory!\n");
         exit(1);
      }
      else {
         printf("\tReserved %d MB of memory.\n", mb);
         printf("\tMemory starts at address %p\n", A);

         for (char i=0; i<number_of_bytes_to_reserve; i++)
         {
            A[i] = 42;
         }

         int sum = 0; 
         for (char i=0; i<number_of_bytes_to_reserve; i++)
         {
            sum += A[i];
         }

         printf("sum is %d\n", sum);

         printf("Now freeing memory ...");
         //free(A);
      }


   } // end while


}