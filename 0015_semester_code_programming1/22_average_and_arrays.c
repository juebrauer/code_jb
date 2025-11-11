#include <stdio.h>


float average(float* numbers, int how_many) {
   float sum = 0.0f;
   for (int j = 0; j < how_many; j++)
      sum += *(numbers+j);
   return sum/how_many;
}

int main() {

   float numbers[100];
      
   printf("\nPlease enter up to 100 numbers. Enter -1 to stop\n");   
   int i = 0;
   do {
      printf("Enter number #%d : ", i+1);
      float f;
      scanf("%f", &f);
      if (f==-1.0)
         break;      
      numbers[i] = f;
      i++;
   } while (i<=99);
      
   printf("\nAverage of the %d numbers you entered is : %.2f\n",
          i, average(numbers, i) );
   
} // end main