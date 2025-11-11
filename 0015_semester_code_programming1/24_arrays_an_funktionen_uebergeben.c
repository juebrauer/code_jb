#include <stdio.h>

void a(int* f)
{
   //f[0] = f[1] + f[2];
   *f = *(f+1) + *(f+2);
}


int main()
{
   int myArray[3];
   myArray[0] = 12;
   myArray[1] = 9;
   myArray[2] = 5;

   a(myArray);

   printf("myArray[0] = %d\n", myArray[0]);


} // end main