#include <stdio.h>


#define pi 3.14159265359
#define welcome "Welcome to a demo for #define\n\n"
#define WRITE printf (
#define WRITEEND );


int main()
{
   printf(welcome);

   printf("The constant pi was defined as %f\n", pi);

   WRITE "Test123\n" WRITEEND 
}