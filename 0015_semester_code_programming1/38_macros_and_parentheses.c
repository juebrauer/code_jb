#include <stdio.h>

int average(int a, int b)
{
    return (a+b)/2;
}

float average_float(float a, float b)
{
    return (a+b)/2;
}

double average_double(double a, double b)
{
    return (a+b)/2;
}

int main() {
   int i1 = 2;
   int i2 = 5;
   int im;
   im = average(i1,i2);
   printf("Average of %d and %d is %d\n", i1, i2, im);

   int f1 = 2.345;
   float f2 = 5.678;
   float fm;
   fm = average_float(f1,f2);
   printf("Average of %.17f and %.2f is %.2f\n", (float)5000, f2, fm);

   double d1 = 2.382932;
   double d2 = 5.433;
   double dm;
   dm = average_double(d1,d2);
   printf("Average of %.2f and %.2f is %.2f\n", d1, d2, dm);
}