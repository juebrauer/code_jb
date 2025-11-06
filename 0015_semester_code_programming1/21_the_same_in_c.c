#include <stdio.h>

int main()
{
    for (int i=1; i<=5; i++)
    {
    printf("%d\n", i);
    printf("%d\n", i*i);
    printf("-----\n");
    }

    float result;
    result = 5 / 2; // Integer-Division!
    //result = 5.0 / 2; // Floating-Point Division!
    //result = 5 / 2.0; // Floating-Point Division!
    //result = (float)5 / 2;
    printf("result=%f\n",result);


}
