#include <stdio.h>


void swap(int c, int d)  // Call-by-value
{
    int tmp=c;
    c = d;
    d = tmp;
    printf("In der Funktion swap: c=%d d=%d\n", c,d);
}

void swap2(int* c, int* d)  // Call-by-reference
{
    int tmp=*c;
    *c = *d;
    *d = tmp;
    printf("In der Funktion swap: c=%d d=%d\n", *c,*d);
}

int main()
{
    int a = 17;
    int b = 29;

    printf("davor: a=%d b=%d\n", a,b);

    swap2(&a,&b);

    printf("danach: a=%d b=%d\n", a,b);
}