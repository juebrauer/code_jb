#include <stdio.h>

int a;


int f1()
{
    a = 5;
    printf("f1: a=%d\n", a);
}

int main()
{
    int a;
    
    a = 2;

    f1();

    printf("main: a=%d\n", a);
}