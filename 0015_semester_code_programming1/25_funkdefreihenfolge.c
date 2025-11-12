#include <stdio.h>

void f1(); // Forward declaration

void f2()
{
    f1();
    printf("Tschuess!\n");
}

void f1()
{
    printf("Hallo\n");
}


int main()
{
    f2();
}