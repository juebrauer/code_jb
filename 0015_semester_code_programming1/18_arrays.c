#include <stdio.h>

int main()
{
    char A[10];
    for (int i=0; i<10; i++)
        A[i] = i*i;

    for (int i=0; i<10; i++)
    {
        printf("A[%d]=%d - Speicheradresse: %p\n", i, A[i], &(A[i]));
    }
}