#include <stdio.h>

int main()
{
    int i = 10;
    do
    {
        printf("Countdown: %d\n", i);
        i--;
    } while (i>=0);

    return 0;
}