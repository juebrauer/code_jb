#include <stdio.h>

int main() {

    double x = 1.0;

    for (int i=1; i<300; i++)
    {
        x = x / 10;
        printf("%03d %.300f\n", i, x);
    }
}