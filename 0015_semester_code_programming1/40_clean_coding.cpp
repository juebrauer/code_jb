#include <iostream>

int main()
{
    int* ptr = nullptr;
    if (ptr=nullptr)
    {
        std::cout << "Ich bin hier drinnen!" << std::endl;
    }

    int a;
    a = 1;
    if (a=42)
    {
        std::cout << "Der Fall, der mich interessiert" << std::endl;
    }

    int b;
    int c;
    a = b = c = 42;
    std::cout << a << "," << b << "," << c << std::endl;
}