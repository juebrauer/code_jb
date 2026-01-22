#include <iostream>

class A
{
    public:
    A()
    {
        std::cout << "Reserviere 5MB Speicher" << std::endl;
        data = (int*) malloc(1024*1024*5);
    }

    ~A()
    {
        std::cout << "Gebe Speicher wieder frei..." << std::endl;
        free(data);
    }

    private:
    int* data;

};

int main()
{
    /*
    A* a1 = new A();
    delete a1;
    */

    A a1;
}