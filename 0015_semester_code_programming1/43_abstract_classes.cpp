#include <iostream>

using namespace std;

class A
{
    public:

    A()
    {    
        cout << "Constructor of A called" << endl;    
    }

    virtual void f1() = 0;

    virtual void f2() = 0;
    

};

class B : public A
{
    public:

    B()
    {
        cout << "Constructor of B called" << endl;   
    }

    void f1() override
    {

    }

    void f2() override
    {

    }

    
};

int main()
{
    B b1;    
}