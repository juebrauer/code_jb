#include <iostream>


class Figure
{
    public:

        Figure(float _x, float _y, std::string _name) : x(_x), y(_y),
           energy(1.0), name(_name), dx(1.0), dy(1.0)
        {

        }

        void update()
        {
            this->x += this->dx;
            this->y += this->dy;
        }
        
        float x;
        float y;
        float dx;
        float dy;
        float energy;
        std::string name;
};


class Knight : public Figure
{   
    public:
        Knight(float x, float y, std::string name, bool horse) : Figure(x,y,name)
        {
            this->horse = horse;
        }

        Knight(float x, float y, bool horse) : Figure(x,y,"Bernd")
        {
            this->horse = horse;
        }

        void update()
        {
            if (this->horse)
            {
                this->x += 2 * this->dx;
                this->y += 2 * this->dy;
            }
            else
            {
                this->x += 0.5 * this->dx;
                this->y += 0.5 * this->dy;
            }
            
        } 

    bool horse;
};


class Archer : public Figure
{
    public:
        Archer(float x, float y, std::string name) : Figure(x,y, name)
        {

        }

      
};


int main()
{
    Knight k1(10,10, "Aragon", true);
    Knight k2(10,10, false);

    std::cout << "Knight " << k1.name << " is at (" << k1.x << "," << k1.y << ")" << std::endl;
    std::cout << "Knight " << k2.name << " is at (" << k2.x << "," << k2.y << ")" << std::endl;
    
    k1.update();
    k2.update();

    std::cout << "Knight " << k1.name << " is at (" << k1.x << "," << k1.y << ")" << std::endl;
    std::cout << "Knight " << k2.name << " is at (" << k2.x << "," << k2.y << ")" << std::endl;
    

    Archer a1(100,100, "Elma");
    Archer a2(102,100, "Legolas");
   
}
