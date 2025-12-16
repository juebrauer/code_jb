#include <iostream>
#include <vector>


class Figure
{
    public:

        Figure(float _x, float _y, std::string _name) : x(_x), y(_y),
           energy(1.0), name(_name), dx(1.0), dy(1.0)
        {

        }

        virtual void update()
        {
            this->x += this->dx;
            this->y += this->dy;
        }

        void show_info()
        {
            std::cout << name << " is at " << x << "," << y << std::endl;
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

        void update() override
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

        void update() override
        {
           this->x += 1.5 * this->dx;
           this->y += 1.5 * this->dy;
        } 

      
};


int main()
{
    std::vector<Figure*> all_figures;
        
    Figure* k1 = new Knight(10,10, "Aragon", true);
    Figure* k2 = new Knight(10,10, false);

    Figure* a1 = new Archer(100,100, "Elma");
    Figure* a2 = new Archer(102,100, "Legolas");

    all_figures.push_back( k1 );
    all_figures.push_back( k2 );
    all_figures.push_back( a1 );
    all_figures.push_back( a2 );

    for (int i=0; i<all_figures.size(); i++)
    {
        all_figures[i]->update();
        all_figures[i]->show_info();
    }
   
    delete k1;
    delete k2;
    delete a1;
    delete a2;
}
