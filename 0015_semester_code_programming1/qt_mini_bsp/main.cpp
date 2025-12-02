#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QLabel>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    int counter = 0;

    QWidget window;
    window.setWindowTitle("Das ist mein erstes Qt Programm");

    auto *layout = new QHBoxLayout(&window);

    QLabel *label = new QLabel("0");
    label->setAlignment(Qt::AlignCenter);
    label->setStyleSheet("font-size: 150px; font-weight: bold; color: red; background-color: black");
    layout->addWidget(label);

    label->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    label->setMaximumHeight(300);

    QPushButton *btn1 = new QPushButton("Zähle um 1 hoch");
    QPushButton *btn5 = new QPushButton("Zähle um 5 hoch");

    btn1->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    btn1->setMaximumHeight(40);

    btn5->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    btn5->setMaximumHeight(40);


    layout->addWidget(btn1);
    layout->addWidget(btn5);

    QObject::connect(btn1, &QPushButton::clicked, [&](){
        counter += 1;
        label->setText(QString::number(counter));
    });

    QObject::connect(btn5, &QPushButton::clicked, [&](){
        counter += 5;
        label->setText(QString::number(counter));
    });

    window.show();
    return app.exec();
}
