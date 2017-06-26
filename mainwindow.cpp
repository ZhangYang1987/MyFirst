#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->comboBox->addItem("2");
    ui->comboBox->addItem("3");

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_Cal_clicked()
{
    ui->comboBox->addItem("222");
    QStringList header;
    header<<"Month"<<"Description";
    ui->tableWidget->setHorizontalHeaderLabels(header);

}
