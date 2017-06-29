#include "mainwindow.h"
#include "ui_mainwindow.h"
QString Data::figure="";
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->comboBox_X->addItem("1");
    ui->comboBox_X->addItem("2");
    ui->comboBox_X->addItem("3");
    ui->comboBox_X->addItem("4");

    ui->comboBox_Y->addItem("1");
    ui->comboBox_Y->addItem("2");
    ui->comboBox_Y->addItem("3");
    ui->comboBox_Y->addItem("4");

    QStringList all;
    all<<"121"<<"222"<<"343";
    ui->listWidget_All->addItems(all);


}

MainWindow::~MainWindow()
{
    delete ui;
}



void MainWindow::on_pushButton_Cal_clicked()
{
    //ui->comboBox->addItem("222");
    //QStringList header;
    //header<<"Month"<<"Description";
    //ui->tableWidget->setHorizontalHeaderLabels(header);
    QTableWidget *tableWidget = new QTableWidget(10,5);


            tableWidget->setWindowTitle("QTableWidget & Item");
                tableWidget->resize(350, 200);  //设置表格
                QStringList header;
                header<<"Month"<<"Description";
                tableWidget->setHorizontalHeaderLabels(header);
                tableWidget->setItem(0,0,new QTableWidgetItem("Jan"));
                tableWidget->setItem(1,0,new QTableWidgetItem("Feb"));
                tableWidget->setItem(2,0,new QTableWidgetItem("Mar"));

                tableWidget->setItem(0,1,new QTableWidgetItem(QIcon("images/IED.png"), "Jan's month"));
                tableWidget->setItem(1,1,new QTableWidgetItem(QIcon("images/IED.png"), "Feb's month"));
                tableWidget->setItem(2,1,new QTableWidgetItem(QIcon("images/IED.png"), "Mar's month"));
                //tableWidget->horizontalHeader()->setVisible(1);
                tableWidget->setAlternatingRowColors(true);
                //tableWidget->resizeColumnToContents(0);
                tableWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
                //tableWidget->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
                int height =0; //高度
                int row = tableWidget->rowCount();
                int i;
                for(i=0;i<row;i++)
                {
                    height += tableWidget->rowHeight(i);
                }

                int width=0; //宽度
                int column= tableWidget->columnCount();
                int j;
                for(j=0;j<column;j++)
                {
                    width+= tableWidget->columnWidth(j);
                }

                tableWidget->resize(width,height);
                tableWidget->show();

}



void MainWindow::on_comboBox_Y_currentIndexChanged(const QString &arg1)
{
    ui->textEdit_Y->setText(ui->comboBox_Y->currentText());
}

void MainWindow::on_comboBox_X_currentIndexChanged(const QString &arg1)
{
    ui->textEdit_X->setText(ui->comboBox_X->currentText());
}

void MainWindow::on_pushButton_R2_Cal_clicked()
{
    ui->textEdit_R2->setText(ui->textEdit_X->toPlainText()+ui->textEdit_Y->toPlainText());
}

void MainWindow::on_pushButton_RandForest_clicked()
{
    for(int i=0;i<ui->listWidget_select_X->count();i++)

        ui->textEdit_Randforest_result->append(ui->listWidget_select_X->item(i)->text());

    //ui->textEdit_Randforest_result->setText(ui->listWidget_select_X->item(0)->text());
    //int i=0;
    //ui->textEdit_Randforest_result->setText(ui->listWidget_select_X->item(i)->text());
}

void MainWindow::on_pushButton_select_Y_clicked()
{
    ui->lineEdit_select_Y->setText(ui->listWidget_All->currentItem()->text());
    //ui->listWidget_All->takeItem(ui->listWidget_All->currentRow());
    ui->lineEdit_select_Y->text();
}

void MainWindow::on_pushButton_select_X_clicked()
{

    ui->listWidget_select_X->addItem(ui->listWidget_All->currentItem()->text());
}

void MainWindow::on_pushButton_select_X_2_clicked()
{
    ui->listWidget_select_X->takeItem(ui->listWidget_select_X->currentRow());
}

void MainWindow::on_read_xlsx_clicked()
{
     QFileDialog  *fileDialog = new QFileDialog(this);//创建一个QFileDialog对象，构造函数中的参数可以有所添加。
     QString file = QFileDialog::getOpenFileName(this, tr("open file"), "C:/Users/zy/python_practice/QT/zy20170626/zy20170626",  tr("xlsxfile(*.xlsx);;xlsfile(*.xls)"));

     QFileInfo fi=QFileInfo(file);
     QString file_name=fi.fileName();
     QString file_path=fi.absolutePath();
     QMessageBox::about(this,"提示框","数据已经加载，可以进行后续计算");
     ui->textEdit_Y->setText(file_name);
     ui->textEdit_X->setText(file_path);





}

void MainWindow::on_pushButton_figure_clicked()
{

    if(Data::figure=="频率分布直方图")
        {
        QWebView *view=new QWebView();
        view->setWindowTitle(Data::figure);
        view->load(QUrl(QString("file:///C:/Users/zy/python_practice/QT/zy20170626/zy20170626/multi_hist_test.svg")));
        //view->showMaximized();
        view->showNormal();
        }



}

void MainWindow::on_radioButton_hist_clicked(bool checked)
{

    Data::figure=ui->radioButton_hist->text();
}
