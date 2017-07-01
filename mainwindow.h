#ifndef MAINWINDOW_H
#define MAINWINDOW_H


#include <QMainWindow>
#include <QFileDialog>

#include <QtWebKit>
#include <QMessageBox>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_Cal_clicked();

    void on_comboBox_Y_currentIndexChanged(const QString &arg1);

    void on_comboBox_X_currentIndexChanged(const QString &arg1);

    void on_pushButton_R2_Cal_clicked();

    void on_pushButton_RandForest_clicked();

    void on_pushButton_select_Y_clicked();

    void on_pushButton_select_X_clicked();

    void on_pushButton_select_X_2_clicked();

    void on_read_xlsx_clicked();

    void on_pushButton_figure_clicked();

    void on_radioButton_hist_clicked(bool checked);

    void on_actionOpen_triggered();

private:
    Ui::MainWindow *ui;
};


#endif // MAINWINDOW_H



#ifndef DATA_H
#define DATA_H
class Data
{
public:
    static QString figure;
};
#endif
