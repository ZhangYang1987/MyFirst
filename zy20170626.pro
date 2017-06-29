#-------------------------------------------------
#
# Project created by QtCreator 2017-06-25T23:18:00
#
#-------------------------------------------------

QT       += core gui
QT       += webkitwidgets
QT +=network #Qt5中webkit模块已经更改为webkitwidgets
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = zy20170626
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui
