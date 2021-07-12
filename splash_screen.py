from PyQt5.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *


class Ui_SplashScreen(object):
    def setupUi(self, SplashScreen):
        if not SplashScreen.objectName():
            SplashScreen.setObjectName(u"SplashScreen")
        SplashScreen.resize(680, 400)
        self.centralwidget = QWidget(SplashScreen)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.dropShadowFrame = QFrame(self.centralwidget)
        self.dropShadowFrame.setObjectName(u"dropShadowFrame")
        self.dropShadowFrame.setStyleSheet(u"QFrame{\n"
"	\n"
"	background-color: rgb(40, 44, 52);\n"
"	color: rgb(150, 150, 150);\n"
"	border-radius: 20px;\n"
"}\n"
"")
        self.dropShadowFrame.setFrameShape(QFrame.StyledPanel)
        self.dropShadowFrame.setFrameShadow(QFrame.Raised)
        self.label_title = QLabel(self.dropShadowFrame)
        self.label_title.setObjectName(u"label_title")
        self.label_title.setGeometry(QRect(0, 50, 661, 91))
        font = QFont()
        font.setFamily(u"El Messiri")
        font.setPointSize(30)
        self.label_title.setFont(font)
        self.label_title.setStyleSheet(u"color: rgb(189, 147, 249);")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_description = QLabel(self.dropShadowFrame)
        self.label_description.setObjectName(u"label_description")
        self.label_description.setGeometry(QRect(0, 130, 661, 31))
        font1 = QFont()
        font1.setFamily(u"Poppins")
        font1.setPointSize(14)
        self.label_description.setFont(font1)
        self.label_description.setStyleSheet(u"color: rgb(150, 150, 150);")
        self.label_description.setAlignment(Qt.AlignCenter)
        self.progressBar = QProgressBar(self.dropShadowFrame)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setGeometry(QRect(50, 240, 561, 23))
        self.progressBar.setStyleSheet(u"QProgressBar {\n"
"\n"
"    background-color: rgb(98, 114, 164);\n"
"    color: rgb(250, 250, 250);\n"
"    border-style: none;\n"
"    border-radius: 10px;\n"
"    text-align: center;\n"
"}\n"
"QProgressBar::chunk{\n"
"    border-radius: 10px;\n"
"    background-color: qlineargradient(spread:pad, x1:0, y1:0.511364, x2:1, y2:0.523, stop:0 rgba(254, 121, 199, 255), stop:1 rgba(189, 147, 249, 255));\n"
"}")
        self.progressBar.setValue(24)
        self.label_credit = QLabel(self.dropShadowFrame)
        self.label_credit.setObjectName(u"label_credit")
        self.label_credit.setGeometry(QRect(0, 340, 645, 31))
        self.label_credit2 = QLabel(self.dropShadowFrame)
        self.label_credit2.setObjectName(u"label_credit")
        self.label_credit2.setGeometry(QRect(20, 340, 401, 31))
        font1 = QFont()
        font1.setFamily(u"Poppins")
        font1.setPointSize(6)
        self.label_credit.setFont(font1)
        self.label_credit.setStyleSheet(u"color: rgb(150, 150, 150);")
        self.label_credit.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_credit2.setFont(font1)
        self.label_credit2.setStyleSheet(u"color: rgb(150, 150, 150);")
        self.label_credit2.setAlignment(Qt.AlignJustify|Qt.AlignVCenter)
        self.label_loading = QLabel(self.dropShadowFrame)
        self.label_loading.setObjectName(u"label_loading")
        self.label_loading.setGeometry(QRect(0, 270, 661, 31))
        font2 = QFont()
        font2.setFamily(u"Poppins")
        font2.setPointSize(12)
        self.label_loading.setFont(font2)
        self.label_loading.setStyleSheet(u"color: rgb(150, 150, 150);")
        self.label_loading.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.dropShadowFrame)

        SplashScreen.setCentralWidget(self.centralwidget)

        self.retranslateUi(SplashScreen)

        QMetaObject.connectSlotsByName(SplashScreen)
    # setupUi

    def retranslateUi(self, SplashScreen):
        SplashScreen.setWindowTitle(QCoreApplication.translate("SplashScreen", u"MainWindow", None))
        self.label_title.setText(QCoreApplication.translate("SplashScreen", u"<strong>PALANTAS</strong>", None))
        self.label_description.setText(QCoreApplication.translate("SplashScreen", u"<html><head/><body><p><span style=\" font-size:14pt;\">YOLOv4</span></p></body></html>", None))
        self.label_credit.setText(QCoreApplication.translate("SplashScreen", u"<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">Created:</span><span style=\" font-size:9pt;\"> alwazirf</span></p></body></html>", None))
        self.label_credit2.setText(QCoreApplication.translate("SplashScreen", u"<html><head/><body><p><span style=\" font-size:9pt;\"> Azhar, S.T., M.T | Hendrawaty, S.T., M.T</span></p></body></html>", None))
        self.label_loading.setText(QCoreApplication.translate("SplashScreen", u"<html><head/><body><p><span style=\" font-size:12pt;\">Please Wait</span></p></body></html>", None))
    # retranslateUi

