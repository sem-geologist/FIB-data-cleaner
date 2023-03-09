# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(1092, 831)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.splitter_3 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setHandleWidth(2)
        self.splitter_3.setObjectName("splitter_3")
        self.splitter_2 = QtWidgets.QSplitter(self.splitter_3)
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setHandleWidth(2)
        self.splitter_2.setObjectName("splitter_2")
        self.groupBox = QtWidgets.QGroupBox(self.splitter_2)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout_6.addWidget(self.label, 1, 0, 1, 1)
        self.transparency_slider = QtWidgets.QSlider(self.groupBox)
        self.transparency_slider.setEnabled(False)
        self.transparency_slider.setMaximum(100)
        self.transparency_slider.setPageStep(0)
        self.transparency_slider.setProperty("value", 50)
        self.transparency_slider.setOrientation(QtCore.Qt.Horizontal)
        self.transparency_slider.setObjectName("transparency_slider")
        self.gridLayout_6.addWidget(self.transparency_slider, 1, 5, 1, 1)
        self.lock_text = QtWidgets.QLabel(self.groupBox)
        self.lock_text.setObjectName("lock_text")
        self.gridLayout_6.addWidget(self.lock_text, 1, 3, 1, 1)
        self.slice_iv = ImageView(self.groupBox)
        self.slice_iv.setObjectName("slice_iv")
        self.gridLayout_6.addWidget(self.slice_iv, 0, 0, 1, 6)
        self.current_slice_n = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.current_slice_n.sizePolicy().hasHeightForWidth())
        self.current_slice_n.setSizePolicy(sizePolicy)
        self.current_slice_n.setObjectName("current_slice_n")
        self.gridLayout_6.addWidget(self.current_slice_n, 1, 1, 1, 1)
        self.slice_lock = QtWidgets.QToolButton(self.groupBox)
        self.slice_lock.setToolTip("")
        self.slice_lock.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.slice_lock.setCheckable(True)
        self.slice_lock.setChecked(False)
        self.slice_lock.setObjectName("slice_lock")
        self.gridLayout_6.addWidget(self.slice_lock, 1, 2, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(self.splitter_2)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.pb_stack_in_arrays = QtWidgets.QPushButton(self.tab_3)
        self.pb_stack_in_arrays.setEnabled(False)
        self.pb_stack_in_arrays.setObjectName("pb_stack_in_arrays")
        self.gridLayout_5.addWidget(self.pb_stack_in_arrays, 4, 0, 1, 1)
        self.check_box_blacklisted = QtWidgets.QCheckBox(self.tab_3)
        self.check_box_blacklisted.setObjectName("check_box_blacklisted")
        self.gridLayout_5.addWidget(self.check_box_blacklisted, 2, 0, 1, 1)
        self.splitter_5 = QtWidgets.QSplitter(self.tab_3)
        self.splitter_5.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_5.setObjectName("splitter_5")
        self.tv_metadata = DataTreeWidget(self.splitter_5)
        self.tv_metadata.setEnabled(False)
        self.tv_metadata.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tv_metadata.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tv_metadata.setObjectName("tv_metadata")
        self.pw_metadata = PlotWidget(self.splitter_5)
        self.pw_metadata.setEnabled(False)
        self.pw_metadata.setObjectName("pw_metadata")
        self.gridLayout_5.addWidget(self.splitter_5, 3, 0, 1, 2)
        self.cb_treeview_source = QtWidgets.QComboBox(self.tab_3)
        self.cb_treeview_source.setEnabled(False)
        self.cb_treeview_source.setObjectName("cb_treeview_source")
        self.cb_treeview_source.addItem("")
        self.cb_treeview_source.addItem("")
        self.gridLayout_5.addWidget(self.cb_treeview_source, 2, 1, 1, 1)
        self.pb_destroy_stack_elements = QtWidgets.QPushButton(self.tab_3)
        self.pb_destroy_stack_elements.setEnabled(False)
        self.pb_destroy_stack_elements.setObjectName("pb_destroy_stack_elements")
        self.gridLayout_5.addWidget(self.pb_destroy_stack_elements, 4, 1, 1, 1)
        self.tabWidget.addTab(self.tab_3, "")
        self.tabAlign = QtWidgets.QWidget()
        self.tabAlign.setObjectName("tabAlign")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.tabAlign)
        self.verticalLayout_7.setContentsMargins(1, 1, 1, 1)
        self.verticalLayout_7.setSpacing(1)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.matrix_widget = QtWidgets.QWidget(self.tabAlign)
        self.matrix_widget.setObjectName("matrix_widget")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.matrix_widget)
        self.verticalLayout_6.setContentsMargins(1, 1, 1, 1)
        self.verticalLayout_6.setSpacing(1)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.widget_2 = QtWidgets.QWidget(self.matrix_widget)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_3.setContentsMargins(1, 1, 1, 1)
        self.horizontalLayout_3.setSpacing(1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_6.addWidget(self.widget_2)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setSpacing(1)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.groupBox_4 = QtWidgets.QGroupBox(self.matrix_widget)
        self.groupBox_4.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout.setContentsMargins(1, 1, 1, 1)
        self.gridLayout.setSpacing(1)
        self.gridLayout.setObjectName("gridLayout")
        self.widget_3 = QtWidgets.QWidget(self.groupBox_4)
        self.widget_3.setEnabled(True)
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_2.setContentsMargins(1, 1, 1, 1)
        self.horizontalLayout_2.setSpacing(1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.gridLayout.addWidget(self.widget_3, 1, 0, 1, 1)
        self.mtv = QtWidgets.QTableView(self.groupBox_4)
        self.mtv.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mtv.sizePolicy().hasHeightForWidth())
        self.mtv.setSizePolicy(sizePolicy)
        self.mtv.setMinimumSize(QtCore.QSize(100, 100))
        self.mtv.setBaseSize(QtCore.QSize(100, 100))
        self.mtv.setObjectName("mtv")
        self.gridLayout.addWidget(self.mtv, 0, 0, 1, 1)
        self.widget = QtWidgets.QWidget(self.groupBox_4)
        self.widget.setObjectName("widget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_5.setContentsMargins(1, 1, 1, 1)
        self.verticalLayout_5.setSpacing(1)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.init_slice_btn = QtWidgets.QPushButton(self.widget)
        self.init_slice_btn.setEnabled(False)
        self.init_slice_btn.setCheckable(True)
        self.init_slice_btn.setObjectName("init_slice_btn")
        self.verticalLayout_5.addWidget(self.init_slice_btn)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem)
        self.widget_4 = QtWidgets.QWidget(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_4.sizePolicy().hasHeightForWidth())
        self.widget_4.setSizePolicy(sizePolicy)
        self.widget_4.setObjectName("widget_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget_4)
        self.gridLayout_4.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setSpacing(1)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.pin_points_to_slice = QtWidgets.QPushButton(self.widget_4)
        self.pin_points_to_slice.setEnabled(False)
        self.pin_points_to_slice.setCheckable(True)
        self.pin_points_to_slice.setObjectName("pin_points_to_slice")
        self.gridLayout_4.addWidget(self.pin_points_to_slice, 1, 0, 1, 1)
        self.reset_triangle = QtWidgets.QToolButton(self.widget_4)
        self.reset_triangle.setEnabled(False)
        self.reset_triangle.setObjectName("reset_triangle")
        self.gridLayout_4.addWidget(self.reset_triangle, 1, 1, 1, 1)
        self.show_ref_points = QtWidgets.QPushButton(self.widget_4)
        self.show_ref_points.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.show_ref_points.sizePolicy().hasHeightForWidth())
        self.show_ref_points.setSizePolicy(sizePolicy)
        self.show_ref_points.setCheckable(True)
        self.show_ref_points.setObjectName("show_ref_points")
        self.gridLayout_4.addWidget(self.show_ref_points, 0, 0, 1, 2)
        self.verticalLayout_5.addWidget(self.widget_4)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem1)
        self.reset_matrix = QtWidgets.QPushButton(self.widget)
        self.reset_matrix.setEnabled(False)
        self.reset_matrix.setObjectName("reset_matrix")
        self.verticalLayout_5.addWidget(self.reset_matrix)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem2)
        self.comboMatrixMode = QtWidgets.QComboBox(self.widget)
        self.comboMatrixMode.setObjectName("comboMatrixMode")
        self.comboMatrixMode.addItem("")
        self.comboMatrixMode.addItem("")
        self.comboMatrixMode.addItem("")
        self.verticalLayout_5.addWidget(self.comboMatrixMode)
        self.pushMultiplyMatrices = QtWidgets.QPushButton(self.widget)
        self.pushMultiplyMatrices.setEnabled(False)
        self.pushMultiplyMatrices.setObjectName("pushMultiplyMatrices")
        self.verticalLayout_5.addWidget(self.pushMultiplyMatrices)
        self.gridLayout.addWidget(self.widget, 0, 1, 2, 1)
        self.gridLayout_3.addWidget(self.groupBox_4, 0, 0, 2, 1)
        self.sc_group = QtWidgets.QGroupBox(self.matrix_widget)
        self.sc_group.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sc_group.sizePolicy().hasHeightForWidth())
        self.sc_group.setSizePolicy(sizePolicy)
        self.sc_group.setAlignment(QtCore.Qt.AlignCenter)
        self.sc_group.setObjectName("sc_group")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.sc_group)
        self.gridLayout_2.setContentsMargins(1, 1, 1, 1)
        self.gridLayout_2.setSpacing(1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.left_button = QtWidgets.QToolButton(self.sc_group)
        self.left_button.setAutoRepeat(True)
        self.left_button.setObjectName("left_button")
        self.gridLayout_2.addWidget(self.left_button, 1, 0, 1, 1)
        self.down_button = QtWidgets.QToolButton(self.sc_group)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.down_button.sizePolicy().hasHeightForWidth())
        self.down_button.setSizePolicy(sizePolicy)
        self.down_button.setAutoRepeat(True)
        self.down_button.setObjectName("down_button")
        self.gridLayout_2.addWidget(self.down_button, 2, 1, 1, 1)
        self.strenght_spinbox = QtWidgets.QSpinBox(self.sc_group)
        self.strenght_spinbox.setMinimum(1)
        self.strenght_spinbox.setProperty("value", 1)
        self.strenght_spinbox.setObjectName("strenght_spinbox")
        self.gridLayout_2.addWidget(self.strenght_spinbox, 1, 1, 1, 1)
        self.right_button = QtWidgets.QToolButton(self.sc_group)
        self.right_button.setAutoRepeat(True)
        self.right_button.setObjectName("right_button")
        self.gridLayout_2.addWidget(self.right_button, 1, 2, 1, 1)
        self.up_button = QtWidgets.QToolButton(self.sc_group)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.up_button.sizePolicy().hasHeightForWidth())
        self.up_button.setSizePolicy(sizePolicy)
        self.up_button.setAutoRepeat(True)
        self.up_button.setObjectName("up_button")
        self.gridLayout_2.addWidget(self.up_button, 0, 1, 1, 1)
        self.label_x_correction = QtWidgets.QLabel(self.sc_group)
        self.label_x_correction.setObjectName("label_x_correction")
        self.gridLayout_2.addWidget(self.label_x_correction, 0, 0, 1, 1)
        self.label_y_correction = QtWidgets.QLabel(self.sc_group)
        self.label_y_correction.setObjectName("label_y_correction")
        self.gridLayout_2.addWidget(self.label_y_correction, 2, 2, 1, 1)
        self.gridLayout_3.addWidget(self.sc_group, 0, 1, 1, 2)
        self.verticalLayout_6.addLayout(self.gridLayout_3)
        self.verticalLayout_7.addWidget(self.matrix_widget)
        self.tabWidget.addTab(self.tabAlign, "")
        self.layoutWidget = QtWidgets.QWidget(self.splitter_3)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.splitter = QtWidgets.QSplitter(self.layoutWidget)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setHandleWidth(2)
        self.splitter.setObjectName("splitter")
        self.groupBox_2 = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_3.setContentsMargins(1, 1, 1, 1)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.v_depth_iv = ImageView(self.groupBox_2)
        self.v_depth_iv.setObjectName("v_depth_iv")
        self.verticalLayout_3.addWidget(self.v_depth_iv)
        self.groupBox_3 = QtWidgets.QGroupBox(self.splitter)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_4.setContentsMargins(1, 1, 1, 1)
        self.verticalLayout_4.setSpacing(1)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.h_depth_iv = ImageView(self.groupBox_3)
        self.h_depth_iv.setObjectName("h_depth_iv")
        self.verticalLayout_4.addWidget(self.h_depth_iv)
        self.verticalLayout.addWidget(self.splitter)
        self.widget_5 = QtWidgets.QWidget(self.layoutWidget)
        self.widget_5.setObjectName("widget_5")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_5)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.widget_5)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.slices_spinbox = QtWidgets.QSpinBox(self.widget_5)
        self.slices_spinbox.setMinimum(1)
        self.slices_spinbox.setMaximum(100)
        self.slices_spinbox.setObjectName("slices_spinbox")
        self.horizontalLayout_4.addWidget(self.slices_spinbox)
        self.to_end_checkbox = QtWidgets.QCheckBox(self.widget_5)
        self.to_end_checkbox.setObjectName("to_end_checkbox")
        self.horizontalLayout_4.addWidget(self.to_end_checkbox)
        self.verticalLayout.addWidget(self.widget_5)
        self.horizontalLayout.addWidget(self.splitter_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1092, 30))
        self.menubar.setObjectName("menubar")
        self.menuFiles = QtWidgets.QMenu(self.menubar)
        self.menuFiles.setObjectName("menuFiles")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuhelp = QtWidgets.QMenu(self.menubar)
        self.menuhelp.setObjectName("menuhelp")
        MainWindow.setMenuBar(self.menubar)
        self.dockConsole = QtWidgets.QDockWidget(MainWindow)
        self.dockConsole.setObjectName("dockConsole")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.console = JupyterConsoleWidget(self.dockWidgetContents)
        self.console.setGeometry(QtCore.QRect(-110, 0, 971, 33))
        self.console.setObjectName("console")
        self.dockConsole.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.dockConsole)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setEnabled(True)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.actionLoad = QtWidgets.QAction(MainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.actionLoad_corrections = QtWidgets.QAction(MainWindow)
        self.actionLoad_corrections.setEnabled(False)
        self.actionLoad_corrections.setObjectName("actionLoad_corrections")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionGet_z_scale = QtWidgets.QAction(MainWindow)
        self.actionGet_z_scale.setEnabled(False)
        self.actionGet_z_scale.setObjectName("actionGet_z_scale")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setEnabled(False)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_corrections = QtWidgets.QAction(MainWindow)
        self.actionSave_corrections.setEnabled(False)
        self.actionSave_corrections.setObjectName("actionSave_corrections")
        self.actionshow_markers = QtWidgets.QAction(MainWindow)
        self.actionshow_markers.setCheckable(True)
        self.actionshow_markers.setEnabled(False)
        self.actionshow_markers.setObjectName("actionshow_markers")
        self.actionCrop = QtWidgets.QAction(MainWindow)
        self.actionCrop.setEnabled(False)
        self.actionCrop.setObjectName("actionCrop")
        self.actionNormalize = QtWidgets.QAction(MainWindow)
        self.actionNormalize.setObjectName("actionNormalize")
        self.actionAbout_Qt = QtWidgets.QAction(MainWindow)
        self.actionAbout_Qt.setObjectName("actionAbout_Qt")
        self.actionAbout_this_software = QtWidgets.QAction(MainWindow)
        self.actionAbout_this_software.setObjectName("actionAbout_this_software")
        self.actionlock_onto_current_slice = QtWidgets.QAction(MainWindow)
        self.actionlock_onto_current_slice.setCheckable(True)
        self.actionlock_onto_current_slice.setObjectName("actionlock_onto_current_slice")
        self.actionVertical_shift_guide = QtWidgets.QAction(MainWindow)
        self.actionVertical_shift_guide.setCheckable(True)
        self.actionVertical_shift_guide.setObjectName("actionVertical_shift_guide")
        self.actionHorizontal_shift_guide = QtWidgets.QAction(MainWindow)
        self.actionHorizontal_shift_guide.setCheckable(True)
        self.actionHorizontal_shift_guide.setObjectName("actionHorizontal_shift_guide")
        self.actionshow_python_console = QtWidgets.QAction(MainWindow)
        self.actionshow_python_console.setCheckable(True)
        self.actionshow_python_console.setObjectName("actionshow_python_console")
        self.actionExport_with_hs = QtWidgets.QAction(MainWindow)
        self.actionExport_with_hs.setEnabled(False)
        self.actionExport_with_hs.setObjectName("actionExport_with_hs")
        self.actionExport_with_pg = QtWidgets.QAction(MainWindow)
        self.actionExport_with_pg.setEnabled(False)
        self.actionExport_with_pg.setObjectName("actionExport_with_pg")
        self.menuFiles.addAction(self.actionLoad)
        self.menuFiles.addAction(self.actionSave)
        self.menuFiles.addAction(self.actionLoad_corrections)
        self.menuFiles.addAction(self.actionSave_corrections)
        self.menuFiles.addSeparator()
        self.menuFiles.addAction(self.actionExport_with_hs)
        self.menuFiles.addAction(self.actionExport_with_pg)
        self.menuFiles.addSeparator()
        self.menuFiles.addAction(self.actionExit)
        self.menuEdit.addAction(self.actionCrop)
        self.menuEdit.addAction(self.actionNormalize)
        self.menuView.addAction(self.actionshow_markers)
        self.menuView.addAction(self.actionlock_onto_current_slice)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionVertical_shift_guide)
        self.menuView.addAction(self.actionHorizontal_shift_guide)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionGet_z_scale)
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionshow_python_console)
        self.menuhelp.addAction(self.actionAbout_Qt)
        self.menuhelp.addAction(self.actionAbout_this_software)
        self.menubar.addAction(self.menuFiles.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuhelp.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.actionExit.triggered.connect(MainWindow.close) # type: ignore
        self.to_end_checkbox.toggled['bool'].connect(self.slices_spinbox.setDisabled) # type: ignore
        self.actionshow_python_console.toggled['bool'].connect(self.dockConsole.setVisible) # type: ignore
        self.dockConsole.visibilityChanged['bool'].connect(self.actionshow_python_console.setChecked) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Standard view (x, y)"))
        self.label.setText(_translate("MainWindow", "Current slice: "))
        self.transparency_slider.setToolTip(_translate("MainWindow", "Transparency of the locked slice"))
        self.lock_text.setText(_translate("MainWindow", " Lock onto current slice"))
        self.current_slice_n.setText(_translate("MainWindow", "None"))
        self.slice_lock.setText(_translate("MainWindow", "..."))
        self.pb_stack_in_arrays.setText(_translate("MainWindow", "push selected to stack_as_arrays"))
        self.check_box_blacklisted.setToolTip(_translate("MainWindow", "<html><head/><body><p>In some FIB-SEM cases, the milling is not refreshing the wall, and consequtive SEM slices taken represent the same wall. Checking this box will remove marked stack when saving the stack as a new file(s).</p></body></html>"))
        self.check_box_blacklisted.setText(_translate("MainWindow", "Blacklist current slice"))
        self.check_box_blacklisted.setShortcut(_translate("MainWindow", "Shift+Del"))
        self.cb_treeview_source.setToolTip(_translate("MainWindow", "<html><head/><body><p>Currently stack_elements in original_metadata of hyperspy signal is provided only when loading separate files and using <span style=\" font-weight:600;\">stack=True</span>, ... Files with whole stack such as hspy can have the files with converted types to bytes or without the stack_elements present. This software createas new branch under original_metadata called &quot;stack_in_arrays&quot; - which aggregates selected numerical metadata from stack_elements into arrays, so that saving and loading hspy would return consistent metadata for slices.</p></body></html>"))
        self.cb_treeview_source.setItemText(0, _translate("MainWindow", "stack_elements"))
        self.cb_treeview_source.setItemText(1, _translate("MainWindow", "stack_in_arrays"))
        self.pb_destroy_stack_elements.setText(_translate("MainWindow", "Discard stack_elements"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "MetaPlot"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Matrix Correction"))
        self.init_slice_btn.setText(_translate("MainWindow", "init for slice"))
        self.pin_points_to_slice.setText(_translate("MainWindow", "🖈⨻"))
        self.reset_triangle.setText(_translate("MainWindow", "..."))
        self.show_ref_points.setText(_translate("MainWindow", "3 point def."))
        self.reset_matrix.setText(_translate("MainWindow", "reset"))
        self.comboMatrixMode.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\">modify current</span>: modify directly matrix of affine transformation of currently set slice.</p><p><span style=\" font-weight:600;\">pre-multiply</span>: widget shows other matrix, which can be used for scaling, translation, skew, rotation, and that is multiplied with current matrix of slice or slices, and multiplication is applied with defined matrix being initial and current matrix being additional.</p><p><span style=\" font-weight:600;\">post-multiply</span>: is like pre-multiply just with different order of multiplication.</p></body></html>"))
        self.comboMatrixMode.setItemText(0, _translate("MainWindow", "modify current slice"))
        self.comboMatrixMode.setItemText(1, _translate("MainWindow", "pre-multiply selection"))
        self.comboMatrixMode.setItemText(2, _translate("MainWindow", "post-multiply selection"))
        self.pushMultiplyMatrices.setToolTip(_translate("MainWindow", "multiply currenly present matrice with affine transformation matrix/-eces of slice(-s)"))
        self.pushMultiplyMatrices.setText(_translate("MainWindow", "Multiply"))
        self.sc_group.setTitle(_translate("MainWindow", "Simple Shift "))
        self.left_button.setToolTip(_translate("MainWindow", "Shift+Left"))
        self.left_button.setText(_translate("MainWindow", "←"))
        self.left_button.setShortcut(_translate("MainWindow", "Shift+Left"))
        self.down_button.setToolTip(_translate("MainWindow", "Shift+Down"))
        self.down_button.setText(_translate("MainWindow", "↓"))
        self.down_button.setShortcut(_translate("MainWindow", "Shift+Down"))
        self.strenght_spinbox.setSuffix(_translate("MainWindow", " px"))
        self.right_button.setToolTip(_translate("MainWindow", "Shift+Right"))
        self.right_button.setText(_translate("MainWindow", "→"))
        self.right_button.setShortcut(_translate("MainWindow", "Shift+Right"))
        self.up_button.setToolTip(_translate("MainWindow", "Shift+Up"))
        self.up_button.setText(_translate("MainWindow", "↑"))
        self.up_button.setShortcut(_translate("MainWindow", "Shift+Up"))
        self.label_x_correction.setText(_translate("MainWindow", "x:"))
        self.label_y_correction.setText(_translate("MainWindow", "y:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabAlign), _translate("MainWindow", "Align"))
        self.groupBox_2.setTitle(_translate("MainWindow", " (z, y) at marked x ( vertical)"))
        self.groupBox_3.setTitle(_translate("MainWindow", "(x, z) at marked y (horizontal)"))
        self.label_2.setText(_translate("MainWindow", "num slices:"))
        self.slices_spinbox.setSuffix(_translate("MainWindow", " sl."))
        self.to_end_checkbox.setText(_translate("MainWindow", "up to end"))
        self.to_end_checkbox.setShortcut(_translate("MainWindow", "Ctrl+E"))
        self.menuFiles.setTitle(_translate("MainWindow", "Files"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuhelp.setTitle(_translate("MainWindow", "Help"))
        self.dockConsole.setWindowTitle(_translate("MainWindow", "QtConsole (Jupyter)"))
        self.actionLoad.setText(_translate("MainWindow", "Load HyperCube"))
        self.actionLoad_corrections.setText(_translate("MainWindow", "Load and apply corrections"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionGet_z_scale.setText(_translate("MainWindow", "update inter-slice distance"))
        self.actionSave.setText(_translate("MainWindow", "Save as ..."))
        self.actionSave_corrections.setText(_translate("MainWindow", "Save corrections to json"))
        self.actionshow_markers.setText(_translate("MainWindow", "ROI at (x,y) extending as lines at (z, y) and (x, z)"))
        self.actionCrop.setText(_translate("MainWindow", "Crop"))
        self.actionNormalize.setText(_translate("MainWindow", "Normalize in between markers"))
        self.actionAbout_Qt.setText(_translate("MainWindow", "About Qt"))
        self.actionAbout_this_software.setText(_translate("MainWindow", "About this software"))
        self.actionlock_onto_current_slice.setText(_translate("MainWindow", "lock onto current slice"))
        self.actionlock_onto_current_slice.setShortcut(_translate("MainWindow", "Ctrl+T"))
        self.actionVertical_shift_guide.setText(_translate("MainWindow", "Vertical guided shift at (z, y)"))
        self.actionHorizontal_shift_guide.setText(_translate("MainWindow", "Horizontal guided shift at (x, z)"))
        self.actionshow_python_console.setText(_translate("MainWindow", "Jupyter console"))
        self.actionshow_python_console.setShortcut(_translate("MainWindow", "Ctrl+Alt+C"))
        self.actionExport_with_hs.setText(_translate("MainWindow", "export as image series (w/o LUT)"))
        self.actionExport_with_pg.setText(_translate("MainWindow", "export as image series (w/ LUT)"))
from .Console import JupyterConsoleWidget
from pyqtgraph import DataTreeWidget, ImageView, PlotWidget
