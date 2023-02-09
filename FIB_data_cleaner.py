import sys
import json
import gc
import numpy as np
from os import path
from copy import copy
from PyQt5 import QtCore, QtGui, QtWidgets
import hyperspy.api as hs
import pyqtgraph as pg
from ui import MainWindow
from scipy.ndimage import affine_transform
pg.setConfigOption('imageAxisOrder', 'row-major')


class FIBSliceCorrector(QtWidgets.QMainWindow,
                        MainWindow.Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.first_time_load = True
        self.at_matrices = {}  # affine transformation matrices
        delegate = SpinBoxDelegate()
        self.mtv.setItemDelegate(delegate)
        self.actionLoad.triggered.connect(self.load_cube)
        self.actionAbout_Qt.triggered.connect(self.show_about_qt)
        self.actionAbout_this_software.triggered.connect(self.show_about)
        self.v_depth_iv.ui.roiBtn.hide()
        self.h_depth_iv.ui.roiBtn.hide()
        self.h_depth_iv.getView().invertY(False)
        self.slice_iv.ui.roiBtn.hide()
        self.slice_iv.ui.menuBtn.hide()
        self.h_depth_iv.ui.menuBtn.hide()
        self.v_depth_iv.ui.menuBtn.hide()
        # simple shift (True), affine matrix (False):
        self.simple_mode = True

    @classmethod
    def gen_affine_transformation_matrix(cls):
        return np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]],
                        dtype=np.float64)

    def get_index_for_manipulation(self):
        if self.slice_lock.isChecked():
            return self.locked_index
        return self.current_i

    def init_at_at_index(self, index_z=None):
        """generate and set initial affine transformation matrix
           for given index, also generate model, which can be
           viewed with QTableView"""
        if index_z is None:
            current_flag = True
            index_z = self.get_index_for_manipulation()
        if index_z not in self.at_matrices:
            self.at_matrices[index_z] = {
                'matrix': self.gen_affine_transformation_matrix()}
            TransformationMatrixModel(self.at_matrices[index_z])
            self.at_matrices[index_z]['initial'] = copy(self.i_data[index_z])
            if current_flag:
                self.switch_to_matrix_correction_widget()

    def remove_at_and_reset(self):
        i = self.get_index_for_manipulation()
        self.i_data[i] = self.at_matrices[i]['initial']
        self.at_matrices[i]['model'].dataChanged.disconnect(
            self.apply_affine_transformation)
        self.mtv.setModel(None)
        del self.at_matrices[i]
        self.mtv.setEnabled(False)
        self.reset_matrix.setEnabled(False)
        self.update_images()
        self.switch_to_simple_shift_widget()

    def switch_to_matrix_correction_widget(self):
        i = self.get_index_for_manipulation()
        self.mtv.setModel(self.at_matrices[i]['model'])
        self.show_ref_points.setEnabled(True)
        self.reset_matrix.setEnabled(True)
        self.init_slice_btn.setEnabled(False)
        self.mtv.setEnabled(True)
        self.at_matrices[i]['model'].dataChanged.connect(
            self.apply_affine_transformation)
        self.sc_group.setEnabled(False)
        self.to_end_checkbox.setChecked(False)
        self.slices_spinbox.setValue(1)
        self.simple_mode = False

    def switch_to_simple_shift_widget(self):
        self.show_ref_points.setEnabled(False)
        self.reset_matrix.setEnabled(False)
        self.init_slice_btn.blockSignals(True)
        self.init_slice_btn.setChecked(False)
        self.init_slice_btn.blockSignals(False)
        self.init_slice_btn.setEnabled(True)
        self.mtv.setEnabled(False)
        self.mtv.setModel(None)
        self.sc_group.setEnabled(True)
        self.simple_mode = True

    def apply_shifts(self, shifts):
        for i in range(self.i_data.shape[0]):
            self.i_data[i] = np.roll(self.i_data[i], shifts[i, 0], axis=1)
            self.i_data[i] = np.roll(self.i_data[i], shifts[i, 1], axis=0)

    def save_corrections(self):
        at_jsonized = {str(i): self.at_matrices[i]['matrix'].tolist()
                       for i in self.at_matrices}
        roi_jsonized = {'pos': tuple(self.roi_main.pos()),
                        'size': tuple(self.roi_main.size())}
        jsonable_dict = {'shifts': self.shifts.tolist(),
                         'at': at_jsonized,
                         'roi': roi_jsonized}
        fn, ft = QtWidgets.QFileDialog.getSaveFileName(
            self, 'give name for correction file', '../',
            "Json correction file (*.json)")
        if ft == '':
            return
        with open(fn, 'w') as fp:
            json.dump(jsonable_dict, fp)

    def load_shifts(self):
        fn, ft = QtWidgets.QFileDialog.getOpenFileName(
            self, 'select correction file', '../',
            "Numpy files (*.npy);; Json correction (*.json)")
        if ft == '':
            return
        self.apply_shifts(-self.shifts)
        if '.npy' in ft:
            shifts = np.load(fn).astype(np.int32)
            self.apply_shifts(shifts)
            self.shifts = shifts
        elif '.json' in ft:
            with open(fn, 'r') as file_pointer:
                corrections = json.load(file_pointer)
            self.shifts = np.asarray(corrections['shifts'], dtype=np.int32)
            self.apply_shifts(self.shifts)
            if 'roi' in corrections:
                self.roi_main.setPos(corrections['roi']['pos'])
                self.roi_main.setSize(corrections['roi']['size'])
            at = {int(i): np.asarray(corrections['at'][i], dtype=np.float64)
                  for i in corrections['at']}
            for i in at:
                self.at_matrices[i] = {'matrix': at[i]}
                TransformationMatrixModel(self.at_matrices[i])
                self.at_matrices[i]['initial'] = copy(self.i_data[i])
                self.at_matrices[i]['model'].dataChanged.connect(
                                              self.apply_affine_transformation)
        self.update_shift_widget()
        self.slice_iv.updateImage()
        self.v_depth_iv.updateImage()
        self.h_depth_iv.updateImage()

    def connection_set_after_load(self):
        self.left_button.pressed.connect(self.shift_left)
        self.right_button.pressed.connect(self.shift_right)
        self.up_button.pressed.connect(self.shift_up)
        self.down_button.pressed.connect(self.shift_down)
        self.slice_lock.toggled.connect(self.lock_current_index)
        self.actionLoad_corrections.setEnabled(True)
        self.actionLoad_corrections.triggered.connect(self.load_shifts)
        self.actionGet_z_scale.setEnabled(True)
        self.actionGet_z_scale.triggered.connect(self.update_aspect_ratio)
        self.actionshow_markers.toggled.connect(self.visual_guides)
        self.v_slice_line.sigPositionChanged.connect(
                                                self.change_vertical_slice)
        self.h_slice_line.sigPositionChanged.connect(
                                                self.change_horizontal_slice)
        self.init_slice_btn.setEnabled(True)
        self.init_slice_btn.pressed.connect(self.init_at_at_index)
        self.reset_matrix.pressed.connect(self.remove_at_and_reset)
        self.actionSave_corrections.setEnabled(True)
        self.actionSave_corrections.triggered.connect(self.save_corrections)

    def visual_guides(self, status):
        self.align_line_x1.setVisible(status)
        self.align_line_x2.setVisible(status)
        self.align_line_y1.setVisible(status)
        self.align_line_y2.setVisible(status)
        self.roi_main.setVisible(status)

    def change_vertical_slice(self):
        pos = int(round(self.v_slice_line.pos()[0]))
        self.slice_yz = self.i_data[:, :, pos]
        self.v_depth_iv.setImage(self.slice_yz.T, autoRange=False,
                                 autoHistogramRange=False, autoLevels=False)

    def change_horizontal_slice(self):
        pos = int(round(self.h_slice_line.pos()[1]))
        self.slice_xz = self.i_data[:, pos, :]
        self.h_depth_iv.setImage(self.slice_xz, autoRange=False,
                                 autoHistogramRange=False, autoLevels=False)

    def lock_current_index(self, state):
        if state:
            self.locked_index = self.current_i
            self.locked_image_item = pg.ImageItem(
                                        self.i_data[self.locked_index, :, :])
            self.slice_iv.addItem(self.locked_image_item)
            self.slice_lock.setText(
                ' locked at index {}'.format(self.locked_index))
            self.transparency_slider.sliderMoved.connect(
                                        self.change_locked_image_opacity)
            self.change_locked_image_opacity(
                self.transparency_slider.value())
            self.horiz_line_2.hide()
            self.vert_line_2.hide()
            self.horiz_line_locked.setPos(self.locked_index)
            self.horiz_line_locked.show()
            self.vert_line_locked.setPos(self.locked_index)
            self.vert_line_locked.show()
        else:
            self.slice_lock.setText(
                  ' Lock onto current slice:')
            self.slice_iv.removeItem(self.locked_image_item)
            self.locked_image_item = None
            self.horiz_line_2.show()
            self.vert_line_2.show()
            self.horiz_line_locked.hide()
            self.vert_line_locked.hide()
            self.update_shift_widget()
            self.which_widget() 

    def change_locked_image_opacity(self, int_value):
        self.locked_image_item.setOpacity(int_value / 100)

    def save_cube(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName()
        if fn is None:
            return
        self.cube.save(fn)

    def crop_cube(self):
        if self.actionlock_onto_current_slice.isChecked():
            self.actionlock_onto_current_slice.setChecked(False)
        self.cube.crop(1, start=int(self.align_line_x1.pos()[0]),
                       end=int(self.align_line_x2.pos()[0]))
        self.cube.crop(2, start=int(self.align_line_y1.pos()[1]),
                       end=int(self.align_line_y2.pos()[1]))
        self.i_data = self.cube.data
        self.shifts = np.zeros(shape=(self.i_data.shape[0], 2),
                               dtype=np.int32)
        self.at_matrices = {}
        self.switch_to_simple_shift_widget()
        self.align_line_x1.setPos(0)
        self.align_line_x2.setPos(self.i_data.shape[2])
        self.align_line_y1.setPos(0)
        self.align_line_y2.setPos(self.i_data.shape[1])
        self.slice_iv.setImage(self.i_data)
        self.setup_slicers()

    def normalize(self):
        x1 = int(self.align_line_x1.pos()[0])
        x2 = int(self.align_line_x2.pos()[0])
        y1 = int(self.align_line_y1.pos()[1])
        y2 = int(self.align_line_y2.pos()[1])
        norma_arr = np.mean(self.i_data[:, y1:y2, x1:x2], axis=(1, 2))
        min_arr = int(norma_arr.min())
        normalization_arr = norma_arr - min_arr
        for i in range(self.i_data.shape[0]):
            self.i_data[i] = np.where(
                self.i_data[i] < int(normalization_arr[i] - 1),
                self.i_data[i],
                self.i_data[i] - int(normalization_arr[i] - 1))
        self.slice_iv.updateImage()
        self.v_depth_iv.updateImage()
        self.h_depth_iv.updateImage()

    def load_cube(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName()
        if fn is None or fn == "":
            return
        # clean up previous
        if self.actionlock_onto_current_slice.isChecked():
            self.actionlock_onto_current_slice.setChecked(False)
        self.cube = None
        self.i_data = None
        self.shifts = None
        self.slice_iv.clear()
        self.slice_xz = None
        self.slice_yz = None
        self.v_depth_iv.clear()
        self.h_depth_iv.clear()
        self.at_matrices = {}  # reset transformation matrices
        gc.collect()
        #
        self.setWindowTitle(path.basename(fn))
        self.cube = hs.load(fn)
        self.i_data = self.cube.data
        self.shifts = np.zeros(shape=(self.i_data.shape[0], 2))
        self.shifts = self.shifts.astype(np.int32)
        self.slice_iv.setImage(self.i_data)
        self.current_i = self.slice_iv.currentIndex
        self.current_slice_n.setText(str(self.current_i))
        self.main_image_item = self.slice_iv.getImageItem()
        self.main_image_item.sigImageChanged.connect(self.update_index)
        self.setup_slicers()
        if self.first_time_load:
            self.gen_align_lines()
            self.connection_set_after_load()
            self.actionSave.setEnabled(True)
            self.actionSave.triggered.connect(self.save_cube)
            self.actionCrop.setEnabled(True)
            self.actionCrop.triggered.connect(self.crop_cube)
            self.actionNormalize.triggered.connect(self.normalize)
            self.slices_spinbox.valueChanged.connect(
                self.update_index)
            self.first_time_load = False

    def setup_slicers(self):
        if self.first_time_load:
            self.horiz_line = pg.InfiniteLine(pos=0, angle=0,
                                              movable=False, pen='g')
            self.h_depth_iv.addItem(self.horiz_line)
            self.horiz_line_2 = pg.InfiniteLine(
                pos=1, angle=0, movable=False,
                pen=pg.mkPen('g', style=QtCore.Qt.DotLine))
            self.horiz_line_locked = pg.InfiniteLine(
                pos=0, angle=0, movable=False, pen='r')
            self.h_depth_iv.addItem(self.horiz_line_locked)
            self.horiz_line_locked.hide()
            self.h_depth_iv.addItem(self.horiz_line_2)
            self.vert_line = pg.InfiniteLine(pos=0, angle=90, movable=False,
                                             pen='g')
            self.vert_line_2 = pg.InfiniteLine(
                pos=1, angle=90, movable=False,
                pen=pg.mkPen('g', style=QtCore.Qt.DotLine))
            self.vert_line_locked = pg.InfiniteLine(
                pos=0, angle=90, movable=False, pen='r')
            self.v_depth_iv.addItem(self.vert_line_locked)
            self.vert_line_locked.hide()
            self.v_depth_iv.addItem(self.vert_line)
            self.v_depth_iv.addItem(self.vert_line_2)
            self.v_slice_line = pg.InfiniteLine(
                pos=self.i_data.shape[2] // 2, movable=True,
                pen='b', bounds=[0, self.i_data.shape[2] - 1])
            self.h_slice_line = pg.InfiniteLine(
                pos=self.i_data.shape[1] // 2, angle=0,
                movable=True, pen='b', bounds=[0, self.i_data.shape[1] - 1])
            self.slice_iv.addItem(self.v_slice_line)
            self.slice_iv.addItem(self.h_slice_line)
        else:
            self.v_slice_line.setPos(self.i_data.shape[2] // 2)
            self.v_slice_line.setBounds([0, self.i_data.shape[2] - 1])
            self.h_slice_line.setPos(self.i_data.shape[1] // 2)
            self.h_slice_line.setBounds([0, self.i_data.shape[1] - 1])
        self.slice_xz = self.i_data[:, self.i_data.shape[1] // 2, :]
        self.slice_yz = self.i_data[:, :, self.i_data.shape[2] // 2]
        self.v_depth_iv.setImage(self.slice_yz.T)
        self.h_depth_iv.setImage(self.slice_xz)

    def gen_align_lines(self):
        self.roi_main = pg.RectROI(pos=(20, 20), size=(980, 480))
        self.roi_main.setPen(pg.mkPen('#ffaa00'))
        self.slice_iv.addItem(self.roi_main)
        self.roi_main.hide()
        self.align_line_x1 = pg.InfiniteLine(angle=90, pos=20,
                                             movable=True, pen='y')
        self.align_line_x2 = pg.InfiniteLine(angle=90, pos=1000,
                                             movable=True, pen='y')
        self.h_depth_iv.addItem(self.align_line_x1)
        self.h_depth_iv.addItem(self.align_line_x2)
        self.align_line_y1 = pg.InfiniteLine(angle=0, pos=20,
                                             movable=True, pen='y')
        self.align_line_y2 = pg.InfiniteLine(angle=0, pos=500,
                                             movable=True, pen='y')
        self.v_depth_iv.addItem(self.align_line_y1)
        self.v_depth_iv.addItem(self.align_line_y2)
        self.visual_guides(False)
        self.actionshow_markers.setEnabled(True)
        self.actionshow_markers.setChecked(False)
        for i in ['x1', 'x2', 'y1', 'y2']:
            getattr(self,
                    'align_line_{}'.format(i)).sigPositionChanged.connect(
                                                            self.update_roi)
        self.roi_main.sigRegionChanged.connect(self.update_markers)

    def update_markers(self, roi):
        for i in ['x1', 'x2', 'y1', 'y2']:
            getattr(self,
                    'align_line_{}'.format(i)).sigPositionChanged.disconnect(
                                                            self.update_roi)
        x1, y1 = roi.pos()
        x2, y2 = roi.pos() + roi.size()
        self.align_line_x1.setPos(x1)
        self.align_line_x2.setPos(x2)
        self.align_line_y1.setPos(y1)
        self.align_line_y2.setPos(y2)
        for i in ['x1', 'x2', 'y1', 'y2']:
            getattr(self,
                    'align_line_{}'.format(i)).sigPositionChanged.connect(
                                                             self.update_roi)

    def update_roi(self):
        self.roi_main.sigRegionChanged.disconnect(self.update_markers)
        self.roi_main.setPos((self.align_line_x1.pos()[0],
                              self.align_line_y1.pos()[1]))
        self.roi_main.setSize(
            (self.align_line_x2.pos()[0] - self.align_line_x1.pos()[0],
             self.align_line_y2.pos()[1] - self.align_line_y1.pos()[1]))
        self.roi_main.sigRegionChanged.connect(self.update_markers)

    def update_index(self):
        # print('update index')
        index = self.current_i = self.slice_iv.currentIndex
        strenght = self.slices_spinbox.value()
        self.current_slice_n.setText(str(index))
        self.horiz_line.setPos(index)
        self.horiz_line_2.setPos(index + strenght)
        self.vert_line.setPos(index)
        self.vert_line_2.setPos(index + strenght)
        self.slices_spinbox.setMaximum(
            self.i_data.shape[0] - index)
        if not self.slice_lock.isChecked():
            self.update_shift_widget()
            self.which_widget() 

    def update_shift_widget(self):
        index = self.get_index_for_manipulation()
        self.label_x_correction.setText('x: {}'.format(self.shifts[index][0]))
        self.label_y_correction.setText('y: {}'.format(self.shifts[index][1]))

    def which_widget(self):
        index = self.get_index_for_manipulation()
        if self.simple_mode:
            if index in self.at_matrices:
                self.switch_to_matrix_correction_widget()
        else:
            if index in self.at_matrices:
                self.update_affine_widget()
            else:
                self.switch_to_simple_shift_widget()


    def update_affine_widget(self):
        index = self.get_index_for_manipulation()
        self.mtv.setModel(self.at_matrices[index]["model"])
    #    index = self.current_i
    #    if self.slice_lock.isChecked():
    #        return
    #    if index in self.at_matrices:
    #        if self.init_slice_btn.isChecked():
    #            self.set_at_at_index()
    #        else:
    #            self.init_slice_btn.setChecked(True)
    #    elif self.init_slice_btn.isChecked():
    #        self.init_slice_btn.setChecked(False)

    def update_aspect_ratio(self):
        z_val, ok_clicked = QtWidgets.QInputDialog.getDouble(
            self, "Get inter-slice interval",
            "distance in-between slices (in nm):", 10.05,
            0, 100,
            10)
        if ok_clicked:
            self.cube.axes_manager[0].scale = z_val
            self.cube.axes_manager[0].units = 'nm'
            x_val = self.cube.axes_manager[1].scale
            self.v_depth_iv.getView().setAspectLocked(True, ratio=z_val/x_val)
            self.h_depth_iv.getView().setAspectLocked(True, ratio=x_val/z_val)

    def update_images(self):
        self.slice_iv.updateImage()
        self.v_depth_iv.updateImage()
        self.h_depth_iv.updateImage()
        if self.slice_lock.isChecked():
            self.locked_image_item.updateImage()

    def apply_affine_transformation(self, index=None):
        """scipy.ndimage.affine_transform"""
        if not isinstance(index, int):
            index = self.get_index_for_manipulation()
        self.i_data[index] = affine_transform(
            self.at_matrices[index]['initial'],
            self.at_matrices[index]['matrix'])
        self.update_images()

    def shift_multi(self, strenght, axis):
        if self.slice_lock.isChecked():
            index = self.locked_index
            end = self.locked_index + 1
        else:
            index = self.current_i
            if self.to_end_checkbox.isChecked():
                end = self.i_data.shape[0]
            else:
                end = index + self.slices_spinbox.value()
        for i in range(index, end):
            self.i_data[i] = np.roll(self.i_data[i], strenght, axis=axis)
        if axis == 1:
            self.shifts[index:end, 0] += strenght
        elif axis == 0:
            self.shifts[index:end, 1] += strenght
        self.update_images()
        self.update_shift_widget()

    def shift_right(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(strenght, 1)

    def shift_left(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(-strenght, 1)

    def shift_up(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(-strenght, 0)

    def shift_down(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(strenght, 0)

    def show_about_qt(self):
        QtWidgets.QMessageBox.aboutQt(self)

    def show_about(self):
        QtWidgets.QMessageBox.about(self, "About this software",
                                    "Created by Petras Jokubauskas.\n"
                                    "This is free open source software "
                                    "licensed under GPLv3.\n\n"
                                    f"Current PyQt5 version: {QtCore.PYQT_VERSION_STR}")


class TransformationMatrixModel(QtCore.QAbstractTableModel):
    def __init__(self, dict_node):
        QtCore.QAbstractTableModel.__init__(self)
        dict_node['model'] = self
        self.data_view = dict_node['matrix'].view()

    def rowCount(self, *args):
        return 3

    def columnCount(self, *args):
        return 3

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                if section == 0:
                    return '(x)'
                if section == 1:
                    return '(y)'
                if section == 2:
                    return 't'
            if orientation == QtCore.Qt.Vertical:
                if section == 0:
                    return 'W'
                elif section == 1:
                    return 'H'
                elif section == 2:
                    return '(z)'

    def data(self, index, role):
        if not index.isValid():
            return False
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            return str(self.data_view[index.row()][index.column()])

        if role == QtCore.Qt.ToolTipRole:
            if index.column() == 2:
                if index.row() == 0:
                    return 'x'
                elif index.row() == 1:
                    return 'y'
            elif index.column() == 0:
                if index.row() == 0:
                    return 'width scaling/rotation(cos(theta))'
                elif index.row() == 1:
                    return 'rotation -sin(theta)/skew in y direction'
            elif index.column() == 1:
                if index.row() == 0:
                    return 'rotation sin(theta)/skew in x direction'
                elif index.row() == 1:
                    return 'height scaling/rotation(cos(theta))'
        if role == QtCore.Qt.BackgroundColorRole:
            if index.row() == index.column():
                return QtGui.QColor(23, 147, 100)

    def flags(self, index):
        if index.isValid():
            if index.row() == 2:
                return QtCore.Qt.ItemIsEnabled
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if role == QtCore.Qt.EditRole:
            self.data_view[index.row()][index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True


class SpinBoxDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        QtWidgets.QStyledItemDelegate.__init__(self, parent)

    def createEditor(self, parent, option, index):
        editor = QtWidgets.QDoubleSpinBox(parent)
        editor.setFrame(False)
        if index.column() < 2:
            editor.setMinimum(-10)
            editor.setMaximum(10)
            editor.setSingleStep(0.001)
            editor.setDecimals(3)
        else:
            editor.setMinimum(-1000)
            editor.setMaximum(1000)
            editor.setSingleStep(0.1)
            editor.setDecimals(2)
        return editor

    def setEditorData(self, spinBox, index):
        value = index.model().data(index, QtCore.Qt.EditRole)
        spinBox.setValue(float(value))

    def setModelData(self, editor, model, index):
        value = editor.value()
        model.setData(index, value, QtCore.Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = FIBSliceCorrector()
    main_window.show()
    app.exec()


if __name__ == "__main__":
    main()
