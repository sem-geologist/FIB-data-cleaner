import importlib
import sys
import json
import gc
import numpy as np
from os import path
from copy import copy
from PyQt5 import QtCore, QtGui, QtWidgets
import hyperspy.api as hs
import pyqtgraph as pg
from .ui import MainWindow
pg.setConfigOption('imageAxisOrder', 'row-major')
from cv2 import (getAffineTransform,
                 warpAffine,
                 INTER_NEAREST,
                 INTER_CUBIC)
from cv2 import __version__ as cv2__version__
from .ui.CustomComponents import (ZAlignmentROI,
                                  SpinBoxDelegate,
                                  TransformationMatrixModel)
__version__ = "0.1.0"

def get_nested(node, branches):
    """gets the item from nested dict, list, tuple or other
    iterable where its elements can be accesed with square brackets"""
    for b in branches:
        node = node[b]
    return node

class FIBSliceCorrector(QtWidgets.QMainWindow,
                        MainWindow.Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.first_time_load = True
        self.at_matrices = {}  # affine transformation matrices
        delegate = SpinBoxDelegate()
        self.mtv.setItemDelegate(delegate)
        self.actionLoad.triggered.connect(self.load_single_file_gui)
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
        self.reset_triangle.setIcon(pg.icons.default.qicon)
        kernel = self.console.kernel_manager.kernel
        if kernel is not None:
            kernel.shell.push(dict(np=np, pg=pg, hs=hs, app=self))
            self.dockConsole.setWidget(self.console)
            self.console.execute("whos")
        else:
            self.actionshow_python_console.setDisabled(True)
        self.dockConsole.hide()
        self.tv_metadata.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.multi_mtv.setVisible(False)
        self.pw_metadata.setVisible(False)
        # use some Qt built-in icons:
        pixmapi = QtWidgets.QStyle.SP_ArrowUp
        self.up_button.setIcon(self.style().standardIcon(pixmapi))
        pixmapi = QtWidgets.QStyle.SP_ArrowDown
        self.down_button.setIcon(self.style().standardIcon(pixmapi))
        pixmapi = QtWidgets.QStyle.SP_ArrowLeft
        self.left_button.setIcon(self.style().standardIcon(pixmapi))
        pixmapi = QtWidgets.QStyle.SP_ArrowRight
        self.right_button.setIcon(self.style().standardIcon(pixmapi))
        # use some pyqtgraph built-in icons:
        self.actionlock_onto_current_slice.toggled.connect(
            self.lock_current_index)
        self.actionlock_onto_current_slice.setIcon(pg.icons.lock.qicon)
        self.slice_lock.setDefaultAction(self.actionlock_onto_current_slice)
        self.reset_triangle.pressed.connect(self.reset_from_three)

    def get_full_slice(self):
        """get full slice selecting whole dimentions of data
        for further manipulatio of slice"""
        if self.i_data is None:
            raise ValueError("no data is loaded")
        return [slice(None)] * self.i_data.ndim

    def get_frame_slice(self, i_frame):
        i_slice = self.get_full_slice()
        i_slice[self.i_depth] = i_frame
        return (*i_slice,)

    def matrix_corection_mode_handler(self, mode):
        if mode == 0:
            self.pushMultiplyMatrices.setDisabled(True)
            try:
                self.pushMultiplyMatrices.pressed.disconnect()
            except TypeError:
                pass
            self.multi_mtv.setVisible(False)
            return
        self._multi_mx = {'matrix': np.identity(3, dtype=np.float32)}
        model = TransformationMatrixModel(self._multi_mx)
        self.multi_mtv.setVisible(True)
        self.multi_mtv.setModel(model)
        self.pushMultiplyMatrices.setEnabled(True)
        if mode == 1:  # pre-multiply
            self.pushMultiplyMatrices.pressed.connect(self.multi_pre_multi)
        elif mode == 2:
            self.pushMultiplyMatrices.pressed.connect(self.multi_post_multi)

    def multi_pre_multi(self):
        index, end = self.get_selection()
        self.setCursor(QtCore.Qt.WaitCursor)
        for i in range(index, end):
            if i not in self.at_matrices:
                self.init_at_at_index(i)
            self.pre_multiply_at(i, self._multi_mx['matrix'])
        self.unsetCursor()
        self.comboMatrixMode.setCurrentIndex(0)

    def multi_post_multi(self):
        index, end = self.get_selection()
        self.setCursor(QtCore.Qt.WaitCursor)
        for i in range(index, end):
            if i not in self.at_matrices:
                self.init_at_at_index(i)
            self.post_multiply_at(i, self._multi_mx['matrix'])
        self.unsetCursor()
        self.comboMatrixMode.setCurrentIndex(0)

    def get_index_for_manipulation(self):
        if self.slice_lock.isChecked():
            return self.locked_index
        return self.current_i

    def init_at_at_index(self, index_z=None):
        """generate and set initial affine transformation matrix
           for given index, also generate model, which can be
           viewed with QTableView"""
        current_flag = False
        if index_z is None:
            current_flag = True
            index_z = self.get_index_for_manipulation()
        if index_z not in self.at_matrices:
            self.at_matrices[index_z] = {
                'matrix': np.identity(3, dtype=np.float32)}
            i_slice = self.get_full_slice()
            x, y = (1, 0) if self.i_width > self.i_height else (0, 1)
            i_slice[self.i_depth] = index_z
            self.i_data[(*i_slice,)] = np.roll(self.i_data[(*i_slice,)],
                                               -self.shifts[index_z],
                                               axis=(x, y))
            TransformationMatrixModel(self.at_matrices[index_z])
            i_slice = self.get_frame_slice(index_z)
            self.at_matrices[index_z]['initial'] = copy(self.i_data[i_slice])
            self.at_matrices[index_z]['matrix'][:2, 2] = self.shifts[index_z]
            self.shifts[index_z][:] = 0, 0
            self.apply_affine_transformation(index_z)
            if current_flag:
                self.switch_to_matrix_correction_widget()
                self.update_images()

    def remove_at_and_reset(self):
        i = self.get_index_for_manipulation()
        i_slice = self.get_frame_slice(i)
        self.i_data[i_slice] = self.at_matrices[i]['initial']
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
        if self.show_ref_points.isChecked():
            self.show_ref_points.setChecked(False)
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
        i_slice = self.get_full_slice()
        x, y = (1, 0) if self.i_width > self.i_height else (0, 1)
        for i in range(self.i_data.shape[self.i_depth]):
            i_slice[self.i_depth] = i
            if i in self.at_matrices:
                mx = np.identity(3, np.float32)
                mx[:2, 2] = shifts[i]
                shifts[i][:] = 0, 0
                self.post_multiply_at(i, mx)
            else:
                self.i_data[(*i_slice,)] = np.roll(self.i_data[(*i_slice,)],
                                                   shifts[i],
                                                   axis=(x, y))

    def post_multiply_at(self, idx, mx):
        self.at_matrices[idx]["matrix"][:] = mx @ self.at_matrices[idx]["matrix"]
        self.apply_affine_transformation(idx)
        if idx == self.get_index_for_manipulation():
            self.update_affine_widget()

    def pre_multiply_at(self, idx, mx):
        self.at_matrices[idx]["matrix"][:] = self.at_matrices[idx]["matrix"] @ mx
        self.apply_affine_transformation(idx)
        if idx == self.get_index_for_manipulation():
            self.update_affine_widget()

    def save_corrections(self):
        at_jsonized = {str(i): self.at_matrices[i]['matrix'].tolist()
                       for i in self.at_matrices}
        roi_jsonized = {'pos': tuple(self.roi_main.pos()),
                        'size': tuple(self.roi_main.size())}
        jsonable_dict = {'shifts': self.shifts.tolist(),
                         'blacklist_mask': self.blacklist_mask.tolist(),
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
        """load and apply pixel shifts"""
        fn, ft = QtWidgets.QFileDialog.getOpenFileName(
            self, 'select shift transformation file', '../',
            "Shifts and affine transformation matrices in json (*.json);;"
            "Shifts as Numpy array (*.npy)")
        if ft == '':
            return
        if '.npy' in ft:
            shifts = np.load(fn).astype(np.int32)
            if len(shifts) != len(self.shifts):
                msg = QtWidgets.QMessageBox.warning(
                    self, "Shift loading failed",
                    "loaded shift lenght is not the same as the number of slices")
                msg.exec()
                return
            self.apply_shifts(-self.shifts)
            self.apply_shifts(shifts)
            self.shifts = shifts
        elif '.json' in ft:
            with open(fn, 'r') as file_pointer:
                corrections = json.load(file_pointer)
            shifts = np.asarray(corrections['shifts'], dtype=np.int32)
            if len(shifts) != len(self.shifts):
                msg = QtWidgets.QMessageBox.warning(
                    self, "Shift loading failed",
                    "loaded shift lenght is not the same as the number of slices")
                msg.exec()
                return
            if "blacklist_mask" in corrections:
                self.blacklist_mask = np.asarray(corrections['blacklist_mask'],
                                                 dtype=bool)
            self.apply_shifts(-self.shifts)
            self.apply_shifts(shifts)
            self.shifts = shifts
            if 'roi' in corrections:
                self.roi_main.setPos(corrections['roi']['pos'])
                self.roi_main.setSize(corrections['roi']['size'])
            at = {int(i): np.asarray(corrections['at'][i],
                                     dtype=np.float64)
                  for i in corrections['at']}
            i_slice = self.get_full_slice()
            for i in at:
                self.at_matrices[i] = {'matrix': at[i]}
                TransformationMatrixModel(self.at_matrices[i])
                i_slice[self.i_depth] = i
                self.at_matrices[i]['initial'] = copy(
                    self.i_data[(*i_slice,)])
                self.apply_affine_transformation(i)
                self.at_matrices[i]['model'].dataChanged.connect(
                    self.apply_affine_transformation)
        self.update_shift_widget()
        self.which_widget()
        self.slice_iv.updateImage()
        self.v_depth_iv.updateImage()
        self.h_depth_iv.updateImage()

    def connection_set_after_load(self):
        self.left_button.pressed.connect(self.shift_left)
        self.right_button.pressed.connect(self.shift_right)
        self.up_button.pressed.connect(self.shift_up)
        self.down_button.pressed.connect(self.shift_down)
        self.actionLoad_corrections.setEnabled(True)
        self.actionLoad_corrections.triggered.connect(self.load_shifts)
        self.actionGet_z_scale.setEnabled(True)
        self.actionGet_z_scale.triggered.connect(self.update_aspect_ratio)
        self.actionshow_markers.toggled.connect(self.visual_guides)
        self.v_slice_line.sigPositionChanged.connect(
                                                self.update_vertical_slice)
        self.h_slice_line.sigPositionChanged.connect(
                                                self.update_horizontal_slice)
        self.init_slice_btn.setEnabled(True)
        self.init_slice_btn.pressed.connect(self.init_at_at_index)
        self.reset_matrix.pressed.connect(self.remove_at_and_reset)
        self.actionSave_corrections.setEnabled(True)
        self.actionSave_corrections.triggered.connect(self.save_corrections)
        self.show_ref_points.toggled.connect(self.show_original_triangulation)
        self.show_ref_points.toggled.connect(self.enable_affine_by_tripoint)
        self.pin_points_to_slice.toggled.connect(
            self.enable_affine_by_triangle)

    def enable_affine_by_tripoint(self, state):
        if state:
            self.pin_points_to_slice.setEnabled(True)
        else:
            self.pin_points_to_slice.setChecked(False)
            self.pin_points_to_slice.setEnabled(False)

    def visual_guides(self, status):
        self.align_line_x1.setVisible(status)
        self.align_line_x2.setVisible(status)
        self.align_line_y1.setVisible(status)
        self.align_line_y2.setVisible(status)
        self.roi_main.setVisible(status)

    def update_vertical_slice(self):
        pos = int(round(self.v_slice_line.pos()[0]))
        i_slice = self.get_full_slice()
        i_slice[self.i_width] = pos
        self.slice_yz = self.i_data[(*i_slice,)]
        if self.i_depth < self.i_height:
            i_z, i_y = 0, 1
        else:
            i_z, i_y = 1, 0
        self.v_depth_iv.setImage(self.slice_yz,
                                 axes={'x': i_z, 'y': i_y},
                                 autoRange=False,
                                 autoHistogramRange=False,
                                 autoLevels=False)

    def update_horizontal_slice(self):
        pos = int(round(self.h_slice_line.pos()[1]))
        i_slice = self.get_full_slice()
        i_slice[self.i_height] = pos
        self.slice_xz = self.i_data[(*i_slice,)]
        i_z, i_x = (0, 1) if self.i_depth < self.i_width else (1, 0)
        self.h_depth_iv.setImage(self.slice_xz,
                                 axes={'x': i_x, 'y': i_z},
                                 autoRange=False,
                                 autoHistogramRange=False,
                                 autoLevels=False)

    def lock_current_index(self, state):
        """
        lock the current index of slice making a overlay ImageItem
        and connect its opacity to slider"""
        if state:
            self.transparency_slider.setEnabled(True)
            self.locked_index = self.current_i
            i_slice = self.get_frame_slice(self.locked_index)
            self.locked_image_item = pg.ImageItem(self.i_data[i_slice])
            self.slice_iv.addItem(self.locked_image_item)
            self.lock_text.setText(
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
            self.lock_text.setText(
                ' Lock current slice')
            self.slice_iv.removeItem(self.locked_image_item)
            self.locked_image_item = None
            self.horiz_line_2.show()
            self.vert_line_2.show()
            self.horiz_line_locked.hide()
            self.vert_line_locked.hide()
            self.transparency_slider.setEnabled(False)
            self.update_shift_widget()
            self.which_widget()
            self.check_box_blacklisted.setChecked(bool(
                self.blacklist_mask[self.get_index_for_manipulation()]))

    def change_locked_image_opacity(self, int_value):
        self.locked_image_item.setOpacity(int_value / 100)

    def enable_affine_by_triangle(self, state):
        """should not be called with same state subsequently;
        should be called by Q*Button signal which emits button
        checked state."""
        for i in self.original_ref:
            i.movable = not state
        if state:
            index = self.get_index_for_manipulation()
            self._original_mx = self.at_matrices[index]['matrix'].copy()
            self._temp_mx = np.identity(3, np.float32)
            points = [(j.x(), j.y()) for j in self.original_ref]
            self.floating_triangle = pg.PolyLineROI(positions=points,
                                                    closed=True, )
            self.slice_iv.addItem(self.floating_triangle)
            self.floating_triangle.sigRegionChanged.connect(
                self.at_from_three)
        else:
            self.floating_triangle.sigRegionChanged.disconnect(
                self.at_from_three)
            self.slice_iv.removeItem(self.floating_triangle)
        self.reset_triangle.setEnabled(state)
        self.comboMatrixMode.setDisabled(state)

    def at_from_three(self):
        """affine transformation from one of point or whole
        tringle movement"""
        index = self.get_index_for_manipulation()
        state = self.floating_triangle.getState()
        pos_offset = state['pos']
        vertices = state['points']
        h_list = np.array([tuple(h + pos_offset) for h in vertices],
                          dtype=np.float32)
        p_list = np.array([tuple(p.pos()) for p in self.original_ref],
                          dtype=np.float32)
        self._temp_mx[:2] = getAffineTransform(p_list, h_list)
        self.at_matrices[index]['matrix'][:] = self._temp_mx @ self._original_mx
        self.apply_affine_transformation(index, fast=True)
        self.update_affine_widget()

    def reset_from_three(self):
        index = self.get_index_for_manipulation()
        self.at_matrices[index]['matrix'][:] = self._original_mx
        self.apply_affine_transformation(index, fast=False)
        self.update_affine_widget()
        self.pin_points_to_slice.setChecked(False)

    def save_cube(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName()
        if fn is None:
            return
        self.setCursor(QtCore.Qt.WaitCursor)
        self.cube.save(fn)
        self.unsetCursor()

    def export_with_hs(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "export with raw-like intensities; give file name and directory",
            None,
            "ImageIO supported Images (*.jpg *.xpm *.png)"
        )
        if fn is None:
            return
        self.setCursor(QtCore.Qt.WaitCursor)
        for i, img in enumerate(self.cube):
            img.save(f"_{i:04d}.".join(fn.rsplit(".", 1)))
        self.unsetCursor()

    def export_with_pg(self):
        """with applied LUT"""
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "export with applied LUT; give a file name and directory",
            None,
            "Qt supported Images (*.bmp *.jpg *.jpeg *.png *.ppm *.xbm *.xpm)")
        if fn is None:
            return
        self.setCursor(QtCore.Qt.WaitCursor)
        self.slice_iv.export(fn)
        self.unsetCursor()

    def crop_cube(self):
        """crop and reload hspy cube to the ROI and blacklist mask
        Please keep in ming that this method overwrites cube.data with
        .data of cropped cube, that releases not cropped array from memory.
        it is incontrast to original Hyperspy implementation
        wchich keeps the reference of original not-cropped data under
        .base attribute preventing memory freeing."""
        if self.actionlock_onto_current_slice.isChecked():
            self.actionlock_onto_current_slice.setChecked(False)
        self.cube.crop("width", start=int(self.align_line_x1.pos()[0]),
                       end=int(self.align_line_x2.pos()[0]))
        self.cube.crop("height", start=int(self.align_line_y1.pos()[1]),
                       end=int(self.align_line_y2.pos()[1]))
        b_slices = len(self.blacklist_mask.nonzero()[0])
        if b_slices > 0:
            n_slices = len((~self.blacklist_mask).nonzero()[0])
            data_sel = self.get_full_slice()
            data_black = data_sel.copy()
            data_sel[self.i_depth] = slice(0, n_slices, 1)
            data_black[self.i_depth] = ~self.blacklist_mask
            self.cube.data[(*data_sel,)] = self.cube.data[(*data_black,)]
            self.cube.crop(self.i_hspy_depth, end=n_slices)
        self.cube.data = self.cube.data.copy()  # released croped data
        self.load_hspy_signal(self.cube)
        self.switch_to_simple_shift_widget()
        self.align_line_x1.setPos(0)
        self.align_line_x2.setPos(self.i_data.shape[self.i_width])
        self.align_line_y1.setPos(0)
        self.align_line_y2.setPos(self.i_data.shape[self.i_height])

    def normalize(self):
        x1 = int(self.align_line_x1.pos()[0])
        x2 = int(self.align_line_x2.pos()[0])
        y1 = int(self.align_line_y1.pos()[1])
        y2 = int(self.align_line_y2.pos()[1])

        i_slice = self.get_full_slice()
        i_slice[self.i_width] = slice(x1, x2)
        i_slice[self.i_height] = slice(y1, y2)
        norma_arr = np.mean(self.i_data[(*i_slice,)],
                            axis=(self.i_height, self.i_width))
        min_arr = int(norma_arr.min())
        normalization_arr = norma_arr - min_arr
        i_slice = self.get_full_slice()
        for i in range(self.i_data.shape[self.i_depth]):
            i_slice = self.get_frame_slice(i)
            self.i_data[i_slice] = np.where(
                self.i_data[i_slice] < int(normalization_arr[i] - 1),
                self.i_data[i_slice],
                self.i_data[i_slice] - int(normalization_arr[i] - 1))
        self.slice_iv.updateImage()
        self.v_depth_iv.updateImage()
        self.h_depth_iv.updateImage()

    def load_single_file_gui(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName()
        if fn is None or fn == "":
            return
        self.unload_file()
        self.load_file(fn)

    def load_file(self, fn):
        self.setCursor(QtCore.Qt.WaitCursor)
        signal = hs.load(fn)
        if type(signal) == list:
            str_signals = [str(i) for i in signal]
            selection = QtWidgets.QInputDialog.getItem(
                self,
                "Signal",
                "Few signals were found inside the file;\n"
                "please, choose one to proceed further", 
                str_signals)
            if selection[1]:
                index = str_signals.index(selection[0])
                signal = signal[index]
            else:
                self.unsetCursor()
                return
        self.load_hspy_signal(signal)
        self.unsetCursor()

    def unload_file(self):
        if self.actionlock_onto_current_slice.isChecked():
            self.actionlock_onto_current_slice.setChecked(False)
        self.cube = None
        self.i_data = None
        self.shifts = None
        self.check_box_blacklisted.setChecked(False)
        self.blacklist_mask = None
        self.slice_iv.clear()
        self.slice_xz = None
        self.slice_yz = None
        self.v_depth_iv.clear()
        self.h_depth_iv.clear()
        self.at_matrices = {}  # reset transformation matrices
        self.actionVertical_shift_guide.setChecked(False)
        self.actionHorizontal_shift_guide.setChecked(False)
        gc.collect()

    def set_index_by_name(self, hspy_sig):
        hd = hspy_sig.axes_manager.as_dictionary()
        n_axes = len(hd)
        for i in range(n_axes):
            if hspy_sig.axes_manager[i].name == 'width':
                self.i_width = hspy_sig.axes_manager[i].index_in_array
                self.i_hspy_width = i
            elif hspy_sig.axes_manager[i].name == 'height':
                self.i_height = hspy_sig.axes_manager[i].index_in_array
            elif n_axes == 3:
                self.i_depth = hspy_sig.axes_manager[i].index_in_array
                self.i_hspy_depth = i
            else:
                raise TypeError("data with more than 3 dimensions currently not supported")

    def load_hspy_signal(self, signal):
        self.unload_file()
        self.setWindowTitle(signal.metadata.General.original_filename)
        self.cube = signal
        self.set_index_by_name(signal)
        self.i_data = self.cube.data
        self.shifts = np.zeros(shape=(self.i_data.shape[self.i_depth], 2))
        self.shifts = self.shifts.astype(np.int32)
        self.blacklist_mask = np.full((self.i_data.shape[self.i_depth]),
                                      False,
                                      dtype=bool)
        self.slice_iv.setImage(self.i_data, axes={'t': self.i_depth,
                                                  'x': self.i_width,
                                                  'y': self.i_height})
        self.current_i = self.slice_iv.currentIndex
        self.current_slice_n.setText(str(self.current_i))
        self.main_image_item = self.slice_iv.getImageItem()
        self.main_image_item.sigImageChanged.connect(self.update_index)
        self.setup_slicers()
        self.setup_metadata_stack_gui()

        if self.first_time_load:
            self.gen_align_lines()
            self.plot_widget_line = self.pw_metadata.plotItem.addLine(x=0)
            self.pw_metadata.plotItem.addLegend()
            self.connection_set_after_load()
            self.actionSave.setEnabled(True)
            self.actionSave.triggered.connect(self.save_cube)
            self.actionCrop.setEnabled(True)
            self.actionExport_with_hs.setEnabled(True)
            self.actionExport_with_pg.setEnabled(True)
            self.actionExport_with_hs.triggered.connect(
                self.export_with_hs)
            self.actionExport_with_pg.triggered.connect(
                self.export_with_pg)
            self.check_box_blacklisted.stateChanged.connect(
                self.black_list_current_slice)
            self.actionCrop.triggered.connect(self.crop_cube)
            self.actionNormalize.triggered.connect(self.normalize)
            self.actionVertical_shift_guide.toggled.connect(
                self.graphic_v_guide)
            self.actionHorizontal_shift_guide.toggled.connect(
                self.graphic_h_guide)
            self.slices_spinbox.valueChanged.connect(
                self.update_index)
            self.v_roi_line_btn = QtWidgets.QPushButton(
                "aligning to Polyline ROI")
            self.v_depth_iv.ui.gridLayout.addWidget(self.v_roi_line_btn)
            self.v_roi_line_btn.setVisible(False)
            self.v_roi_line_btn.pressed.connect(self.align_to_v_guide_roi)
            self.h_roi_line_btn = QtWidgets.QPushButton(
                "aligning to Polyline ROI")
            self.h_depth_iv.ui.gridLayout.addWidget(self.h_roi_line_btn)
            self.h_roi_line_btn.setVisible(False)
            self.h_roi_line_btn.pressed.connect(self.align_to_h_guide_roi)
            self.first_time_load = False
            self.comboMatrixMode.currentIndexChanged.connect(
                self.matrix_corection_mode_handler)
            self.tv_metadata.itemSelectionChanged.connect(
                self.plot_selected_metadata)
            self.cb_treeview_source.currentIndexChanged.connect(
                self.change_plot_metadata_source)
            self.pb_destroy_stack_elements.pressed.connect(
                self.discard_stack_elements)
            self.pb_stack_in_arrays.pressed.connect(
                self.stack_selected_metadata)

    def change_plot_metadata_source(self, index):
        if index == 0:
            self.tv_metadata.setData(
                self.cube.metadata.as_dictionary(),
                hideRoot=True)
            self.tv_metadata.collapseAll()
            self.pb_stack_in_arrays.setEnabled(False)
            self.pw_metadata.setVisible(False)
        elif index == 1:
            try:
                self.tv_metadata.setData(
                    self.cube.original_metadata.stack_elements.element0.as_dictionary(),
                    hideRoot=True)
                self.pw_metadata.setVisible(True)
                self.pb_destroy_stack_elements.setEnabled(True)
                self.pb_stack_in_arrays.setEnabled(True)
            except AttributeError:
                self.tv_metadata.setData({})
                self.pw_metadata.setVisible(False)
            self.tv_metadata.collapseAll()
        else:
            self.tv_metadata.setData(
                self.cube.original_metadata.stack_in_arrays.as_dictionary(),
                hideRoot=True)
            self.tv_metadata.collapseAll()
            self.pw_metadata.setVisible(True)
            self.pb_stack_in_arrays.setEnabled(False)

    def setup_metadata_stack_gui(self):
        if "stack_elements" in self.cube.original_metadata:
            self.cb_treeview_source.setEnabled(True)
            self.cb_treeview_source.setCurrentIndex(1)
            self.change_plot_metadata_source(1)
            if "stack_in_arrays" not in self.cube.original_metadata:
                self.cube.original_metadata.add_node("stack_in_arrays")
        elif "stack_in_arrays" in self.cube.original_metadata:
            self.cb_treeview_source.setCurrentIndex(2)
            self.cb_treeview_source.setEnabled(True)
            self.change_plot_metadata_source(2)
            self.pb_destroy_stack_elements.setEnabled(False)
            self.tv_metadata.itemSelectionChanged.connect(
                self.plot_selected_metadata)
        else:
            # self.tv_metadata.setEnabled(False)
            self.pw_metadata.setVisible(False)
            self.change_plot_metadata_source(0)
            self.cb_treeview_source.setEnabled(False)
            self.pb_destroy_stack_elements.setEnabled(False)
            self.pb_stack_in_arrays.setEnabled(False)
            # self.tv_metadata.setData({})

    def discard_stack_elements(self):
        """strip up Hyperspy signal from stacked original metadata,
        which significantly reduces file size, and makes loading
        and saving speeds few orders faster"""
        if "stack_elements" in self.cube.original_metadata:
            del self.cube.original_metadata.stack_elements
            self.change_plot_metadata_source(1)
            self.cb_treeview_source.setEnabled(False)

    def stack_selected_metadata(self):
        """consolidate selected metadata across slices into single
        array"""
        selected_items = self.tv_metadata.selectedItems()
        om = self.cube.original_metadata
        valid_items = [item for item in selected_items
                       if item.data(1, 0) in ['int', 'float']]
        if len(valid_items) > 0:
            paths = [item for item in self.tv_metadata.nodes.items()
                     if item[1] in valid_items]
            n_slices = self.slice_iv.nframes()
            for j, p in enumerate(paths):
                plot_data = [
                    get_nested(om.stack_elements[f"element{i}"], p[0])
                    for i in range(n_slices)]
                dtb_path, ok = QtWidgets.QInputDialog.getText(
                    self,
                    f"enter raplacement name/path for {p[0]}",
                    "can be defined as path with dot as separation:",
                    QtWidgets.QLineEdit.EchoMode.Normal,
                    ".".join([str(pa) for pa in p[0]]))
                self.cube.original_metadata.stack_in_arrays.set_item(
                    dtb_path, np.array(plot_data))

    def plot_selected_metadata(self):
        # currently metadata contains no arrays, those there is no need to
        # plot common metadata:
        if self.cb_treeview_source.currentIndex() == 0:
            return
        selected_items = self.tv_metadata.selectedItems()
        om = self.cube.original_metadata
        if self.cb_treeview_source.currentIndex() == 1:
            valid_items = [item for item in selected_items
                           if item.data(1, 0) in ['int', 'float']]
        else:
            valid_items = [item for item in selected_items
                           if item.data(1, 0) == 'ndarray']
        if len(valid_items) > 0:
            self.pw_metadata.plotItem.clear()
            paths = [item for item in self.tv_metadata.nodes.items()
                     if item[1] in valid_items]
            n_slices = self.slice_iv.nframes()
            if self.cb_treeview_source.currentIndex() == 1:
                for j, p in enumerate(paths):
                    plot_data = [
                        get_nested(om.stack_elements[f"element{i}"], p[0])
                        for i in range(n_slices)]
                    plotitem = pg.PlotDataItem(plot_data,
                                               name="{0}/{1}".format(*p[0][-2:]),
                                               pen=pg.Color(j))
                    self.pw_metadata.addItem(plotitem)
            else:
                for j, p in enumerate(paths):
                    plot_data = om.stack_in_arrays.get_item(".".join(p[0]))
                    plotitem = pg.PlotDataItem(plot_data,
                                               name=".".join(p[0]),
                                               pen=pg.Color(j))
                    self.pw_metadata.addItem(plotitem)

            self.pw_metadata.addItem(self.plot_widget_line)

    def black_list_current_slice(self, check_state):
        index = self.get_index_for_manipulation()
        self.blacklist_mask[index] = True if check_state else False

    def graphic_v_guide(self, state):
        self.v_roi_line_btn.setVisible(state)
        if state:
            self.v_guide_roi = ZAlignmentROI(self, orientation="h")
            self.v_depth_iv.addItem(self.v_guide_roi)
        else:
            self.v_depth_iv.removeItem(self.v_guide_roi)
            self.v_guide_roi = None

    def align_to_v_guide_roi(self):
        shifts = self.v_guide_roi.getShifts()
        self.actionVertical_shift_guide.setChecked(False)
        self.apply_shifts(shifts)
        self.shifts += shifts
        self.update_images()

    def graphic_h_guide(self, state):
        self.h_roi_line_btn.setVisible(state)
        if state:
            self.h_guide_roi = ZAlignmentROI(self, orientation="v")
            self.h_depth_iv.addItem(self.h_guide_roi)
        else:
            self.h_depth_iv.removeItem(self.h_guide_roi)
            self.h_guide_roi = None

    def align_to_h_guide_roi(self):
        shifts = self.h_guide_roi.getShifts()
        self.actionHorizontal_shift_guide.setChecked(False)
        self.apply_shifts(shifts)
        self.shifts += shifts
        self.update_images()

    def setup_slicers(self):
        v_bound = self.i_data.shape[self.i_width]
        h_bound = self.i_data.shape[self.i_height]
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
                pos=v_bound // 2, movable=True,
                pen='b', bounds=[0, v_bound - 1])
            self.h_slice_line = pg.InfiniteLine(
                pos=h_bound // 2, angle=0,
                movable=True, pen='b',
                bounds=[0, h_bound - 1])
            self.slice_iv.addItem(self.v_slice_line)
            self.slice_iv.addItem(self.h_slice_line)
        else:
            self.v_slice_line.setPos(v_bound // 2)
            self.v_slice_line.setBounds([0, v_bound - 1])
            self.h_slice_line.setPos(h_bound // 2)
            self.h_slice_line.setBounds([0, h_bound - 1])
        xz_slice = self.get_full_slice()
        xz_slice[self.i_height] = h_bound // 2
        self.slice_xz = self.i_data[(*xz_slice,)]
        yz_slice = self.get_full_slice()
        yz_slice[self.i_width] = v_bound // 2
        self.slice_yz = self.i_data[(*yz_slice,)]
        if self.i_depth < self.i_height:
            i_z, i_y = 0, 1
        else:
            i_z, i_y = 1, 0
        self.v_depth_iv.setImage(self.slice_yz,
                                 axes={'x': i_z, 'y': i_y})
        if self.i_depth < self.i_width:
            i_z, i_x = 0, 1
        else:
            i_z, i_x = 1, 0
        self.h_depth_iv.setImage(self.slice_xz,
                                 axes={'x': i_x, 'y': i_z})
        self.gen_original_triangulation_ref()

    def gen_original_triangulation_ref(self):
        x1 = self.i_data.shape[self.i_width] // 10
        x2 = x1 * 9
        y1 = self.i_data.shape[self.i_height] // 10
        y2 = y1 * 9
        self.original_ref = [pg.TargetItem(pos=(x1, y1)),
                             pg.TargetItem(pos=(x2, y1)),
                             pg.TargetItem(pos=(x2, y2))]

    def show_original_triangulation(self, state):
        if state:
            for i in self.original_ref:
                self.slice_iv.addItem(i)
        else:
            for i in self.original_ref:
                self.slice_iv.removeItem(i)

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
            self.i_data.shape[self.i_depth] - index)
        self.plot_widget_line.setPos(index)
        if not self.slice_lock.isChecked():
            self.update_shift_widget()
            self.which_widget()
            self.check_box_blacklisted.setChecked(bool(self.blacklist_mask[index]))

    def update_shift_widget(self):
        index = self.get_index_for_manipulation()
        self.label_x_correction.setText(f"x: {self.shifts[index][0]}")
        self.label_y_correction.setText(f"y: {self.shifts[index][1]}")

    def which_widget(self):
        index = self.get_index_for_manipulation()
        if self.simple_mode:
            if index in self.at_matrices:
                self.switch_to_matrix_correction_widget()
            else:
                self.switch_to_simple_shift_widget()
        else:
            if index in self.at_matrices:
                self.update_affine_widget()
            else:
                self.switch_to_simple_shift_widget()

    def update_affine_widget(self):
        index = self.get_index_for_manipulation()
        model = self.at_matrices[index]["model"]
        self.mtv.setModel(model)
        for i in range(2):
            for j in range(3):
                self.mtv.update(model.index(i, j))

    def update_aspect_ratio(self):
        i_depth_hspy = self.i_hspy_depth
        i_width_hspy = self.i_hspy_width
        init_val = self.cube.axes_manager[i_depth_hspy].scale
        z_val, ok_clicked = QtWidgets.QInputDialog.getDouble(
            self,
            "set interval size",
            "set inter-slice length\n"
            "(changes the aspect ratio for (z,y) and (x,z) plots)\n"
            "the distance in-between slices (in nm):",
            init_val,
            0, 500,
            10)
        if ok_clicked:
            self.cube.axes_manager[i_depth_hspy].units = 'nm'
            self.cube.axes_manager[i_depth_hspy].scale = z_val
            z_val = self.cube.axes_manager[i_depth_hspy].scale_as_quantity
            x_val = self.cube.axes_manager[i_width_hspy].scale_as_quantity
            self.v_depth_iv.getView().setAspectLocked(True,
                                                      ratio=float(z_val/x_val))
            self.h_depth_iv.getView().setAspectLocked(True,
                                                      ratio=float(x_val/z_val))

    def update_images(self):
        self.slice_iv.updateImage()
        self.v_depth_iv.updateImage()
        self.h_depth_iv.updateImage()
        if self.slice_lock.isChecked():
            self.locked_image_item.updateImage()

    def apply_affine_transformation(self, index=None, fast=False):
        """kwords: index=None, fast=False;
        if index is not integer (including None) then current slice index
        is used;
        if fast=True, then faster INTER_NEAREST interpolation is used,
        if fast=False then more precise INTER_CUBIC opencv interpolation
        is used."""
        if fast:
            interp_mode = INTER_NEAREST
        else:
            interp_mode = INTER_CUBIC
        if not isinstance(index, int):
            index = self.get_index_for_manipulation()
        i_slice = self.get_frame_slice(index)
        self.i_data[i_slice] = warpAffine(
            self.at_matrices[index]['initial'],
            self.at_matrices[index]['matrix'][:2],
            (self.i_data.shape[self.i_width],
             self.i_data.shape[self.i_height]),
            interp_mode)
        self.update_images()

    def get_selection(self):
        if self.slice_lock.isChecked():
            start = self.locked_index
            end = self.locked_index + 1
        else:
            start = self.current_i
            if self.to_end_checkbox.isChecked():
                end = self.i_data.shape[self.i_depth]
            else:
                end = start + self.slices_spinbox.value()
        return start, end

    def shift_multi(self, strenght, axis):
        index, end = self.get_selection()
        for i in range(index, end):
            self.i_data[self.get_frame_slice(i)] = np.roll(
                self.i_data[self.get_frame_slice(i)], strenght, axis=axis)
        x, y = (0, 1) if self.i_width > self.i_height else (1, 0)
        if axis == 1:
            self.shifts[index:end, x] += strenght
        elif axis == 0:
            self.shifts[index:end, y] += strenght
        self.update_images()
        self.update_shift_widget()

    def shift_right(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(strenght,
                         1 if self.i_width > self.i_height else 0)

    def shift_left(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(-strenght,
                         1 if self.i_width > self.i_height else 0)

    def shift_up(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(-strenght,
                         0 if self.i_width > self.i_height else 1)

    def shift_down(self):
        strenght = self.strenght_spinbox.value()
        self.shift_multi(strenght,
                         0 if self.i_width > self.i_height else 1)

    def show_about_qt(self):
        QtWidgets.QMessageBox.aboutQt(self)

    def show_about(self):
        QtWidgets.QMessageBox.about(self, "About this software",
                                    "Created by Petras Jokubauskas.\n"
                                    "This is free open source software "
                                    "licensed under GPLv3.\n\n"
                                    f"Current version: {__version__}\n\n"
                                    "Library versions:\n"
                                    f" PyQt5: {QtCore.PYQT_VERSION_STR}\n"
                                    f" PyQtGraph: {pg.__version__}\n"
                                    f" HypersPy: {hs.__version__}\n"
                                    f" OpenCV: {cv2__version__}\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print(f"FIB-SEM Data Cleaner version {__version__}\n\n"
              "Usage: fibdatacleaner [file]    launch and load specified file\n"
              "   or: fibdatacleaner           launch without loading anything")
        return 
    app = QtWidgets.QApplication(sys.argv)
    main_window = FIBSliceCorrector()
    main_window.show()
    if len(sys.argv) == 2:
        main_window.load_file(sys.argv[1])
    app.exec()


if __name__ == "__main__":
    main()
